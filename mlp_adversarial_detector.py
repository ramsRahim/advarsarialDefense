import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import clip
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import argparse
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
# Import needed functions from existing code
from utils import build_test_data_loader, clip_classifier

# FGSM attack implementation
def fgsm_attack(model, images, labels, epsilon, text_features):
    images.requires_grad = True
    
    # Forward pass
    image_features = model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logits = 100.0 * image_features @ text_features
    
    # Calculate loss
    loss = F.cross_entropy(logits, labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Create perturbation
    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()
    
    # Create adversarial example
    perturbed_image = images + epsilon * sign_data_grad
    
    # Clamp to ensure valid pixel range [0,1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image

# Feature extractor class
class FeatureExtractor:
    def __init__(self, model, text_features):
        self.model = model
        self.text_features = text_features
        self.device = next(model.parameters()).device
        
        # Reference distribution parameters for LID
        self.feature_bank = []
        self.nn_model = None
        self.k_neighbors = 20
    
    def extract_features1(self, images, labels, is_clean=True, update_reference=True):
        """Extract enhanced detection features"""
        with torch.no_grad():
            # Get original features
            image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ self.text_features
            
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            is_correct = (preds == labels)
            
            # Basic features
            confidence = probs.max(dim=1)[0]
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            
            # Enhanced features
            top2_confidence = torch.topk(probs, 2, dim=1)[0]
            confidence_margin = top2_confidence[:, 0] - top2_confidence[:, 1]
            predicted_class_prob = torch.gather(probs, 1, labels.unsqueeze(1)).squeeze()
            
            # Update reference if clean and correct
            if is_clean and update_reference and is_correct.any():
                self._update_reference_distribution(image_features, is_correct)
            
            # Compute LID
            lid = self._compute_lid(image_features)
            
            return {
                'confidence': confidence,
                'entropy': entropy,
                'confidence_margin': confidence_margin,
                'predicted_class_prob': predicted_class_prob,
                'lid': lid,
                'is_correct': is_correct
            }
            
    def extract_features(self, images, labels, is_clean=True, update_reference=True):
        """Extract detection features from images"""
        with torch.no_grad():
            # Get features and logits
            image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ self.text_features
            
            # Get predictions and probabilities
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            # Determine which predictions are correct
            is_correct = (preds == labels)
            
            # Compute confidence (max probability)
            confidence = probs.max(dim=1)[0]
            
            # Compute entropy - MORE STABLE VERSION
            entropy = -torch.sum(torch.where(
                probs > 1e-6,  # Only consider non-zero probabilities 
                probs * torch.log(torch.clamp(probs, min=1e-6)),  # Use clamp instead of addition
                torch.zeros_like(probs)  # 0 * log(0) = 0 in entropy calculation
            ), dim=1)
            
            # If this is clean data and we're supposed to update reference
            if is_clean and update_reference and is_correct.any():
                self._update_reference_distribution(image_features, is_correct)
                
            # Compute LID
            #lid = self._compute_lid(image_features)
            
            # Replace NaN/Inf values before returning
            confidence = torch.nan_to_num(confidence, nan=0.5)
            entropy = torch.nan_to_num(entropy, nan=0.0)
            #lid = torch.nan_to_num(lid, nan=0.0, posinf=1000.0)
            
            # Return all features
            return {
                'confidence': confidence,
                'entropy': entropy,
                #'lid': lid,
                'is_correct': is_correct
            }
        
    def _update_reference_distribution(self, features, is_correct):
        """Update reference distribution with new clean features."""
        clean_features = features[is_correct].cpu().numpy()
        
        # Update feature bank for LID
        self.feature_bank.append(clean_features)
        total_features = np.concatenate(self.feature_bank, axis=0)
        
        # If we have enough samples, rebuild nearest neighbors model
        if len(total_features) > self.k_neighbors + 1:
            if len(total_features) > 10000:
                # Subsample if too many features
                indices = np.random.choice(len(total_features), 10000, replace=False)
                total_features = total_features[indices]
            
            self.nn_model = NearestNeighbors(n_neighbors=self.k_neighbors+1).fit(total_features)
    
    def _compute_lid(self, features):
        """Compute Local Intrinsic Dimensionality."""
        if self.nn_model is None:
            # Return a large value if nearest neighbors model not established
            return torch.ones(features.size(0), device=features.device) * float('inf')
        
        features_np = features.cpu().numpy()
        
        try:
            # Get distances to k nearest neighbors
            distances, _ = self.nn_model.kneighbors(features_np)
            
            # Exclude the first distance (distance to self = 0)
            distances = distances[:, 1:]
            
            # Compute LID: -1 / mean(log(r_i / r_max))
            r_max = distances[:, -1:]
            lid_scores = -1.0 / np.mean(np.log(distances / (r_max + 1e-12)), axis=1)
            
            return torch.tensor(lid_scores, device=features.device)
        except:
            # If computation fails, return a large value
            return torch.ones(features.size(0), device=features.device) * float('inf')

# MLP model for classification
class AdversarialClassifier(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[128, 64, 32]):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.clamp(self.model(x), min=1e-7, max=1.0-1e-7)

# Collect data and features
def collect_features(model, text_features, data_loader, num_samples=1000, epsilon=0.03):
    # Create feature extractor
    extractor = FeatureExtractor(model, text_features)
    device = next(model.parameters()).device
    
    # Storage for features
    clean_features = []
    adv_features = []
    
    # First process clean samples to build reference distribution
    print("Processing clean samples...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(data_loader)):
            if i >= num_samples:
                break
            
            # Move to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Extract features (this will update reference distribution)
            features = extractor.extract_features(images, labels, is_clean=True)
            
            # Create feature tensor with 3 features (excluding Mahalanobis)
            feature_tensor = torch.stack([
                features['confidence'],
                features['entropy'],
                #features['lid']
            ], dim=1)
            
            # Replace inf/NaN values with large finite values
            feature_tensor = torch.nan_to_num(feature_tensor, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Store feature vector
            clean_features.append(feature_tensor)
    
    # Now generate and process adversarial samples
    print("Generating and processing adversarial samples...")
    data_iter = iter(data_loader)
    for i in range(num_samples):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            images, labels = next(data_iter)
        
        # Move to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Generate adversarial example
        with torch.enable_grad():
            adv_images = fgsm_attack(model, images, labels, epsilon, text_features)
        
        # Extract features (don't update reference distribution)
        with torch.no_grad():
            features = extractor.extract_features(adv_images, labels, is_clean=False, update_reference=False)
            
            # Create feature tensor with 3 features (excluding Mahalanobis)
            feature_tensor = torch.stack([
                features['confidence'],
                features['entropy'],
                #features['lid']
            ], dim=1)
            
            # Replace inf/NaN values with large finite values
            feature_tensor = torch.nan_to_num(feature_tensor, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Store feature vector
            adv_features.append(feature_tensor)
    
    # Concatenate features
    clean_features = torch.cat(clean_features, dim=0).cpu()
    adv_features = torch.cat(adv_features, dim=0).cpu()
    
    # Create labels (0 = clean, 1 = adversarial)
    clean_labels = torch.zeros(clean_features.size(0))
    adv_labels = torch.ones(adv_features.size(0))
    
    # Combine and shuffle
    all_features = torch.cat([clean_features, adv_features], dim=0)
    all_labels = torch.cat([clean_labels, adv_labels], dim=0)
    
    indices = torch.randperm(all_features.size(0))
    all_features = all_features[indices]
    all_labels = all_labels[indices]
    
    return all_features, all_labels

# Train model
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            
            # Safety check - ensure outputs are between 0 and 1
            outputs = torch.clamp(outputs, min=1e-7, max=1.0-1e-7)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features).squeeze()
                
                # Apply same safety check in validation
                outputs = torch.clamp(outputs, min=1e-7, max=1.0-1e-7)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        scheduler.step(avg_val_loss)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model

# Visualize feature distributions
def plot_features(features, labels):
    # Define feature names for all three features
    feature_names = ['Confidence', 'LID'] #'Entropy'
    
    # Check that we have valid data to plot
    clean_mask = (labels == 0)
    adv_mask = (labels == 1)
    
    print(f"Plotting {clean_mask.sum().item()} clean samples and {adv_mask.sum().item()} adversarial samples")
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot each feature separately with error handling
    for i in range(min(3, features.shape[1])):
        plt.subplot(1, 3, i+1)
        
        try:
            # Extract feature values
            clean_values = features[clean_mask, i].numpy()
            adv_values = features[adv_mask, i].numpy()
            
            # Print stats about the data
            print(f"Feature {feature_names[i]} statistics:")
            print(f"  Clean: min={np.nanmin(clean_values) if len(clean_values) > 0 else 'N/A'}, "
                 f"max={np.nanmax(clean_values) if len(clean_values) > 0 else 'N/A'}, "
                 f"finite={np.isfinite(clean_values).sum()}/{len(clean_values)}")
            print(f"  Adversarial: min={np.nanmin(adv_values) if len(adv_values) > 0 else 'N/A'}, "
                 f"max={np.nanmax(adv_values) if len(adv_values) > 0 else 'N/A'}, "
                 f"finite={np.isfinite(adv_values).sum()}/{len(adv_values)}")
            
            # Filter to finite values
            clean_values = clean_values[np.isfinite(clean_values)]
            adv_values = adv_values[np.isfinite(adv_values)]
            
            print(f"Feature {feature_names[i]}: {len(clean_values)} clean, {len(adv_values)} adversarial")
            
            # Only plot if we have enough data
            if len(clean_values) > 0 and len(adv_values) > 0:
                # For LID, we might need to limit the range
                if i == 2:  # LID feature
                    # Clip extreme values for better visualization
                    max_value = np.percentile(np.concatenate([clean_values, adv_values]), 95)
                    clean_values = np.clip(clean_values, None, max_value)
                    adv_values = np.clip(adv_values, None, max_value)
                    plt.xlim(0, max_value)
                
                sns.histplot(clean_values, color='blue', alpha=0.5, label='Clean', kde=True)
                sns.histplot(adv_values, color='red', alpha=0.5, label='Adversarial', kde=True)
                
                plt.title(f'{feature_names[i]} Distribution')
                plt.xlabel(feature_names[i])
                plt.legend()
            else:
                plt.text(0.5, 0.5, 'Insufficient data', horizontalalignment='center', 
                         verticalalignment='center', transform=plt.gca().transAxes)
        except Exception as e:
            print(f"Error plotting feature {feature_names[i]}: {str(e)}")
            plt.text(0.5, 0.5, f'Error plotting: {str(e)}', horizontalalignment='center', 
                    verticalalignment='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    
    try:
        plt.savefig('feature_distributions_no_mahalanobis.png')
        plt.close()
        print("Plot saved successfully")
    except Exception as e:
        print(f"Error saving plot: {str(e)}")

# Main function
def main():
    parser = argparse.ArgumentParser(description='Train MLP detector for adversarial examples')
    parser.add_argument('--samples', type=int, default=2000, help='Number of samples to use')
    parser.add_argument('--epsilon', type=float, default=0.03, help='FGSM epsilon')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load CLIP model
    print("Loading CLIP model...")
    model, preprocess = clip.load("RN50")
    model.eval()
    device = next(model.parameters()).device
    
    # Load dataset
    print("Loading dataset...")
    data_loader, classnames, template = build_test_data_loader('I', './dataset/', preprocess)
    text_features = clip_classifier(classnames, template, model)
    
    # Collect features
    print(f"Collecting features using {args.samples} samples with epsilon={args.epsilon}...")
    features, labels = collect_features(model, text_features, data_loader, 
                                       num_samples=args.samples, epsilon=args.epsilon)
    
    # Explicitly convert tensors to float32
    features = features.float()
    labels = labels.float()
    
    # Check for class balance
    print(f"Clean samples: {(labels == 0).sum().item()}")
    print(f"Adversarial samples: {(labels == 1).sum().item()}")
    
    # Visualize feature distributions
    print("Plotting feature distributions...")
    #plot_features(features, labels)
    
    # Normalize features
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    normalized_features = (features - mean) / (std + 1e-8)
    
    # Split data
    n_samples = features.size(0)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    indices = torch.randperm(n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Create data loaders
    train_data = TensorDataset(normalized_features[train_indices], labels[train_indices])
    val_data = TensorDataset(normalized_features[val_indices], labels[val_indices])
    test_data = TensorDataset(normalized_features[test_indices], labels[test_indices])
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)
    
    # Train model
    print("Training MLP classifier...")
    classifier = AdversarialClassifier(input_dim=2)  # Changed input_dim to 3
    classifier = train_model(classifier, train_loader, val_loader, epochs=args.epochs)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    classifier.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = classifier(features).squeeze()
            
            # Apply same safety check in testing
            outputs = torch.clamp(outputs, min=1e-7, max=1.0-1e-7)
            
            preds = (outputs >= 0.5).float()
            
            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(outputs)
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Print results
    print("\nClassification Report:")
    print(report)
    print(f"ROC AUC: {roc_auc:.3f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Clean', 'Adversarial'],
               yticklabels=['Clean', 'Adversarial'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_no_mahalanobis.png')
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_no_mahalanobis.png')
    
    # Save model
    save_dict = {
        'model_state_dict': classifier.state_dict(),
        'feature_mean': mean,
        'feature_std': std
    }
    torch.save(save_dict, 'adversarial_mlp_detector_no_mahalanobis.pth')
    print("Model saved as 'adversarial_mlp_detector_no_mahalanobis.pth'")

if __name__ == "__main__":
    main()