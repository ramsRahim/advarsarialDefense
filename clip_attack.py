import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torchvision import datasets, transforms

# Import utilities from the existing codebase
from utils import build_test_data_loader, get_config_file, clip_classifier
from datasets import build_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='CLIP Adversarial Attack')
    parser.add_argument('--dataset', type=str, default='A', help='Dataset to use (A, V, R, S, or other dataset names)')
    parser.add_argument('--data-root', type=str, default='./dataset', help='Path to data directory')
    parser.add_argument('--config', type=str, default='configs', help='Path to config directory')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--backbone', type=str, default='RN50', help='CLIP backbone')
    parser.add_argument('--epsilon', type=float, default=0.03, help='Attack strength')
    parser.add_argument('--alpha', type=float, default=0.007, help='PGD step size')
    parser.add_argument('--iters', type=int, default=10, help='PGD iterations')
    parser.add_argument('--use-fp32', action='store_true', help='Use FP32 precision instead of default')
    return parser.parse_args()

def classify(model, images, text_features):
    """Helper function for classification with CLIP model."""
    with torch.cuda.amp.autocast(enabled=model.visual.conv1.weight.dtype == torch.float16):
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features
    return logits

def fgsm_attack(model, images, labels, epsilon, text_features):
    """
    Fast Gradient Sign Method attack.
    
    Args:
        model: CLIP model
        images: Input images 
        labels: Target labels
        epsilon: Attack strength
        text_features: Text features for classification
        
    Returns:
        Adversarial examples
    """
    # Ensure images are on the same device as the model
    device = next(model.parameters()).device
    
    # Proper handling of batch dimension
    if images.dim() == 5:  # [batch, 1, channels, height, width]
        images = images.squeeze(1)
    
    # Move to correct device and convert to float
    images = images.clone().detach().to(device, dtype=torch.float32)
    labels = labels.to(device)
    images.requires_grad = True
    
    # Forward pass
    logits = classify(model, images, text_features)
    
    # Calculate loss
    loss = F.cross_entropy(logits, labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Create adversarial examples
    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()
    
    # Generate perturbed image
    perturbed_image = images + epsilon * sign_data_grad
    
    # Ensure valid image range [0,1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image.detach()

def pgd_attack(model, images, labels, epsilon, alpha, iters, text_features):
    """
    Projected Gradient Descent attack.
    
    Args:
        model: CLIP model
        images: Input images
        labels: Target labels
        epsilon: Maximum perturbation
        alpha: Step size
        iters: Number of iterations
        text_features: Text features for classification
        
    Returns:
        Adversarial examples
    """
    # Ensure images are on the same device as model
    device = next(model.parameters()).device
    
    # Proper handling of batch dimension
    if images.dim() == 5:
        images = images.squeeze(1)
    
    # Move to correct device and convert to float
    images = images.clone().detach().to(device, dtype=torch.float32)
    labels = labels.to(device)
    
    # Save original images for projection step
    ori_images = images.clone().detach()
    
    for i in range(iters):
        # Reset gradients
        images.requires_grad = True
        
        # Forward pass
        logits = classify(model, images, text_features)
        
        # Calculate loss
        loss = F.cross_entropy(logits, labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Get gradient sign
        data_grad = images.grad.data
        sign_data_grad = data_grad.sign()
        
        # Update images
        images = images.detach() + alpha * sign_data_grad
        
        # Project back to epsilon ball
        eta = torch.clamp(images - ori_images, -epsilon, epsilon)
        images = torch.clamp(ori_images + eta, 0, 1)
    
    return images.detach()

# Evaluation pipeline
def evaluate_attack(model, dataloader, attack_fn, attack_params, text_features, attack_name="Attack", device="cuda"):
    model.eval()
    total, clean_correct, adv_correct = 0, 0, 0
    
    for batch_idx, batch in enumerate(dataloader):
        try:
            # Handle different dataloader output formats
            if isinstance(batch, list) and len(batch) == 2:
                # Check if the first item is a list of tensors (individual images)
                if isinstance(batch[0], list) and all(isinstance(x, torch.Tensor) for x in batch[0]):
                    # Stack the individual image tensors
                    images = torch.stack(batch[0]).to(device)
                    
                    # Create expanded labels with the same value repeated batch_size times
                    if batch[1].size(0) == 1:
                        # Extract the single label value and repeat it
                        label_value = batch[1].item()
                        labels = torch.full((len(batch[0]),), label_value, 
                                           dtype=torch.long, device=device)
                    else:
                        # Normal case - labels match batch size
                        labels = batch[1].to(device)
                elif isinstance(batch[0], torch.Tensor):
                    # Standard case - batch[0] is a tensor of stacked images
                    images = batch[0].to(device)
                    
                    # Create expanded labels with the same value repeated batch_size times
                    if batch[1].size(0) == 1:
                        # Extract the single label value and repeat it
                        label_value = batch[1].item()
                        labels = torch.full((batch[0].size(0),), label_value, 
                                           dtype=torch.long, device=device)
                    else:
                        # Normal case - labels match batch size
                        labels = batch[1].to(device)
                else:
                    raise ValueError(f"Unexpected batch[0] format: {type(batch[0])}")
            else:
                # Standard PyTorch dataloader format
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
            
            # Fix image shape if needed
            if images.dim() == 5:  # If shape is [batch, 1, channels, height, width]
                images = images.squeeze(1)  # Remove the extra dimension
            
            # Print info for the first batch
            if batch_idx == 0:
                print(f"Processing with images shape: {images.shape}, labels shape: {labels.shape}")
            
            # Skip empty batches
            if images.size(0) == 0:
                continue
                
            total += labels.size(0)
            
            # Clean accuracy
            with torch.no_grad():
                logits = classify(model, images, text_features)
            preds = logits.argmax(dim=1)
            clean_correct += (preds == labels).sum().item()
            
            # Adversarial accuracy
            adv_images = attack_fn(model, images, labels, text_features=text_features, **attack_params)
            
            with torch.no_grad():
                logits_adv = classify(model, adv_images, text_features)
            preds_adv = logits_adv.argmax(dim=1)
            adv_correct += (preds_adv == labels).sum().item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches...")
                
        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if total == 0:
        print("Warning: No valid batches were processed!")
        return 0.0, 0.0
        
    print(f"[{attack_name}] Clean Acc: {clean_correct/total*100:.2f}% | Adv Acc: {adv_correct/total*100:.2f}%")
    return clean_correct/total, adv_correct/total


def main():
    args = parse_args()
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CLIP model
    model, preprocess = clip.load(args.backbone, device=device)
    model.eval()
    
    # Disable grad computation for the model
    for param in model.parameters():
        param.requires_grad = False
    
    # Load dataset and text features using existing utility functions
    dataset_name = args.dataset
    data_root = args.data_root
    
    # Get test dataloader and other dataset info
    print(f"Loading dataset {dataset_name} from {data_root}...")
    test_loader, classnames, template = build_test_data_loader(
        dataset_name, data_root, preprocess, batch_size=args.batch_size
    )
    
    # Create text features using existing clip_classifier function
    clip_weights = clip_classifier(classnames, template, model)
    
    print(f"Loaded dataset: {dataset_name} with {len(classnames)} classes")
    print(f"Running attack with epsilon={args.epsilon}, alpha={args.alpha}, iters={args.iters}")
    
        
    # Run attacks
    fgsm_results = evaluate_attack(
        model, test_loader, fgsm_attack,
        {'epsilon': args.epsilon}, 
        clip_weights, "FGSM", device
    )
    
    pgd_results = evaluate_attack(
        model, test_loader, pgd_attack,
        {'epsilon': args.epsilon, 'alpha': args.alpha, 'iters': args.iters}, 
        clip_weights, "PGD", device
    )
    
    # Summarize results
    print("\nSummary:")
    print(f"Dataset: {dataset_name}")
    print(f"Clean Accuracy: {fgsm_results[0]*100:.2f}%")
    print(f"FGSM Adversarial Accuracy: {fgsm_results[1]*100:.2f}%")
    print(f"PGD Adversarial Accuracy: {pgd_results[1]*100:.2f}%")

if __name__ == "__main__":
    main()