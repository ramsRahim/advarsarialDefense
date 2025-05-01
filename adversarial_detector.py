import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors

class AdversarialDetector:
    """Enhanced detector for adversarial examples using multiple detection methods."""
    
    def __init__(self, model, text_features, thresholds=None, require_misclassification=True):
        """
        Initialize the detector.
        
        Args:
            model: CLIP model
            text_features: Text features for classification
            thresholds: Dictionary with detection thresholds
            require_misclassification: Whether to require samples to be misclassified
        """
        self.model = model
        self.text_features = text_features
        self.require_misclassification = require_misclassification
        
        # Default thresholds
        default_thresholds = {
            'confidence': 0.75,  # High confidence threshold
            'entropy': 0.5,      # Low entropy threshold
            'mahalanobis': 10.0, # High Mahalanobis distance threshold
            'lid': 8.0           # High LID threshold
        }
        
        # Update with provided thresholds
        if thresholds is not None:
            if isinstance(thresholds, dict):
                default_thresholds.update({k: v for k, v in thresholds.items() if k in default_thresholds})
        
        # Set thresholds
        self.conf_threshold = default_thresholds['confidence']
        self.entropy_threshold = default_thresholds['entropy']
        self.mahalanobis_threshold = default_thresholds['mahalanobis']
        self.lid_threshold = default_thresholds['lid']
        
        # Initialize Mahalanobis distance reference
        self.feature_mean = None
        self.feature_cov_inv = None
        self.n_samples = 0
        self.min_samples_for_cov = 100
        
        # Initialize LID reference
        self.feature_bank = []
        self.k_neighbors = min(20, 100)  # Number of neighbors for LID calculation
        self.nn_model = None
        
        print(f"Enhanced adversarial detector initialized with:")
        print(f"  Confidence threshold: {self.conf_threshold}")
        print(f"  Entropy threshold: {self.entropy_threshold}")
        print(f"  Mahalanobis threshold: {self.mahalanobis_threshold}")
        print(f"  LID threshold: {self.lid_threshold}")
        print(f"  Require misclassification: {self.require_misclassification}")
    
    def _update_reference_distribution(self, features, is_correct):
        """Update reference distribution with new clean features."""
        # Only use correctly classified samples to update reference
        if not is_correct.any():
            return
            
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
        
        # Update Mahalanobis distance reference
        if self.feature_mean is None:
            self.feature_mean = np.mean(clean_features, axis=0)
            self.n_samples = len(clean_features)
        else:
            # Incremental update of mean
            new_n = self.n_samples + len(clean_features)
            self.feature_mean = (self.feature_mean * self.n_samples + np.sum(clean_features, axis=0)) / new_n
            self.n_samples = new_n
        
        # Update covariance if we have enough samples
        if self.n_samples >= self.min_samples_for_cov:
            # Compute covariance with regularization
            centered = total_features - self.feature_mean
            cov = np.cov(centered, rowvar=False)
            
            # Add regularization to ensure invertibility
            cov += np.eye(cov.shape[0]) * 0.01
            
            try:
                self.feature_cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                # If inversion fails, use pseudo-inverse
                self.feature_cov_inv = np.linalg.pinv(cov)
    
    def _compute_mahalanobis(self, features):
        """Compute Mahalanobis distance for given features."""
        if self.feature_mean is None or self.feature_cov_inv is None:
            # Return a large value if reference distribution not established
            return torch.ones(features.size(0), device=features.device) * float('inf')
        
        # Convert to numpy for Mahalanobis computation
        features_np = features.cpu().numpy()
        
        # Compute centered features
        centered = features_np - self.feature_mean
        
        # Compute Mahalanobis distance
        mahalanobis_dist = np.sqrt(np.sum(np.dot(centered, self.feature_cov_inv) * centered, axis=1))
        
        return torch.tensor(mahalanobis_dist, device=features.device)
    
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
    
    def detect(self, images, labels):
        """
        Detect adversarial examples using multiple metrics.
        
        Args:
            images: Input images
            labels: Ground truth labels
            
        Returns:
            Dictionary with detection results
        """
        # Make sure we don't use gradients
        with torch.no_grad():
            # Get batch size and device
            batch_size = images.size(0)
            device = images.device
            
            # Initialize results
            results = {
                'confidence': torch.zeros(batch_size, device=device),
                'entropy': torch.zeros(batch_size, device=device),
                'mahalanobis': torch.zeros(batch_size, device=device),
                'lid': torch.zeros(batch_size, device=device),
                'is_adversarial': torch.zeros(batch_size, dtype=torch.bool, device=device)
            }
            
            # Compute features and logits
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
            results['confidence'] = confidence
            
            # Compute entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            results['entropy'] = entropy
            
            # Compute Mahalanobis distance and LID
            features_flat = image_features.view(batch_size, -1)
            
            # Mahalanobis distance
            mahalanobis_dist = self._compute_mahalanobis(features_flat)
            results['mahalanobis'] = mahalanobis_dist
            
            # Local Intrinsic Dimensionality
            lid_scores = self._compute_lid(features_flat)
            results['lid'] = lid_scores
            
            # Update reference distributions with correctly classified samples
            if is_correct.any():
                self._update_reference_distribution(features_flat, is_correct)
            
            # Detect adversarial examples based on multiple criteria
            confidence_flag = confidence > self.conf_threshold
            entropy_flag = entropy < self.entropy_threshold
            mahalanobis_flag = mahalanobis_dist > self.mahalanobis_threshold
            lid_flag = lid_scores > self.lid_threshold
            
            # Count number of flags
            flag_count = confidence_flag.float() + entropy_flag.float() + \
                         mahalanobis_flag.float() + lid_flag.float()
            
            # Flag as suspicious if at least 2 methods indicate adversarial
            is_suspicious = flag_count >= 2
            
            if self.require_misclassification:
                # Only flag examples that are both suspicious and misclassified
                is_misclassified = ~is_correct
                results['is_adversarial'] = is_suspicious & is_misclassified
            else:
                # Flag all suspicious examples
                results['is_adversarial'] = is_suspicious
        
        return results