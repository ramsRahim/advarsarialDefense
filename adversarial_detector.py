import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np

class AdversarialDetector:
    """Ensemble of methods to detect adversarial examples."""
    
    def __init__(self, model, text_features, thresholds=None):
        """
        Initialize the adversarial detector.
        
        Args:
            model: CLIP model
            text_features: Text embeddings for classification
            thresholds: Dictionary of thresholds for each detection method
        """
        self.model = model
        self.text_features = text_features
        
        # Set default thresholds if not provided
        default_thresholds = {
            'confidence': 0.75,   # High confidence often means adversarial
            'entropy': 1.0        # Low entropy often means adversarial
        }
        
        if thresholds is None:
            thresholds = default_thresholds
            
        # Set thresholds
        self.conf_th = thresholds.get('confidence', default_thresholds['confidence'])
        self.ent_th = thresholds.get('entropy', default_thresholds['entropy'])
    
    def detect(self, images, labels):
        """
        Detect if images are adversarial using simple methods that don't require gradients.
        
        Args:
            images: Input images
            labels: Ground truth labels
            
        Returns:
            Dictionary with detection results
        """
        batch_size = images.size(0)
        device = images.device
        
        # Initialize detection results
        results = {
            'confidence': torch.zeros(batch_size, device=device),
            'entropy': torch.zeros(batch_size, device=device),
            'is_adversarial': torch.zeros(batch_size, dtype=torch.bool, device=device)
        }
        
        # Compute features and logits in eval mode to avoid any gradient issues
        self.model.eval()
        with torch.no_grad():
            # Get image features
            image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute logits
            logits = 100.0 * image_features @ self.text_features
            
            # Compute probabilities
            probs = F.softmax(logits, dim=1)
            
            # Compute confidence (max probability)
            confidence, pred_classes = probs.max(dim=1)
            results['confidence'] = confidence
            
            # Compute entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            results['entropy'] = entropy
            
            # Simple rule for adversarial detection:
            # High confidence OR low entropy indicates potential adversarial example
            is_adversarial = (confidence > self.conf_th) | (entropy < self.ent_th)
            results['is_adversarial'] = is_adversarial
            
            # Check if prediction differs from ground truth (could be an additional signal)
            is_misclassified = (pred_classes != labels)
            
            # Combine signals - a sample is adversarial if it's both flagged and misclassified
            results['is_adversarial'] = is_adversarial & is_misclassified
        
        return results