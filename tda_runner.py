import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
import operator
from clip_attack import fgsm_attack, pgd_attack
import clip
from utils import *
from adversarial_detector import AdversarialDetector

def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')

     # Add adversarial attack parameters
    parser.add_argument('--attack', dest='attack', type=str, choices=['none', 'fgsm', 'pgd'], default='none', help='Adversarial attack to apply (none, fgsm, pgd).')
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.03, help='Attack strength (max perturbation).')
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.007, help='Step size for PGD attack.')
    parser.add_argument('--iters', dest='iters', type=int, default=10, help='Number of iterations for PGD attack.')

    args = parser.parse_args()

    return args


def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]


def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits

def run_test_tda(pos_cfg, neg_cfg, adv_cfg, loader, clip_model, clip_weights, attack_config=None , log_path=None):
    with open(log_path, 'w') as log:
        with torch.no_grad():
            # Initialize caches and stats
            pos_cache, neg_cache, adv_cache = {}, {}, {}
            accuracies = []
            clean_correct, adv_correct, total_samples = 0, 0, 0
            adv_detected = 0
            
            # Unpack hyperparameters
            pos_enabled = pos_cfg['enabled']
            neg_enabled = neg_cfg['enabled'] 
            adv_enabled = adv_cfg['enabled']
            
            if pos_enabled:
                pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
            
            if neg_enabled:
                neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}
            
            if adv_enabled:
                # Extract basic parameters for adverse cache
                adv_params = {k: adv_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'mask_threshold']}
                
                # Get detection parameters - handle both old and new config formats
                detection_params = adv_cfg.get('detection', {})
                
                if detection_params:
                    # New configuration format with 'detection' key
                    detector_thresholds = {
                        'confidence': detection_params.get('confidence_threshold', 0.75),
                        'entropy': detection_params.get('entropy_threshold', 0.5),
                        # Add new metrics from enhanced detector
                        'mahalanobis': detection_params.get('mahalanobis_threshold', 10.0),
                        'lid': detection_params.get('lid_threshold', 8.0)
                    }
                    # Get penalty multiplier
                    penalty_multiplier = detection_params.get('penalty_multiplier', 1.5)
                    # Whether to require misclassification
                    require_misclassification = detection_params.get('require_misclassification', True)
                else:
                    # Legacy configuration format
                    ensemble_thresholds = adv_cfg.get('ensemble_thresholds', {})
                    detector_thresholds = {
                        'confidence': ensemble_thresholds.get('confidence', 0.75),
                        'entropy': ensemble_thresholds.get('entropy', 0.5),
                        'mahalanobis': ensemble_thresholds.get('mahalanobis', 10.0),
                        'lid': ensemble_thresholds.get('lid', 8.0)
                    }
                    penalty_multiplier = 1.5  # Default
                    require_misclassification = True  # Default
                
                # Initialize enhanced detector with thresholds
                detector = AdversarialDetector(
                    clip_model, 
                    clip_weights, 
                    thresholds=detector_thresholds,
                    require_misclassification=require_misclassification
                )

            # Set up attack
            attack_type = attack_config.get('type', 'none') if attack_config else 'none'
            print(f"Running with attack type: {attack_type}")

            # Test-time adaptation loop
            for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
                # Move data to GPU
                device = next(clip_model.parameters()).device
                if isinstance(images, list):
                    # Convert list to tensor - assuming list of tensors
                    try:
                        images = torch.stack(images).to(device)
                        if images.size(0) > 1 and target.size(0) == 1:
                            # Broadcast target to match batch size
                            target = target.repeat(images.size(0))
                    except:
                        print("Error: Failed to convert images list to tensor. Skipping batch.")
                        continue
                else:
                    # Standard tensor case
                    images = images.to(device)
                
                if images.dim() == 5:
                    # Images have shape [batch_size, 1, channels, height, width]
                    # We need to squeeze out that extra dimension
                    images = images.squeeze(1)
                    print(f"Squeezed image tensor from 5D to 4D, new shape: {images.shape}")

                # Handle target similarly if needed
                if isinstance(target, list):
                    target = torch.tensor(target).to(device)
                else:
                    target = target.to(device)
                
                # Continue with the function
                total_samples += target.size(0)
                
                # Get clean predictions
                clean_image_features = clip_model.encode_image(images)
                clean_image_features = clean_image_features / clean_image_features.norm(dim=-1, keepdim=True)
                clean_logits = 100.0 * clean_image_features @ clip_weights
                clean_pred = clean_logits.argmax(dim=1)
                clean_correct += (clean_pred == target).sum().item()
                
                # Apply adversarial attack if specified
                if attack_type != 'none':
                    with torch.enable_grad():
                        if attack_type == 'fgsm':
                            adv_images = fgsm_attack(
                                clip_model, images, target, 
                                attack_config.get('epsilon', 0.03), 
                                clip_weights
                            )
                        elif attack_type == 'pgd':
                            adv_images = pgd_attack(
                                clip_model, images, target, 
                                attack_config.get('epsilon', 0.03),
                                attack_config.get('alpha', 0.007),
                                attack_config.get('iters', 10),
                                clip_weights
                            )
                        else:
                            adv_images = images
                else:
                    adv_images = images
                
                # Now run TDA on the possibly adversarial images
                image_features = clip_model.encode_image(adv_images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                clip_logits = 100.0 * image_features @ clip_weights
                
                # Compute probability map
                prob_map = F.softmax(clip_logits, dim=1)
                
                # Calculate a scalar loss value for caching - use negative confidence
                confidence = prob_map.max(dim=1)[0]
                loss = -confidence.mean()  # Higher confidence = lower loss
                
                # Calculate entropy
                entropy = -torch.sum(prob_map * torch.log(prob_map + 1e-10), dim=1)
                
                # Get prediction
                pred = clip_logits.argmax(dim=1)
                
                # Handle batch dimension for caching
                batch_size = images.size(0)
                
                # Update positive cache (one entry per sample in batch)
                if pos_enabled:
                    for j in range(batch_size):
                        # Use individual confidence as loss
                        sample_loss = -confidence[j].item()
                        update_cache(pos_cache, pred[j].item(), 
                                [image_features[j:j+1], sample_loss], 
                                pos_params['shot_capacity'])

                # Check for adversarial examples using enhanced detector
                if adv_enabled:
                    # Use enhanced detector with Mahalanobis and LID metrics
                    detection_results = detector.detect(adv_images, target)
                    is_adversarial = detection_results['is_adversarial']
                    
                    # Count detected adversarial examples
                    if attack_type != 'none':
                        adv_detected += is_adversarial.sum().item()
                    
                    # Add to adverse cache if detected as adversarial
                    for j in range(batch_size):
                        if is_adversarial[j]:
                            sample_loss = -confidence[j].item()
                            update_cache(adv_cache, pred[j].item(), 
                                    [image_features[j:j+1], sample_loss, prob_map[j:j+1]], 
                                    adv_params['shot_capacity'], True)
                            
                            # Log detection details occasionally for debugging
                            if i % 200 == 0 and j == 0:  # Just log the first one in the batch occasionally
                                print(f"\nAdversarial sample detected - Details:")
                                print(f"  Confidence: {detection_results['confidence'][j]:.4f} (threshold: {detector.conf_threshold:.4f})")
                                print(f"  Entropy: {detection_results['entropy'][j]:.4f} (threshold: {detector.entropy_threshold:.4f})")
                                print(f"  Mahalanobis: {detection_results['mahalanobis'][j]:.4f} (threshold: {detector.mahalanobis_threshold:.4f})")
                                print(f"  LID: {detection_results['lid'][j]:.4f} (threshold: {detector.lid_threshold:.4f})")
                
                # Check for negative cache
                if neg_enabled:
                    for j in range(batch_size):
                        # Get entropy for this specific sample
                        sample_entropy = entropy[j].item()
                        sample_loss = -confidence[j].item()
                        
                        # Check entropy threshold for negative cache
                        if (neg_params['entropy_threshold']['lower'] < sample_entropy < 
                            neg_params['entropy_threshold']['upper']):
                            
                            # Check if already in adverse cache
                            in_adverse = False
                            if adv_enabled and is_adversarial[j]:
                                in_adverse = True
                                
                            # Add to negative cache if not in adverse
                            if not in_adverse:
                                update_cache(neg_cache, pred[j].item(), 
                                        [image_features[j:j+1], sample_loss, prob_map[j:j+1]], 
                                        neg_params['shot_capacity'], True)

                # Compute final logits using all caches
                final_logits = clip_logits.clone()
                
                # Add positive cache contribution
                if pos_enabled and pos_cache:
                    pos_logits = compute_cache_logits(
                        image_features, pos_cache, 
                        pos_params['alpha'], pos_params['beta'], 
                        clip_weights
                    )
                    final_logits += pos_logits
                
                # Subtract negative cache contribution
                if neg_enabled and neg_cache:
                    neg_logits = compute_cache_logits(
                        image_features, neg_cache, 
                        neg_params['alpha'], neg_params['beta'], 
                        clip_weights, 
                        (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper'])
                    )
                    final_logits -= neg_logits
                
                # Apply stronger penalty for adverse cache samples
                if adv_enabled and adv_cache:
                    adv_logits = compute_cache_logits(
                        image_features, adv_cache, 
                        adv_params['alpha'], adv_params['beta'], 
                        clip_weights, 
                        (adv_params['mask_threshold']['lower'], adv_params['mask_threshold']['upper'])
                    )
                    # Apply configurable penalty for adverse examples
                    final_logits -= penalty_multiplier * adv_logits
                    
                # Calculate accuracy
                acc = (final_logits.argmax(dim=1) == target).float().mean().item() * 100
                accuracies.append(acc)
                adv_correct += (final_logits.argmax(dim=1) == target).sum().item()

                # Print progress
                if i % 100 == 0:
                    print(f"---- Iteration {i}/{len(loader)}, TDA's test accuracy: {sum(accuracies)/len(accuracies):.2f}%. ----")
                    if adv_enabled:
                        print(f"Adverse cache entries: {sum(len(v) for v in adv_cache.values())}")
                        log.write(f"Adverse cache entries: {sum(len(v) for v in adv_cache.values())}\n")
                        if attack_type != 'none':
                            detect_rate = adv_detected / total_samples * 100 if total_samples > 0 else 0
                            print(f"Adversarial detection rate: {detect_rate:.2f}%")
                            log.write(f"Adversarial detection rate: {detect_rate:.2f}%\n")
                    if neg_enabled:
                        print(f"Negative cache entries: {sum(len(v) for v in neg_cache.values())}")
                        log.write(f"Negative cache entries: {sum(len(v) for v in neg_cache.values())}\n")
                    if attack_type != 'none':
                        clean_acc = clean_correct / total_samples * 100
                        adv_acc = adv_correct / total_samples * 100
                        print(f"Clean accuracy: {clean_acc:.2f}%, Adversarial accuracy: {adv_acc:.2f}%")
                        log.write(f"Clean accuracy: {clean_acc:.2f}%, Adversarial accuracy: {adv_acc:.2f}%\n")
        
        # Final accuracy
        final_acc = sum(accuracies)/len(accuracies)
        print("---- TDA's test accuracy: {:.2f}%. ----".format(final_acc))
        log.write("---- TDA's test accuracy: {:.2f}%. ----\n".format(final_acc))
        
    # Print final attack statistics if applicable
    if attack_type != 'none':
        clean_acc = clean_correct / total_samples * 100
        adv_acc = adv_correct / total_samples * 100
        detect_rate = adv_detected / total_samples * 100 if total_samples > 0 else 0
        
        log.write(f"\nFinal results with {attack_type.upper()} attack:")
        log.write(f"Clean accuracy: {clean_acc:.2f}%")
        log.write(f"Adversarial accuracy: {adv_acc:.2f}%")
        log.write(f"Adversarial detection rate: {detect_rate:.2f}%")
        log.write(f"Robustness (adv_acc/clean_acc): {adv_acc/clean_acc*100:.2f}%")
    
    # Print cache statistics
    log.write("\nCache Statistics:")
    if pos_enabled:
        log.write(f"Positive cache entries: {sum(len(v) for v in pos_cache.values())}")
    if neg_enabled:
        log.write(f"Negative cache entries: {sum(len(v) for v in neg_cache.values())}")
    if adv_enabled:
        log.write(f"Adverse cache entries: {sum(len(v) for v in adv_cache.values())}")
    
    return final_acc


def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)
    
    # Create attack configuration
    attack_config = {
        'type': args.attack,
        'epsilon': args.epsilon,
        'alpha': args.alpha,
        'iters': args.iters
    }

    print(f"Attack configuration: {attack_config}")

    # Run TDA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model)

        log_dir="logs"
        os.makedirs(log_dir, exist_ok=True)
    
        # Create a timestamp and unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        attack_type = attack_config.get('type', 'none')
        filename = f"{timestamp}_{dataset_name}_{attack_type}.log"
        filepath = os.path.join(log_dir, filename)
        # Pass all three configs to run_test_tda
        acc = run_test_tda(
            cfg['positive'], 
            cfg['negative'], 
            cfg.get('adverse', {'enabled': False}),  # Use adverse config if present
            test_loader, 
            clip_model, 
            clip_weights,
            attack_config,
            filepath
        )

if __name__ == "__main__":
    main()