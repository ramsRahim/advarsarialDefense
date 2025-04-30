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

def run_test_tda(pos_cfg, neg_cfg, adv_cfg, loader, clip_model, clip_weights, attack_config=None):
    with torch.no_grad():
        # Initialize caches and accuracy list
        pos_cache, neg_cache, adv_cache, accuracies = {}, {}, {}, []
        
        # Unpack hyperparameters
        pos_enabled, neg_enabled, adv_enabled = pos_cfg['enabled'], neg_cfg['enabled'], adv_cfg['enabled']
        
        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
        
        if neg_enabled:
            neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}
        
        if adv_enabled:
            adv_params = {k: adv_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'lipschitz_threshold', 'mask_threshold']}

        # Set up attack if specified
        attack_type = attack_config.get('type', 'none') if attack_config else 'none'
        print(f"Running with attack type: {attack_type}")
        
        # Statistics for attacks
        clean_correct, adv_correct = 0, 0
        total_samples = 0

        # Test-time adaptation loop
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            target = target.cuda()
            total_samples += target.size(0)
            
            # Get clean predictions first
            clean_image_features, clean_logits, _, _, _ = get_clip_logits(images, clip_model, clip_weights)
            clean_pred = clean_logits.argmax(dim=1)
            clean_correct += (clean_pred == target).sum().item()
            
            # Apply adversarial attack if specified
            if attack_type != 'none':
                # For adversarial training, we need to keep gradients
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
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(adv_images, clip_model, clip_weights)
            prop_entropy = get_entropy(loss, clip_weights)

            # Update positive cache
            if pos_enabled:
                update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'])

            # Check if sample should go to negative or adverse cache
            if (neg_enabled or adv_enabled) and (
                neg_enabled and 
                neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']
            ):
                # Sample is eligible for negative/adverse cache
                if adv_enabled:
                    # Calculate Lipschitz constant to determine sensitivity
                    lipschitz = estimate_lipschitz_for_clip(image_features, clip_weights)
                    
                    if lipschitz > adv_params['lipschitz_threshold']:
                        # Highly sensitive sample - add to adverse cache
                        update_cache(adv_cache, pred, [image_features, loss, prob_map], 
                                   adv_params['shot_capacity'], True)
                    elif neg_enabled:
                        # Regular sample - add to negative cache
                        update_cache(neg_cache, pred, [image_features, loss, prob_map], 
                                   neg_params['shot_capacity'], True)
                elif neg_enabled:
                    # Adverse cache not enabled, use negative cache
                    update_cache(neg_cache, pred, [image_features, loss, prob_map], 
                               neg_params['shot_capacity'], True)

            # Compute final logits using all caches
            final_logits = clip_logits.clone()
            
            # Add positive cache contribution
            if pos_enabled and pos_cache:
                final_logits += compute_cache_logits(
                    image_features, pos_cache, 
                    pos_params['alpha'], pos_params['beta'], 
                    clip_weights
                )
            
            # Subtract negative cache contribution
            if neg_enabled and neg_cache:
                final_logits -= compute_cache_logits(
                    image_features, neg_cache, 
                    neg_params['alpha'], neg_params['beta'], 
                    clip_weights, 
                    (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper'])
                )
            
            # Apply stronger penalty for adverse cache samples
            if adv_enabled and adv_cache:
                adv_logits = compute_cache_logits(
                    image_features, adv_cache, 
                    adv_params['alpha'], adv_params['beta'], 
                    clip_weights, 
                    (adv_params['mask_threshold']['lower'], adv_params['mask_threshold']['upper'])
                )
                # Apply higher weight penalty (1.5x) for adverse examples
                final_logits -= 1.5 * adv_logits
            
            # Calculate accuracy and log
            acc = cls_acc(final_logits, target)  
            accuracies.append(acc)
            adv_correct += (final_logits.argmax(dim=1) == target).sum().item()

            # Print progress
            if i % 1000 == 0:
                print("---- TDA's test accuracy: {:.2f}. ----".format(sum(accuracies)/len(accuracies)))
                if adv_enabled:
                    print(f"Adverse cache entries: {sum(len(v) for v in adv_cache.values())}")
                if attack_type != 'none':
                    clean_acc = clean_correct / total_samples * 100
                    adv_acc = adv_correct / total_samples * 100
                    print(f"Clean accuracy: {clean_acc:.2f}%, Adversarial accuracy: {adv_acc:.2f}%")
        
        # Final accuracy
        final_acc = sum(accuracies)/len(accuracies)
        print("---- TDA's test accuracy: {:.2f}. ----".format(final_acc))
        
        # Print final attack statistics if applicable
        if attack_type != 'none':
            clean_acc = clean_correct / total_samples * 100
            adv_acc = adv_correct / total_samples * 100
            print(f"\nFinal results with {attack_type.upper()} attack:")
            print(f"Clean accuracy: {clean_acc:.2f}%")
            print(f"Adversarial accuracy: {adv_acc:.2f}%")
            print(f"Robustness (adv_acc/clean_acc): {adv_acc/clean_acc*100:.2f}%")
        
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

        # Pass all three configs to run_test_tda
        acc = run_test_tda(
            cfg['positive'], 
            cfg['negative'], 
            cfg.get('adverse', {'enabled': False}),  # Use adverse config if present
            test_loader, 
            clip_model, 
            clip_weights,
            attack_config
        )

if __name__ == "__main__":
    main()