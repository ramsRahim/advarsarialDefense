import random
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
import operator

import clip
from utils import *
import time

def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')

    args = parser.parse_args()

    return args

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        # Unpack the features, loss, scale, and zero_point
        image_features, loss = features_loss[0], features_loss[1]
        scale = features_loss[2] if len(features_loss) > 2 else None
        zero_point = features_loss[3] if len(features_loss) > 3 else None
        
        # Create the cache item
        if include_prob_map:
            # For negative cache
            prob_map = features_loss[4] if len(features_loss) > 4 else None
            item = [image_features, loss, scale, zero_point, prob_map]
        else:
            # For positive cache
            item = [image_features, loss, scale, zero_point]
        
        # Update the cache
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:  # Compare loss values
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

    

def get_tensor_size(tensor):
    """
    Calculate the size of a PyTorch tensor in bytes.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        int: The size of the tensor in bytes.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")
    
    # print(tensor.element_size())
    return tensor.element_size() * tensor.numel()


def compute_real_cache_size(cache):
    """
    Compute the real memory size of a cache (positive or negative) in bytes.
    """
    total_size = 0
    for class_entries in cache.values():
        for item in class_entries:
            # Calculate size of feature embedding (first element)
            feature_size = get_tensor_size(item[0]) if isinstance(item[0], torch.Tensor) else 8  # Default to 8 bytes for non-tensors
            
            # Calculate size of metadata (remaining elements)
            metadata_size = 0
            for x in item[1:]:
                if isinstance(x, torch.Tensor):
                    metadata_size += get_tensor_size(x)
                elif x is not None:
                    # For non-tensor values (like scalars), estimate size
                    metadata_size += 8  # Assume 8 bytes for numbers (like float64)
            
            total_size += feature_size + metadata_size
    return total_size


def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights, log_file):
    with open(log_file, 'w') as log:
        with torch.no_grad():
            pos_cache, neg_cache, accuracies = {}, {}, []
            total_lookup_time = 0.0
            total_inference_time = 0.0
            num_lookups = 0

            # Unpack all hyperparameters
            pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
            if pos_enabled:
                pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
            if neg_enabled:
                neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}
                
                # Enhanced extraction of entropy_threshold with better error handling
                entropy_config = neg_params['entropy_threshold']
                if isinstance(entropy_config, dict):
                    neg_entropy_threshold = (entropy_config.get('lower', 0.3) + entropy_config.get('upper', 0.8)) / 2
                else:
                    neg_entropy_threshold = 0.5  # Default value
                    
                # Process neg_mask_threshold
                neg_mask_thresholds = (
                    neg_params['mask_threshold']['lower'],
                    neg_params['mask_threshold']['upper']
                )

            for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
                try:
                    start_time = time.time()  # Start inference time measurement

                    # Get image features and logits
                    image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images, clip_model, clip_weights)
                    target = target.cuda()
                    
                    # Safely calculate entropy
                    try:
                        prop_entropy = get_entropy(loss, clip_weights)
                    except Exception as e:
                        log.write(f"Error calculating entropy: {e}\n")
                        prop_entropy = 0.5  # Default middle value
                    
                    # Safe unpacking with error handling
                    try:
                        if isinstance(image_features, tuple) and len(image_features) == 3:
                            img_features_int8, img_scale, img_zero_point = image_features
                        else:
                            log.write(f"Warning: Expected tuple of 3 elements but got {type(image_features)}\n")
                            # Handle the case gracefully with defaults
                            if isinstance(image_features, tuple):
                                img_features_int8 = image_features[0]
                            else:
                                img_features_int8 = image_features
                            img_scale = torch.tensor(1.0, device=image_features[0].device if isinstance(image_features, tuple) else image_features.device)
                            img_zero_point = torch.tensor(0, device=image_features[0].device if isinstance(image_features, tuple) else image_features.device)
                    except Exception as e:
                        log.write(f"Error unpacking image_features: {e}\n")
                        continue  # Skip this iteration

                    # Update caches with quantized features
                    if pos_enabled:
                        update_cache(pos_cache, pred, [img_features_int8, loss, img_scale, img_zero_point], 
                                   pos_params['shot_capacity'])
                    
                    if neg_enabled and prop_entropy > neg_entropy_threshold:
                        update_cache(neg_cache, pred, [img_features_int8, loss, img_scale, img_zero_point, prob_map],
                                   neg_params['shot_capacity'], True)
                    
                    # Compute final logits using quantized cache
                    final_logits = clip_logits.clone()

                    # Measure lookup time for cache
                    lookup_start = time.time()
                    if pos_enabled and pos_cache:
                        pos_logits = compute_cache_logits(image_features, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
                        final_logits += pos_logits  # Add positive cache logits

                    if neg_enabled and neg_cache:
                        neg_logits = compute_cache_logits(image_features, neg_cache, neg_params['alpha'], neg_params['beta'], clip_weights, neg_mask_thresholds)
                        final_logits -= neg_logits  # Subtract negative cache logits

                    # Accumulate total lookup time
                    lookup_end = time.time()
                    lookup_time = lookup_end - lookup_start
                    total_lookup_time += lookup_time
                    num_lookups += 1

                    acc = cls_acc(final_logits, target)
                    accuracies.append(acc)
                    inference_time = time.time() - start_time  # Measure total inference time
                    total_inference_time += inference_time

                    # Monitor the KV table size
                    if i % 1000 == 0:
                        pos_cache_size = compute_real_cache_size(pos_cache)
                        neg_cache_size = compute_real_cache_size(neg_cache)
                        avg_lookup_time = total_lookup_time / num_lookups if num_lookups > 0 else 0
                        avg_inference_time = total_inference_time / (i + 1)

                        log.write(f"---- Iteration {i} ----\n")
                        log.write(f"Positive Cache Size: {pos_cache_size} bytes\n")
                        log.write(f"Negative Cache Size: {neg_cache_size} bytes\n")
                        log.write(f"Average Cache Lookup Time: {avg_lookup_time:.6f} seconds\n")
                        log.write(f"Average Inference Time: {avg_inference_time:.6f} seconds\n")
                        log.write(f"Top 1 Accuracy: {sum(accuracies) / len(accuracies):.2f}%\n\n")
                        
                except Exception as e:
                    # Catch any exceptions to prevent segfaults
                    log.write(f"Error in iteration {i}: {str(e)}\n")
                    print(f"Error in iteration {i}: {str(e)}")
                    continue  # Skip this iteration

            # Calculate overall accuracy
            avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0

            log.write('=' * 50 + '\n')
            log.write(f"Final Top 1 Accuracy: {avg_acc:.2f}%\n")
            log.write('=' * 50 + '\n')

            return avg_acc


def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)


    # Run TDA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model)
        log_file = f"logs_{dataset_name}_tda_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights, log_file)

if __name__ == "__main__":
    main()
