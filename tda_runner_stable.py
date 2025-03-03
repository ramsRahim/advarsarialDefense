import random
import argparse
import gc
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
import operator
import numpy as np

import clip
from utils import *
import time
import os
import psutil

def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')
    parser.add_argument('--use-int8', dest='use_int8', action='store_true', help='Use INT8 quantization for features.')

    args = parser.parse_args()
    return args

def monitor_memory():
    """Monitor memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Return in MB

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        try:
            # Handle potential scalar or tensor pred
            pred_key = pred.item() if isinstance(pred, torch.Tensor) else pred
            
            # Unpack the features - might be a quantized tuple or regular tensor
            image_features = features_loss[0]
            
            # Handle potential scalar or tensor loss values
            loss_val = features_loss[1]
            if isinstance(loss_val, torch.Tensor):
                if loss_val.numel() == 1:
                    # Single value tensor
                    loss_val = loss_val.item()
                else:
                    # Multiple values, take mean
                    loss_val = loss_val.mean().item()
            
            # Create the cache item based on whether it includes probability map
            if include_prob_map:
                if len(features_loss) > 2:
                    prob_map = features_loss[2]
                    item = [image_features, loss_val, prob_map]
                else:
                    # Not enough elements, skip this update
                    return
            else:
                item = [image_features, loss_val]
            
            # Update the cache based on the prediction key
            if pred_key in cache:
                # Cache entry exists for this class
                if len(cache[pred_key]) < shot_capacity:
                    # Still have room in the cache
                    cache[pred_key].append(item)
                elif loss_val < cache[pred_key][-1][1]:
                    # Replace worst item in cache with current one
                    cache[pred_key][-1] = item
                
                # Sort entries by loss (lower loss is better)
                cache[pred_key] = sorted(cache[pred_key], key=operator.itemgetter(1))
            else:
                # Create new cache entry for this class
                cache[pred_key] = [item]
                
        except Exception as e:
            print(f"Error in update_cache: {e}")
            # Skip this update if there's an error

def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using cache."""
    with torch.no_grad():
        try:
            if not cache:
                # Determine number of classes
                if isinstance(clip_weights, list):
                    num_classes = len(clip_weights)
                else:
                    num_classes = clip_weights.size(1)
                return torch.zeros((1, num_classes), device=image_features.device)
            
            # Normalize image features
            img_features_norm = F.normalize(image_features.float(), p=2, dim=1)
            
            # Collect cache keys and values
            cache_keys = []
            cache_values = []
            
            for class_index in sorted(cache.keys()):
                for item in cache[class_index]:
                    # Get the feature vector
                    feature = item[0]
                    
                    # Ensure feature is a tensor of the right type
                    if not isinstance(feature, torch.Tensor):
                        continue
                    
                    # Normalize feature
                    feature_norm = F.normalize(feature.float(), p=2, dim=0)
                    cache_keys.append(feature_norm.unsqueeze(0))
                    
                    # Handle values based on cache type
                    if neg_mask_thresholds and len(item) > 2:
                        # For negative cache with probability map
                        cache_values.append(item[2])
                    else:
                        # For positive cache with just class index
                        cache_values.append(class_index)
            
            # If no valid items were collected, return zeros
            if not cache_keys:
                if isinstance(clip_weights, list):
                    num_classes = len(clip_weights)
                else:
                    num_classes = clip_weights.size(1)
                return torch.zeros((1, num_classes), device=image_features.device)
            
            # Stack all normalized features
            cache_keys = torch.cat(cache_keys, dim=0).to(image_features.device)
            
            # Determine number of classes
            if isinstance(clip_weights, list):
                num_classes = len(clip_weights)
            else:
                num_classes = clip_weights.size(1)
            
            # Compute cosine similarity - but check dimensions first!
            # Instead of using t() which only works on 2D tensors, use permute or transpose
            if cache_keys.dim() > 2:
                # Handle 3D tensor - reshape or squeeze as needed
                if cache_keys.size(1) == 1:  # If middle dimension is 1, we can squeeze
                    cache_keys = cache_keys.squeeze(1)
                else:
                    # Reshape to 2D by flattening the last dimensions
                    cache_keys = cache_keys.reshape(cache_keys.size(0), -1)
            
            # Now we can safely compute the affinity
            affinity = img_features_norm @ cache_keys.transpose(0, 1)
            
            # Process based on cache type
            if neg_mask_thresholds and len(cache_values) > 0 and isinstance(cache_values[0], torch.Tensor):
                # For negative cache - use probability maps with thresholds
                prob_maps = torch.cat([p.unsqueeze(0) if p.dim() == 1 else p for p in cache_values], dim=0).to(image_features.device)
                
                # Apply thresholds to create masks
                masks = ((prob_maps > neg_mask_thresholds[0]) & 
                        (prob_maps < neg_mask_thresholds[1])).float()
                
                # Compute weighted affinities
                exp_affinity = torch.exp((-1) * (beta - beta * affinity))
                
                # Initialize logits
                cache_logits = torch.zeros((img_features_norm.shape[0], num_classes), device=img_features_norm.device)
                
                # Apply mask for each class
                for c in range(num_classes):
                    if masks.shape[1] > c:  # Check if this class exists in the mask
                        class_mask = masks[:, c]
                        cache_logits[:, c] = exp_affinity @ class_mask
            else:
                # For positive cache - use class indices
                one_hot = F.one_hot(torch.tensor(cache_values, device=image_features.device), 
                                 num_classes=num_classes).float()
                
                # Compute weighted logits
                exp_affinity = torch.exp((-1) * (beta - beta * affinity))
                cache_logits = exp_affinity @ one_hot
            
            return alpha * cache_logits
        
        except Exception as e:
            print(f"Error in compute_cache_logits: {e}")
            # Return zeros as fallback
            if isinstance(clip_weights, list):
                num_classes = len(clip_weights)
            else:
                num_classes = clip_weights.size(1)
            return torch.zeros((1, num_classes), device=image_features.device)

def get_tensor_size(tensor):
    """Calculate the size of a PyTorch tensor in bytes."""
    if not isinstance(tensor, torch.Tensor):
        return 8  # Default size for non-tensors
    
    return tensor.element_size() * tensor.nelement()

def compute_real_cache_size(cache):
    """Compute the real memory size of a cache in bytes."""
    try:
        total_size = 0
        for class_entries in cache.values():
            for item in class_entries:
                # Calculate size of feature embedding (first element)
                feature_size = get_tensor_size(item[0]) if isinstance(item[0], torch.Tensor) else 8
                
                # Calculate size of metadata (remaining elements)
                metadata_size = sum(get_tensor_size(x) if isinstance(x, torch.Tensor) else 8 for x in item[1:])
                
                total_size += feature_size + metadata_size
        return total_size
    except Exception as e:
        print(f"Error calculating cache size: {e}")
        return 0

def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights, log_file, use_int8=False):
    with open(log_file, 'w') as log:
        with torch.no_grad():
            # Initialize variables
            pos_cache, neg_cache, accuracies = {}, {}, []
            total_lookup_time = 0.0
            total_inference_time = 0.0
            num_lookups = 0
            
            # Record starting memory
            start_memory = monitor_memory()
            log.write(f"Starting memory usage: {start_memory:.2f} MB\n")
            log.write(f"Using INT8 quantization: {use_int8}\n")

            # Unpack configuration parameters with safety checks
            pos_enabled = pos_cfg.get('enabled', False)
            neg_enabled = neg_cfg.get('enabled', False)
            
            if pos_enabled:
                pos_params = {
                    'shot_capacity': pos_cfg.get('shot_capacity', 1),
                    'alpha': pos_cfg.get('alpha', 1.0),
                    'beta': pos_cfg.get('beta', 1.0)
                }
            
            if neg_enabled:
                neg_params = {
                    'shot_capacity': neg_cfg.get('shot_capacity', 1),
                    'alpha': neg_cfg.get('alpha', 1.0),
                    'beta': neg_cfg.get('beta', 1.0),
                    'entropy_threshold': neg_cfg.get('entropy_threshold', {'lower': 0.3, 'upper': 0.8}),
                    'mask_threshold': neg_cfg.get('mask_threshold', {'lower': 0.03, 'upper': 1.0})
                }
                
                # Extract entropy and mask thresholds
                entropy_config = neg_params['entropy_threshold']
                mask_config = neg_params['mask_threshold']
                
                neg_entropy_threshold = (entropy_config['lower'] + entropy_config['upper']) / 2 if isinstance(entropy_config, dict) else 0.5
                neg_mask_thresholds = (mask_config['lower'], mask_config['upper']) if isinstance(mask_config, dict) else (0.03, 1.0)

            # Iterate through the dataset
            for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
                try:
                    # Start timing
                    start_time = time.time()
                    
                    # Move inputs to CUDA
                    images = images.cuda()
                    target = target.cuda()
                    
                    # Get features using the quantized model
                    # If quantized, encode_image returns (features_int8, scale, zero_point)
                    image_features = clip_model.encode_image(images)
                    
                    # Handle possible return formats
                    if isinstance(image_features, tuple) and len(image_features) == 3:
                        # We have quantized features (features_int8, scale, zero_point)
                        image_features_store = image_features  # Store quantized version
                        
                        # Dequantize for computation if the tensor supports it
                        if hasattr(image_features[0], 'dequantize'):
                            image_features_use = image_features[0].dequantize()
                        else:
                            # If not a quantized tensor, just use as is
                            image_features_use = image_features[0]
                    else:
                        # Regular non-quantized features
                        image_features_store = image_features
                        image_features_use = image_features
                    
                    # Normalize features for computation
                    image_features_norm = F.normalize(image_features_use, p=2, dim=1)
                    
                    # Compute logits with class weights
                    if isinstance(clip_weights, list):
                        class_weights = torch.stack(clip_weights, dim=1).to(image_features_norm.device)
                    else:
                        class_weights = clip_weights
                        
                    logits = 100. * image_features_norm @ class_weights
                    
                    # Get softmax probabilities
                    loss = F.softmax(logits, dim=1)
                    
                    # Get probability map and prediction
                    prob_map = loss.clone()
                    pred = torch.argmax(logits, dim=1)
                    
                    # Calculate entropy for negative cache
                    if neg_enabled:
                        try:
                            num_classes = len(clip_weights) if isinstance(clip_weights, list) else clip_weights.size(1)
                            entropy = -torch.sum(loss * torch.log2(loss + 1e-10), dim=1)
                            prop_entropy = float(entropy.mean() / np.log2(num_classes))
                        except Exception as e:
                            log.write(f"Error calculating entropy: {e}\n")
                            prop_entropy = 0.5  # Default middle value
                    
                    # Update caches
                    if pos_enabled:
                        update_cache(
                            pos_cache, 
                            pred[0],  # Use first prediction
                            [image_features_store, loss[0].mean()], 
                            pos_params['shot_capacity']
                        )
                    
                    if neg_enabled and prop_entropy > neg_entropy_threshold:
                        update_cache(
                            neg_cache, 
                            pred[0], 
                            [image_features_store, loss[0].mean(), prob_map[0]], 
                            neg_params['shot_capacity'],
                            True  # Include probability map
                        )
                    
                    # Compute final logits using cache
                    final_logits = logits.clone()
                    
                    # Measure lookup time
                    lookup_start = time.time()
                    
                    # Apply positive cache if enabled and not empty
                    if pos_enabled and pos_cache:
                        pos_logits = compute_cache_logits(
                            image_features_norm,  # Pass normalized features for computation
                            pos_cache, 
                            pos_params['alpha'], 
                            pos_params['beta'], 
                            clip_weights
                        )
                        final_logits += pos_logits
                    
                    # Apply negative cache if enabled and not empty
                    if neg_enabled and neg_cache:
                        neg_logits = compute_cache_logits(
                            image_features_norm,  # Pass normalized features for computation
                            neg_cache, 
                            neg_params['alpha'], 
                            neg_params['beta'], 
                            clip_weights, 
                            neg_mask_thresholds
                        )
                        final_logits -= neg_logits
                    
                    # Compute lookup time
                    lookup_end = time.time()
                    lookup_time = lookup_end - lookup_start
                    total_lookup_time += lookup_time
                    num_lookups += 1
                    
                    # Calculate accuracy
                    acc = cls_acc(final_logits, target)
                    accuracies.append(acc)
                    
                    # Compute total inference time
                    inference_time = time.time() - start_time
                    total_inference_time += inference_time
                    
                    # Log progress periodically
                    if i % 1000 == 0:
                        pos_cache_size = compute_real_cache_size(pos_cache)
                        neg_cache_size = compute_real_cache_size(neg_cache)
                        
                        avg_lookup_time = total_lookup_time / num_lookups if num_lookups > 0 else 0
                        avg_inference_time = total_inference_time / (i + 1)
                        
                        current_memory = monitor_memory()
                        
                        log.write(f"---- Iteration {i} ----\n")
                        log.write(f"Positive Cache Size: {pos_cache_size} bytes\n")
                        log.write(f"Negative Cache Size: {neg_cache_size} bytes\n")
                        log.write(f"Memory Usage: {current_memory:.2f} MB\n")
                        log.write(f"Memory Increase: {current_memory - start_memory:.2f} MB\n")
                        log.write(f"Average Cache Lookup Time: {avg_lookup_time:.6f} seconds\n")
                        log.write(f"Average Inference Time: {avg_inference_time:.6f} seconds\n")
                        log.write(f"Top 1 Accuracy: {sum(accuracies) / len(accuracies):.2f}%\n\n")
                        
                        # Force garbage collection
                        gc.collect()
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    log.write(f"Error in iteration {i}: {str(e)}\n")
                    print(f"Error in iteration {i}: {str(e)}")
                    continue
            
            # Calculate final accuracy
            avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0
            
            # Log final results
            log.write('=' * 50 + '\n')
            log.write(f"Final Top 1 Accuracy: {avg_acc:.2f}%\n")
            log.write(f"Final Memory Usage: {monitor_memory():.2f} MB\n")
            log.write('=' * 50 + '\n')
            
            return avg_acc

def main():
    # Parse arguments
    args = get_arguments()
    config_path = args.config
    use_int8 = args.use_int8
    
    # Print if using INT8
    if use_int8:
        print("Using INT8 quantization for model.")
    
    # Set random seed for reproducibility
    random.seed(1)
    torch.manual_seed(1)
    
    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone, jit=False)  # Disable JIT to allow quantization
    clip_model.eval()
    
    # Apply quantization if requested
    if use_int8:
        try:
            from clip.model import quantize_clip_model
            # Quantize the model - this uses the implementation from clip/model.py
            clip_model = quantize_clip_model(clip_model)
            print("Model quantized successfully using clip/model.py implementation.")
        except Exception as e:
            print(f"Error quantizing model: {e}")
            print("Continuing with non-quantized model.")
    
    # Process each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        
        # Get dataset configuration
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        
        # Build data loader and get class information
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        
        # Create text embeddings for zero-shot classification
        clip_weights = clip_classifier(classnames, template, clip_model)
        
        # Set up log file
        log_file = f"logs_{dataset_name}_tda_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Run test-time adaptation
        acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights, log_file, use_int8)
        
        print(f"{dataset_name} dataset processed. Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()