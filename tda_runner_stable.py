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
    parser.add_argument('--quantize', action='store_true', help='Quantize the model.')

    args = parser.parse_args()

    return args

def get_tensor_size(tensor):
    """Calculate the size of a PyTorch tensor in bytes."""
    if not isinstance(tensor, torch.Tensor):
        return 0
    
    return tensor.element_size() * tensor.nelement()


def compute_real_cache_size(cache):
    """Compute the real memory size of a cache in bytes, handling INT8 quantized features."""
    total_size = 0
    for class_entries in cache.values():
        for item in class_entries:
            # Handle feature tuple format: (int8_tensor, scale)
            features = item[0]
            if isinstance(features, tuple) and len(features) == 2:
                int8_tensor, scale = features
                if isinstance(int8_tensor, torch.Tensor):
                    total_size += get_tensor_size(int8_tensor)
                if isinstance(scale, torch.Tensor):
                    total_size += get_tensor_size(scale)
            elif isinstance(features, torch.Tensor):
                total_size += get_tensor_size(features)
            
            # Handle loss value (should be scalar now)
            if isinstance(item[1], torch.Tensor):
                total_size += get_tensor_size(item[1])
            
            # Handle prob_map if present (for negative cache)
            if len(item) > 2 and isinstance(item[2], torch.Tensor):
                total_size += get_tensor_size(item[2])
    
    return total_size

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with pure INT8 quantized features."""
    with torch.no_grad():
        # Extract features
        image_features = features_loss[0]
        device = image_features.device if not isinstance(image_features, tuple) else image_features[0].device
        
        # Ensure we're storing INT8 features
        if not isinstance(image_features, tuple):
            # Quantize to INT8
            feat_max = torch.max(torch.abs(image_features)).detach()
            feat_scale = feat_max / 127.0 if feat_max > 0 else torch.tensor(1.0, device=device)
            feat_int8 = torch.round(image_features / feat_scale).clamp(-127, 127).to(dtype=torch.int8, device=device)
            image_features = (feat_int8, feat_scale)
        
        # Extract loss value
        if torch.is_tensor(features_loss[1]):
            if features_loss[1].numel() > 1:
                loss_value = features_loss[1].mean().item()
            else:
                loss_value = features_loss[1].item()
        else:
            loss_value = features_loss[1]
        
        # Create cache item
        item = [image_features, loss_value]
        if include_prob_map and len(features_loss) > 2:
            item.append(features_loss[2])
            
        # Update cache with sorted insertion
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif loss_value < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=lambda x: x[1])
        else:
            cache[pred] = [item]
            
# def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
#     """Update cache with INT8 quantized features."""
#     with torch.no_grad():
#         # Get image features and device
#         image_features = features_loss[0]
#         if isinstance(image_features, tuple) and len(image_features) == 2:
#             # Already quantized
#             feat_int8, feat_scale = image_features
#             device = feat_int8.device
#         else:
#             device = image_features.device
#             # Quantize to INT8
#             feat_max = torch.max(torch.abs(image_features)).detach()
#             if feat_max > 0:
#                 feat_scale = feat_max / 127.0
#             else:
#                 feat_scale = torch.tensor(1.0, device=device)
            
#             # Round and clamp to INT8 range
#             feat_int8 = torch.round(image_features / feat_scale).clamp(-127, 127).to(dtype=torch.int8, device=device)
#             image_features = (feat_int8, feat_scale)
        
#         # Process loss value
#         if torch.is_tensor(features_loss[1]):
#             if features_loss[1].numel() > 1:
#                 loss_value = features_loss[1].mean().item()
#             else:
#                 loss_value = features_loss[1].item()
#         else:
#             loss_value = features_loss[1]
        
#         # Create cache item
#         item = [image_features, loss_value]
        
#         # Add probability map if needed
#         if include_prob_map and len(features_loss) > 2:
#             item.append(features_loss[2])
            
#         # Update cache
#         if pred in cache:
#             if len(cache[pred]) < shot_capacity:
#                 cache[pred].append(item)
#             elif loss_value < cache[pred][-1][1]:
#                 cache[pred][-1] = item
                
#             # Sort by loss value
#             cache[pred] = sorted(cache[pred], key=lambda x: x[1])
#         else:
#             cache[pred] = [item]
            
# def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
#     """Compute cache logits with INT8 quantization when possible."""
#     with torch.no_grad():
#         try:
#             # Get device
#             device = clip_weights.device
            
#             # Check if using quantized features
#             if isinstance(image_features, tuple) and len(image_features) == 2:
#                 img_int8, img_scale = image_features
#                 img_int8 = img_int8.to(device)
#                 if isinstance(img_scale, torch.Tensor):
#                     img_scale = img_scale.to(device)
#             else:
#                 # Fall back to float computation for non-quantized features
#                 return compute_cache_logits_float(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds)
            
#             # Initialize output tensor
#             batch_size = img_int8.shape[0]
#             logits = torch.zeros((batch_size, clip_weights.size(1)), device=device)
            
#             # Process each class
#             for cls_id, items in cache.items():
#                 if len(items) == 0:
#                     continue
                    
#                 cls_sim = torch.zeros(batch_size, device=device)
#                 total_weight = 0.0
                
#                 for item in items:
#                     feat_data = item[0]
#                     loss_val = item[1]
                    
#                     # Extract quantized feature
#                     if isinstance(feat_data, tuple) and len(feat_data) == 2:
#                         feat_int8, feat_scale = feat_data
#                         feat_int8 = feat_int8.to(device)
#                         if isinstance(feat_scale, torch.Tensor):
#                             feat_scale = feat_scale.to(device)
                            
#                         # Use the triton kernel
#                         try:
#                             from utils import triton_int8_matmul
#                             sim = triton_int8_matmul(img_int8, img_scale, feat_int8, feat_scale)
#                         except Exception as e:
#                             # Fallback to float matmul with explicit type conversion
#                             print(f"Triton kernel error: {e}")
#                             img_float = img_int8.float() * img_scale
#                             feat_float = feat_int8.float() * feat_scale
#                             sim = torch.matmul(img_float, feat_float.t())
#                     else:
#                         # Handle non-quantized feature with explicit type conversion
#                         feat = feat_data.to(device).float()
#                         img_float = img_int8.float() * img_scale
#                         sim = torch.matmul(img_float, feat.t())
                    
#                     # Apply mask for negative samples
#                     if neg_mask_thresholds and len(item) > 2:
#                         mask = item[2].to(device)
#                         mask = ((mask > neg_mask_thresholds[0]) & 
#                               (mask < neg_mask_thresholds[1])).float()
#                         mask_val = mask.mean().item()
#                         sim = sim * mask_val
                    
#                     # Accumulate weighted similarity
#                     weight = 1.0 / loss_val
#                     cls_sim += sim.squeeze(1) * weight
#                     total_weight += weight
                
#                 # Apply parameters
#                 if total_weight > 0:
#                     logits[:, cls_id] = alpha * (cls_sim / total_weight) * beta
            
#             return logits
                
#         except Exception as e:
#             print(f"INT8 computation failed with: {e}. Falling back to float.")
#             return compute_cache_logits_float(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds)
 
def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Pure Triton INT8 cache logits computation without fallbacks."""
    with torch.no_grad():
        # Get device
        device = clip_weights.device
        
        # Ensure we're working with INT8 quantized features
        if isinstance(image_features, tuple) and len(image_features) == 2:
            img_int8, img_scale = image_features
            img_int8 = img_int8.to(device)
            if isinstance(img_scale, torch.Tensor):
                img_scale = img_scale.to(device)
        else:
            # Quantize on-the-fly if needed
            image_features = image_features.to(device)
            feat_max = torch.max(torch.abs(image_features)).detach()
            feat_scale = feat_max / 127.0 if feat_max > 0 else torch.tensor(1.0, device=device)
            img_int8 = torch.round(image_features / feat_scale).clamp(-127, 127).to(dtype=torch.int8, device=device)
            img_scale = feat_scale
        
        # Empty cache check
        if not cache:
            return torch.zeros((img_int8.shape[0], clip_weights.size(1)), device=device)
        
        # Initialize output tensor
        batch_size = img_int8.shape[0]
        logits = torch.zeros((batch_size, clip_weights.size(1)), device=device)
        
        # Process each class
        for cls_id, items in cache.items():
            if len(items) == 0:
                continue
                
            cls_sim = torch.zeros(batch_size, device=device)
            total_weight = 0.0
            
            for item in items:
                feat_data = item[0]
                loss_val = item[1]
                
                # Ensure we have INT8 features
                if isinstance(feat_data, tuple) and len(feat_data) == 2:
                    feat_int8, feat_scale = feat_data
                    feat_int8 = feat_int8.to(device)
                    if isinstance(feat_scale, torch.Tensor):
                        feat_scale = feat_scale.to(device)
                else:
                    # Quantize if not already done - part of the pure INT8 path
                    feat = feat_data.to(device)
                    feat_max = torch.max(torch.abs(feat)).detach()
                    feat_scale = feat_max / 127.0 if feat_max > 0 else torch.tensor(1.0, device=device)
                    feat_int8 = torch.round(feat / feat_scale).clamp(-127, 127).to(dtype=torch.int8, device=device)
                
                # Use Triton kernel directly - no try/except or fallback
                from utils import triton_int8_matmul
                sim = triton_int8_matmul(img_int8, img_scale, feat_int8, feat_scale)
                
                # Apply mask for negative samples
                if neg_mask_thresholds and len(item) > 2:
                    mask = item[2].to(device)
                    mask = ((mask > neg_mask_thresholds[0]) & 
                            (mask < neg_mask_thresholds[1])).float()
                    mask_val = mask.mean()
                    sim = sim * mask_val
                
                # Accumulate weighted similarity
                weight = 1.0 / loss_val if loss_val > 0 else 1.0
                cls_sim += sim.squeeze(1) * weight
                total_weight += weight
            
            # Apply parameters
            if total_weight > 0:
                logits[:, cls_id] = alpha * (cls_sim / total_weight) * beta
        
        return logits
           
def compute_cache_logits_float(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute cache logits with float precision with dtype handling."""
    with torch.no_grad():
        device = clip_weights.device
        
        # Extract image_features if it's a quantized tuple
        if isinstance(image_features, tuple) and len(image_features) == 2:
            img_int8, img_scale = image_features
            image_features = img_int8.float() * img_scale  # Always use float32
        else:
            # Convert to float32 explicitly
            image_features = image_features.float()
        
        # Ensure image_features is on the correct device
        image_features = image_features.to(device)
        
        # Initialize output tensor
        batch_size = image_features.shape[0]
        logits = torch.zeros((batch_size, clip_weights.size(1)), device=device)
        
        # Process each class
        for cls_id, items in cache.items():
            if len(items) == 0:
                continue
                
            # Process each cached item
            cls_sim = torch.zeros(batch_size, device=device)
            total_weight = 0.0
            
            for item in items:
                # Extract feature - handle both tensor and tuple formats
                feature = item[0]
                
                # Handle quantized features stored as tuples
                if isinstance(feature, tuple) and len(feature) == 2:
                    feat_int8, feat_scale = feature
                    # Convert to float32 explicitly
                    feature = feat_int8.float() * feat_scale
                else:
                    # Convert to float32 explicitly
                    feature = feature.float()
                
                # Ensure feature is on correct device
                feature = feature.to(device)
                
                # Get loss value for weighting
                loss_val = item[1]
                weight = 1.0 / loss_val
                
                # Compute similarity
                sim = torch.matmul(image_features, feature.t())
                
                # Handle negative masking if needed
                if neg_mask_thresholds and len(item) > 2:
                    mask = item[2].to(device)
                    mask = ((mask > neg_mask_thresholds[0]) & 
                           (mask < neg_mask_thresholds[1])).float()
                    mask_val = mask.mean().item()
                    sim = sim * mask_val
                
                # Accumulate weighted similarity
                cls_sim += sim.squeeze(1) * weight
                total_weight += weight
            
            # Apply parameters
            if total_weight > 0:
                logits[:, cls_id] = alpha * (cls_sim / total_weight) * beta
        
        return logits

def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights, log_file=None):
    with open(log_file, 'w') as log:
        
        with torch.no_grad():
            pos_cache, neg_cache, accuracies = {}, {}, []
            
            start_time = time.time()
            total_lookup_time = 0.0
            total_inference_time = 0.0
            num_lookups = 0
            
            #Unpack all hyperparameters
            pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
            if pos_enabled:
                pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
            if neg_enabled:
                neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}

            #Test-time adaptation
            for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
                start_inference = time.time()
                
                image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images ,clip_model, clip_weights)
                target, prop_entropy = target.cuda(), get_entropy(loss, clip_weights)
                    
                if pos_enabled:
                    update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'])

                if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
                    update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_params['shot_capacity'], True)

                final_logits = clip_logits.clone()
                
                lookup_start = time.time()
                
                if pos_enabled and pos_cache:
                    final_logits += compute_cache_logits(image_features, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
                if neg_enabled and neg_cache:
                    final_logits -= compute_cache_logits(image_features, neg_cache, neg_params['alpha'], neg_params['beta'], clip_weights, (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))

                end_lookup = time.time()
                lookup_time = end_lookup-lookup_start
                total_lookup_time += lookup_time
                num_lookups += 1
                
                acc = cls_acc(final_logits, target)  
                accuracies.append(acc)
                
                end_inference = time.time()
                inference_time = end_inference - start_inference
                total_inference_time += inference_time

                if i%1000==0:
                    pos_cache_size = compute_real_cache_size(pos_cache)
                    neg_cache_size = compute_real_cache_size(neg_cache)
                    
                    avg_lookup_time = total_lookup_time / num_lookups if num_lookups > 0 else 0
                    avg_inference_time = total_inference_time / (i + 1)
                    
                    # Log the results
                    log.write(f"---- Iteration {i} ----\n")
                    log.write(f"Positive Cache Size: {pos_cache_size} bytes\n")
                    log.write(f"Negative Cache Size: {neg_cache_size} bytes\n")
                    log.write(f"Average Lookup Time: {avg_lookup_time:.6f} seconds\n")
                    log.write(f"Average Inference Time: {avg_inference_time:.6f} seconds\n")
                    log.write(f"Accuracy: {sum(accuracies)/len(accuracies):.2f}\n")
                    log.write("\n")
                                 
                    print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
            print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))   
            log.write(f"Final accuracy: {sum(accuracies)/len(accuracies):.2f}. Total time: {time.time() - start_time:.2f}.\n")
            return sum(accuracies)/len(accuracies)

    
def main():
    args = get_arguments()
    config_path = args.config
    quantize = args.quantize
    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone , quantize=quantize)
    clip_model.eval().cuda()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)
    
    from utils import precompile_triton_kernels
    precompile_triton_kernels()
    
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
        if quantize:
            log_file = f"logs_{dataset_name}_tda_quantized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights, log_file=log_file)

if __name__ == "__main__":
    main()