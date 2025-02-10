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
    """Get arguments for the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='Settings of TDA on specific dataset in YAML format.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')
    parser.add_argument('--quantization-mode', dest='quant_mode', type=str, default='8bit', help='Choose quantization mode: 8bit or 4bit.')

    args = parser.parse_args()
    return args


def quantize_item(item, mode='8bit'):
    """
    Quantize the entire item, including the feature vector, loss, and optional probability map.
    
    Args:
        item (list): A list containing [feature vector, loss, (optional) probability map].
        mode (str): Quantization mode ('8bit' or '4bit').
    
    Returns:
        list: A list containing quantized components, each with scale and zero point.
    """
    quantized_item = []
    quantization_mode = {
        '8bit': {'max_level': 255, 'dtype': torch.quint8},
        '4bit': {'max_level': 15, 'dtype': torch.quint4x2}
    }

    if mode not in quantization_mode:
        raise ValueError("Unsupported quantization mode. Choose '8bit' or '4bit'.")

    max_level = quantization_mode[mode]['max_level']
    dtype = quantization_mode[mode]['dtype']

    # Quantize each part of the item
    for tensor in item:
        if isinstance(tensor, torch.Tensor):  # Only quantize tensors
            # Ensure the tensor is in Float32
            tensor = tensor.to(torch.float32)
            
            min_val, max_val = tensor.min().item(), tensor.max().item()  # Extract values as floats
            if min_val == max_val:
                scale = 1.0
                zero_point = 0  # Set zero_point to 0 when min_val == max_val
                q_tensor = torch.quantize_per_tensor(torch.empty_like(tensor), scale=scale, zero_point=zero_point, dtype=dtype)
            else:
                scale = (max_val - min_val) / max_level
                zero_point = int((0 - min_val) / scale)  # Convert zero_point to int
                q_tensor = torch.quantize_per_tensor(tensor, scale=scale, zero_point=zero_point, dtype=dtype)
            quantized_item.append((q_tensor, scale, zero_point))
        else:
            quantized_item.append(tensor)  # Keep non-tensors as they are

    return quantized_item


def dequantize_item(quantized_item):
    """
    Dequantize the entire item.

    Args:
        quantized_item (list): A list containing quantized components with scale and zero point.
    
    Returns:
        list: A list of dequantized components.
    """
    dequantized_item = []
    for part in quantized_item:
        if isinstance(part, tuple) and len(part) == 3:  # Quantized part with (tensor, scale, zero_point)
            q_tensor, scale, zero_point = part
            dequantized_item.append(q_tensor.dequantize())
        else:
            dequantized_item.append(part)  # Non-quantized part

    return dequantized_item

def update_cache(cache, pred, features_loss, shot_capacity, quant_mode, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        # Prepare the full item (feature vector, loss, and optional probability map)
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        
        # Quantize the entire item
        quantized_item = quantize_item(item, mode=quant_mode)

        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(quantized_item)
            elif features_loss[1] < cache[pred][-1][1][0].dequantize().item():
                cache[pred][-1] = quantized_item
            cache[pred] = sorted(cache[pred], key=lambda x: x[1][0].dequantize().item())  # Sort by dequantized loss value
        else:
            cache[pred] = [quantized_item]



def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache and measure lookup time (excluding dequantization)."""
    with torch.no_grad():
        if not cache:
            return torch.zeros_like(image_features), 0.0  # Return early if cache is empty**

        cache_keys = []
        cache_values = []

        # **Pre-dequantize cache entries BEFORE timing the lookup
        for class_index in sorted(cache.keys()):
            for quantized_item in cache[class_index]:
                dequantized_item = dequantize_item(quantized_item)  # Pre-dequantization
                feature_vector = dequantized_item[0]  # Extract dequantized feature vector

                # Ensure correct dimensions
                if feature_vector.dim() == 1:
                    feature_vector = feature_vector.unsqueeze(0)  # Add batch dimension
                cache_keys.append(feature_vector)

                if neg_mask_thresholds:
                    prob_map = dequantized_item[2]
                    if prob_map.dim() == 0:
                        prob_map = prob_map.unsqueeze(0)
                    cache_values.append(prob_map)
                else:
                    cache_values.append(torch.tensor([class_index], device=image_features.device))

        # Start timing after dequantization**
        start_time = time.perf_counter()

        # Handle empty cache case **again**
        if len(cache_keys) == 0 or len(cache_values) == 0:
            return torch.zeros_like(image_features), 0.0

        # Concatenate cache keys and values at once
        cache_keys = torch.cat(cache_keys, dim=0).to(image_features.device).half()
        image_features = image_features.to(cache_keys.device).half()

        # Ensure correct dimensions
        if cache_keys.dim() == 3:
            cache_keys = cache_keys.squeeze(1)  # Remove singleton dimension if present

        if cache_keys.shape[0] != image_features.shape[1]:
            cache_keys = cache_keys.permute(1, 0)  # Ensure alignment with `image_features`

        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0).to(image_features.device)
            cache_values = (((cache_values > neg_mask_thresholds[0]) &
                             (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
        else:
            cache_values = F.one_hot(
                torch.cat(cache_values, dim=0), num_classes=clip_weights.size(1)
            ).cuda().half()

        # Compute affinity matrix**
        affinity = torch.matmul(image_features, cache_keys)  #  **Fix: Ensure valid matrix multiplication**

        # 🔹 **Fix: Ensure `cache_values` has matching dimensions**
        if cache_values.dim() == 1:
            cache_values = cache_values.unsqueeze(1)  # Convert (N,) to (N,1) if needed

        if affinity.shape[1] != cache_values.shape[0]:  
            cache_values = cache_values.T  # Fix dimension mismatch for `@` operation

        lookup_time = time.perf_counter() - start_time

        # Compute logits
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

        inference_time = time.perf_counter() - start_time
        # 🔥 **End measuring lookup time (ONLY lookup + computation)**
        

        return alpha * cache_logits.squeeze(0), lookup_time , inference_time  # Return logits + lookup time**

    

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
    #print(tensor.element_size())
    return tensor.element_size() * tensor.numel()


def compute_real_cache_size(cache):
    """
    Compute the real memory size of a cache (positive or negative) in bytes.
    
    Args:
        cache (dict): The cache dictionary, where each entry is a list of quantized items.
        
    Returns:
        int: Total memory size of the cache in bytes.
    """
    total_size = 0
    for class_entries in cache.values():
        for item in class_entries:
            if isinstance(item[0], tuple) and isinstance(item[0][0], torch.Tensor):
                # Extract the quantized tensor and calculate its size
                quantized_tensor = item[0][0]
                feature_size = get_tensor_size(quantized_tensor)  # Quantized feature embedding
                # Add the size of metadata (scale and zero point)
                scale_size = get_tensor_size(torch.tensor(item[0][1]))  # Scale
                zero_point_size = get_tensor_size(torch.tensor(item[0][2]))  # Zero point
                total_size += feature_size + scale_size + zero_point_size
            else:
                # Handle other metadata if present (e.g., loss)
                total_size += sum(get_tensor_size(x) for x in item[1:] if isinstance(x, torch.Tensor))
    return total_size



def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights, quant_mode, log_file):
    with open(log_file, 'w') as log:
        pos_cache, neg_cache, accuracies = {}, {}, []
        total_lookup_time = 0.0
        total_inference_time = 0.0
        num_lookups = 0

        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images, clip_model, clip_weights)
            target = target.cuda()

            if pos_cfg['enabled']:
                update_cache(pos_cache, pred, [image_features, loss], pos_cfg['shot_capacity'], quant_mode)

            if neg_cfg['enabled']:
                update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_cfg['shot_capacity'], quant_mode, include_prob_map=True)

            final_logits = clip_logits.clone()

            # Measure lookup time for cache
            pos_lookup_time = 0.0
            neg_lookup_time = 0.0
            pos_inference_time = 0.0
            neg_inference_time = 0.0

            if pos_cache:
                pos_logits, pos_lookup_time, pos_inference_time = compute_cache_logits(image_features, pos_cache, pos_cfg['alpha'], pos_cfg['beta'], clip_weights)
                final_logits += pos_logits  # Add positive cache logits

            if neg_cache:
                neg_mask_thresholds = (neg_cfg['mask_threshold']['lower'], neg_cfg['mask_threshold']['upper'])
                neg_logits, neg_lookup_time,neg_inference_time = compute_cache_logits(image_features, neg_cache, neg_cfg['alpha'], neg_cfg['beta'], clip_weights, neg_mask_thresholds)
                final_logits -= neg_logits  # Subtract negative cache logits

            # Accumulate total lookup time
            total_lookup_time += pos_lookup_time + neg_lookup_time
            num_lookups += 1

            acc = cls_acc(final_logits, target)
            accuracies.append(acc)

            total_inference_time += pos_inference_time + neg_inference_time  # Measure total inference time

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
                log.write(f"---- TDA's test accuracy: {sum(accuracies)/len(accuracies):.2f}. ----\n\n")

        # Final logging
        avg_lookup_time = total_lookup_time / num_lookups if num_lookups > 0 else 0
        avg_inference_time = total_inference_time / len(loader)

        log.write("---- Final Results ----\n")
        log.write(f"Positive Cache Size: {compute_real_cache_size(pos_cache)} bytes\n")
        log.write(f"Negative Cache Size: {compute_real_cache_size(neg_cache)} bytes\n")
        log.write(f"Average Cache Lookup Time: {avg_lookup_time:.6f} seconds\n")
        log.write(f"Average Inference Time: {avg_inference_time:.6f} seconds\n")
        log.write(f"TDA's test accuracy: {sum(accuracies) / len(accuracies):.2f}.\n")

    return sum(accuracies) / len(accuracies)



def main():
    args = get_arguments()
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        cfg = get_config_file(args.config, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model)

        log_file = f"logs_{dataset_name}_{args.quant_mode}Quantized_tda_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights, args.quant_mode, log_file)


if __name__ == "__main__":
    main()
