import random
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
import operator

import clip
from utils import *


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


def quantize_to_8bit(tensor):
    """Quantize tensor to 8-bit and return quantized tensor, scale, and zero point."""
    min_val, max_val = tensor.min(), tensor.max()
    if min_val == max_val:
        scale = 1.0
        zero_point = min_val
        quantized_tensor = torch.zeros_like(tensor, dtype=torch.uint8)
    else:
        scale = (max_val - min_val) / 255.0
        zero_point = min_val
        quantized_tensor = ((tensor - min_val) / scale).round().clamp(0, 255).to(torch.uint8)
    return quantized_tensor, scale, zero_point


def quantize_to_4bit(tensor):
    """Quantize tensor to 4-bit and return quantized tensor, scale, and zero point."""
    min_val, max_val = tensor.min(), tensor.max()
    if min_val == max_val:
        scale = 1.0
        zero_point = min_val
        quantized_tensor = torch.zeros_like(tensor, dtype=torch.uint8)
    else:
        scale = (max_val - min_val) / 15.0
        zero_point = min_val
        quantized_tensor = ((tensor - min_val) / scale).round().clamp(0, 15).to(torch.uint8)
    return quantized_tensor, scale, zero_point


def dequantize(tensor, scale, zero_point):
    """Reconstruct original tensor from quantized tensor."""
    return tensor.to(torch.float32) * scale + zero_point


def update_cache(cache, pred, features_loss, shot_capacity, quant_mode, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        
        # Quantize the item based on the selected mode
        if quant_mode == '8bit':
            quantized_item, scale, zero_point = quantize_to_8bit(item[0])
        elif quant_mode == '4bit':
            quantized_item, scale, zero_point = quantize_to_4bit(item[0])
        else:
            raise ValueError("Unsupported quantization mode!")

        quantized_item = [quantized_item, scale, zero_point] + item[1:]  # Append scale, zero point

        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(quantized_item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = quantized_item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [quantized_item]


def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, quant_mode, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                quantized_feature, scale, zero_point = item[0], item[1], item[2]
                dequantized_feature = dequantize(quantized_feature, scale, zero_point)

                # Ensure the tensor has the correct dimensions
                if dequantized_feature.dim() == 1:  # If it's a 1D tensor
                    dequantized_feature = dequantized_feature.unsqueeze(0)  # Add batch dimension
                cache_keys.append(dequantized_feature)

                if neg_mask_thresholds:
                    prob_map = item[3]
                    if prob_map.dim() == 0:  # Ensure prob_map is at least 1D
                        prob_map = prob_map.unsqueeze(0)
                    cache_values.append(prob_map)
                else:
                    cache_values.append(torch.tensor([class_index], device=image_features.device))

        # Handle empty cache
        if len(cache_keys) == 0 or len(cache_values) == 0:
            return torch.zeros_like(image_features)

        # Concatenate cache keys and values
        cache_keys = torch.cat(cache_keys, dim=0)  # Concatenate into 2D tensor
        if cache_keys.dim() != 2:
            raise ValueError(f"cache_keys has invalid dimensions: {cache_keys.shape}")  # Debugging assertion

        cache_keys = cache_keys.permute(1, 0).half()  # Permute and convert to Half precision
        image_features = image_features.half()

        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & 
                             (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
        else:
            cache_values = F.one_hot(torch.cat(cache_values, dim=0).to(torch.int64), num_classes=clip_weights.size(1)).cuda().half()

        # Compute affinity and logits
        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits



def compute_kv_table_size(cache, feature_dim, quant_mode):
    """Compute the total size of the KV cache in bytes."""
    key_size_per_row = feature_dim * (1 if quant_mode == '8bit' else 0.5)  # 8-bit = 1 byte, 4-bit = 0.5 byte
    metadata_size_per_row = 4  # Assuming loss (float32)
    row_size = key_size_per_row + metadata_size_per_row
    total_rows = sum(len(entries) for entries in cache.values())
    return int(total_rows * row_size)


def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights, feature_dim, quant_mode, log_file):
    with open(log_file, 'w') as log:
        pos_cache, neg_cache, accuracies = {}, {}, []

        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images, clip_model, clip_weights)
            target = target.cuda()

            if pos_cfg['enabled']:
                update_cache(pos_cache, pred, [image_features, loss], pos_cfg['shot_capacity'], quant_mode)

            if neg_cfg['enabled']:
                update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_cfg['shot_capacity'], quant_mode, include_prob_map=True)

            final_logits = clip_logits.clone()
            if pos_cache:
                final_logits += compute_cache_logits(image_features, pos_cache, pos_cfg['alpha'], pos_cfg['beta'], clip_weights, quant_mode)
            if neg_cache:
                final_logits -= compute_cache_logits(image_features, neg_cache, neg_cfg['alpha'], neg_cfg['beta'], clip_weights, quant_mode)

            acc = cls_acc(final_logits, target)
            accuracies.append(acc)

            # Monitor the KV table size
            if i % 1000 == 0:
                pos_cache_size = compute_kv_table_size(pos_cache, feature_dim, quant_mode)
                neg_cache_size = compute_kv_table_size(neg_cache, feature_dim,quant_mode)
                log.write(f"---- Iteration {i} ----\n")
                log.write(f"Positive Cache Size: {pos_cache_size} bytes\n")
                log.write(f"Negative Cache Size: {neg_cache_size} bytes\n")
                log.write(f"---- TDA's test accuracy: {sum(accuracies)/len(accuracies):.2f}. ----\n\n")

        log.write("---- Final Results ----\n")
        log.write(f"Positive Cache Size: {compute_kv_table_size(pos_cache, feature_dim,quant_mode)} bytes\n")
        log.write(f"Negative Cache Size: {compute_kv_table_size(neg_cache, feature_dim,quant_mode)} bytes\n")
        log.write(f"TDA's test accuracy: {sum(accuracies) / len(accuracies):.2f}.\n")

    return sum(accuracies) / len(accuracies)


def main():
    args = get_arguments()
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    dummy_image = torch.randn(1, 3, 224, 224).cuda()
    with torch.no_grad():
        feature_dim = clip_model.encode_image(dummy_image).shape[1]

    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        cfg = get_config_file(args.config, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model)

        log_file = f"logs_{dataset_name}_{args.quant_mode}Quantized_tda_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights, feature_dim, args.quant_mode, log_file)


if __name__ == "__main__":
    main()
