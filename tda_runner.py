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

def compute_kv_table_size(cache, feature_dim, quantized=False):
    """
    Compute the total size of the KV cache in bytes.

    Args:
        cache (dict): The KV cache dictionary.
        feature_dim (int): Dimensionality of the feature vector (e.g., 512 for CLIP).
        quantized (bool): Whether the features are quantized (8-bit) or not (32-bit).
    
    Returns:
        int: Total size of the KV cache in bytes.
    """
    key_size_per_row = feature_dim * (1 if quantized else 4)  # 1 byte for int8, 4 bytes for float32
    metadata_size_per_row = 4  # Assuming loss (float32)
    row_size = key_size_per_row + metadata_size_per_row
    
    total_rows = sum(len(entries) for entries in cache.values())
    total_size = total_rows * row_size
    
    return total_size

def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights, feature_dim, quantized, log_file):
    with open(log_file, 'w') as log:
        with torch.no_grad():
            pos_cache, neg_cache, accuracies = {}, {}, []

            # Unpack all hyperparameters
            pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
            if pos_enabled:
                pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
            if neg_enabled:
                neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}

            # Test-time adaptation
            for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
                image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images, clip_model, clip_weights)
                target, prop_entropy = target.cuda(), get_entropy(loss, clip_weights)

                if pos_enabled:
                    update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'])

                if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
                    update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_params['shot_capacity'], True)

                final_logits = clip_logits.clone()
                if pos_enabled and pos_cache:
                    final_logits += compute_cache_logits(image_features, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
                if neg_enabled and neg_cache:
                    final_logits -= compute_cache_logits(image_features, neg_cache, neg_params['alpha'], neg_params['beta'], clip_weights, (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))

                acc = cls_acc(final_logits, target)  
                accuracies.append(acc)

                # Monitor the KV table size
                if i % 1000 == 0:
                    pos_cache_size = compute_kv_table_size(pos_cache, feature_dim, quantized=quantized)
                    neg_cache_size = compute_kv_table_size(neg_cache, feature_dim, quantized=quantized)
                    log.write(f"---- Iteration {i} ----\n")
                    log.write(f"Positive Cache Size: {pos_cache_size} bytes\n")
                    log.write(f"Negative Cache Size: {neg_cache_size} bytes\n")
                    log.write(f"---- TDA's test accuracy: {sum(accuracies)/len(accuracies):.2f}. ----\n\n")

            log.write("---- Final Results ----\n")
            log.write(f"Positive Cache Size: {compute_kv_table_size(pos_cache, feature_dim, quantized=quantized)} bytes\n")
            log.write(f"Negative Cache Size: {compute_kv_table_size(neg_cache, feature_dim, quantized=quantized)} bytes\n")
            log.write(f"TDA's test accuracy: {sum(accuracies) / len(accuracies):.2f}.\n")

    return sum(accuracies) / len(accuracies)

def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    # Dynamically retrieve feature dimension
    dummy_image = torch.randn(1, 3, 224, 224).cuda()  # Dummy input tensor
    with torch.no_grad():
        dummy_features = clip_model.encode_image(dummy_image)
        feature_dim = dummy_features.shape[1]

    quantized = False
    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    log_file = "results.log"

    # Run TDA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model)

        acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights, feature_dim, quantized, log_file)

if __name__ == "__main__":
    main()
