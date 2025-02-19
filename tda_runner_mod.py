import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
import operator

import clip
from utils import *
import time
import triton
import triton.language as tl

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


# @triton.jit
# def matmul_kernel(
#     A, B, C, 
#     stride_am, stride_ak,  # Strides for A
#     stride_bk, stride_bn,  # Strides for B
#     stride_cm, stride_cn,  # Strides for C
#     M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, 
#     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr
# ):
#     pid = tl.program_id(axis=0)  # Unique program ID (1D index)

#     # Compute number of program IDs along M and N
#     num_pid_m = tl.cdiv(M, BLOCK_M)  # Number of row-blocks
#     num_pid_n = tl.cdiv(N, BLOCK_N)  # Number of column-blocks

#     # Number of programs in a single group
#     num_pid_in_group = GROUP_SIZE_M * num_pid_n

#     # Determine group ID
#     group_id = pid // num_pid_in_group  # Which super-group this belongs to

#     # Compute first row index of the group
#     first_pid_m = group_id * GROUP_SIZE_M

#     # Handle last group case (if `num_pid_m` isn't a multiple of `GROUP_SIZE_M`)
#     group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

#     # Compute row-major traversal within groups
#     pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
#     pid_n = (pid % num_pid_in_group) // group_size_m

#     # Compute block memory offsets
#     off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
#     off_k = tl.arange(0, BLOCK_K)

#     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

#     # Correct memory pointer arithmetic
#     a_ptrs = A + (off_m[:, None] * stride_am) + (off_k[None, :] * stride_ak)
#     b_ptrs = B + (off_k[:, None] * stride_bk) + (off_n[None, :] * stride_bn)

#     for k in range(0, K, BLOCK_K):
#         is_last_iter = k + BLOCK_K >= K

#         a_mask = (off_m[:, None] < M) & (off_k[None, :] + k < K) if is_last_iter else None
#         b_mask = (off_k[:, None] + k < K) & (off_n[None, :] < N) if is_last_iter else None

#         A_shared = tl.load(a_ptrs + k * stride_ak, mask=a_mask, other=0)
#         B_shared = tl.load(b_ptrs + k * stride_bk, mask=b_mask, other=0)

#         acc += tl.dot(A_shared, B_shared)

#     # Compute final memory address for storing `C`
#     c_ptrs = C + (off_m[:, None] * stride_cm) + (off_n[None, :] * stride_cn)

#     tl.store(c_ptrs, acc, mask=(off_m[:, None] < M) & (off_n[None, :] < N))


    
# def quantize_item(item, mode='8bit'):
#     """
#     Quantize the entire item, including the feature vector, loss, and optional probability map.
    
#     Args:
#         item (list): A list containing [feature vector, loss, (optional) probability map].
#         mode (str): Quantization mode ('8bit' or '4bit').
    
#     Returns:
#         list: A list containing quantized components, each with scale and zero point.
#     """
#     quantized_item = []
#     quantization_mode = {
#         '8bit': {'max_level': 255, 'dtype': torch.quint8},
#         '4bit': {'max_level': 15, 'dtype': torch.quint4x2}
#     }

#     if mode not in quantization_mode:
#         raise ValueError("Unsupported quantization mode. Choose '8bit' or '4bit'.")

#     max_level = quantization_mode[mode]['max_level']
#     dtype = quantization_mode[mode]['dtype']

#     # Quantize each part of the item
#     for tensor in item:
#         if isinstance(tensor, torch.Tensor):  # Only quantize tensors
#             # Ensure the tensor is in Float32
#             tensor = tensor.to(torch.float32)
            
#             min_val, max_val = tensor.min().item(), tensor.max().item()  # Extract values as floats
#             if min_val == max_val:
#                 scale = 1.0
#                 zero_point = 0  # Set zero_point to 0 when min_val == max_val
#                 q_tensor = torch.quantize_per_tensor(torch.empty_like(tensor), scale=scale, zero_point=zero_point, dtype=dtype)
#             else:
#                 scale = (max_val - min_val) / max_level
#                 zero_point = int((0 - min_val) / scale)  # Convert zero_point to int
#                 q_tensor = torch.quantize_per_tensor(tensor, scale=scale, zero_point=zero_point, dtype=dtype)
#             quantized_item.append((q_tensor, scale, zero_point))
#         else:
#             quantized_item.append(tensor)  # Keep non-tensors as they are

#     return quantized_item

# def dequantize_item(quantized_item):
#     """
#     Dequantize the entire item.

#     Args:
#         quantized_item (list): A list containing quantized components with scale and zero point.
    
#     Returns:
#         list: A list of dequantized components.
#     """
#     dequantized_item = []
#     for part in quantized_item:
#         if isinstance(part, tuple) and len(part) == 3:  # Quantized part with (tensor, scale, zero_point)
#             q_tensor, scale, zero_point = part
#             dequantized_item.append(q_tensor.dequantize())
#         else:
#             dequantized_item.append(part)  # Non-quantized part

#     return dequantized_item

@triton.jit
def quantize_kernel(
    x_ptr,  # pointer to input
    q_ptr,  # pointer to output (quantized)
    scale_ptr,  # pointer to scale
    zero_ptr,  # pointer to zero point
    n_elements,  # number of elements
    BLOCK_SIZE: tl.constexpr,  # size of parallel block
    max_val: tl.constexpr,  # maximum quantized value
):
    # Compute current block
    pid = tl.program_id(0)
    
    # Create block offset
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load scale and zero point (single values)
    scale = tl.load(scale_ptr + 0)
    zero_point = tl.load(zero_ptr + 0)
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Quantize values
    quantized = tl.minimum(
        tl.maximum(
            tl.math.round((x / scale) + zero_point),  # Changed tl.round to tl.math.round
            0.0
        ),
        float(max_val)
    )
    
    # Store quantized values
    tl.store(q_ptr + offsets, quantized, mask=mask)


@triton.jit
def dequantize_kernel(
    q_ptr,  # pointer to input (quantized)
    x_ptr,  # pointer to output (dequantized)
    scale_ptr,  # pointer to scale
    zero_ptr,  # pointer to zero point
    n_elements,  # number of elements
    BLOCK_SIZE: tl.constexpr,  # size of parallel block
):
    # Compute current block
    pid = tl.program_id(0)
    
    # Create block offset
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load scale and zero point (single values)
    scale = tl.load(scale_ptr + 0)
    zero_point = tl.load(zero_ptr + 0)
    
    # Load quantized values
    q = tl.load(q_ptr + offsets, mask=mask, other=0.0)
    
    # Dequantize values
    dequantized = (q - zero_point) * scale
    
    # Store dequantized values
    tl.store(x_ptr + offsets, dequantized, mask=mask)

def quantize_item(item, mode='8bit'):
    """
    Quantize a list of tensors using Triton kernels.
    """
    assert isinstance(item, list), "Item must be a list"
    
    # Set quantization parameters
    max_val = 255 if mode == '8bit' else 15
    BLOCK_SIZE = 1024
    
    quantized_item = []
    
    for tensor in item:
        if isinstance(tensor, torch.Tensor):
            # Ensure tensor is on CUDA and contiguous
            tensor = tensor.cuda().contiguous().to(torch.float32)
            
            # Calculate quantization parameters
            min_val = tensor.min().item()
            max_val_tensor = tensor.max().item()
            
            if min_val == max_val_tensor:
                scale = 1.0
                zero_point = 0.0
                quantized_tensor = torch.zeros_like(tensor, dtype=torch.float32)
            else:
                scale = (max_val_tensor - min_val) / max_val
                zero_point = -min_val / scale
                
                # Prepare tensors
                quantized_tensor = torch.empty_like(tensor, dtype=torch.float32)
                scale_tensor = torch.tensor([scale], dtype=torch.float32, device=tensor.device)
                zero_point_tensor = torch.tensor([zero_point], dtype=torch.float32, device=tensor.device)
                
                # Calculate grid size
                n_elements = tensor.numel()
                grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
                
                # Launch kernel
                quantize_kernel[grid](
                    tensor,
                    quantized_tensor,
                    scale_tensor,
                    zero_point_tensor,
                    n_elements,
                    BLOCK_SIZE,
                    max_val,
                )
            
            quantized_item.append((quantized_tensor, scale, zero_point))
        else:
            quantized_item.append(tensor)
    
    return quantized_item

def dequantize_item(quantized_item):
    """
    Dequantize a list of quantized tensors using Triton kernels.
    """
    assert isinstance(quantized_item, list), "Quantized item must be a list"
    
    BLOCK_SIZE = 1024
    dequantized_item = []
    
    for item in quantized_item:
        if isinstance(item, tuple) and len(item) == 3:
            q_tensor, scale, zero_point = item
            
            # Ensure tensor is on CUDA and contiguous
            q_tensor = q_tensor.cuda().contiguous().to(torch.float32)
            
            # Prepare tensors
            dequantized_tensor = torch.empty_like(q_tensor, dtype=torch.float32)
            scale_tensor = torch.tensor([scale], dtype=torch.float32, device=q_tensor.device)
            zero_point_tensor = torch.tensor([zero_point], dtype=torch.float32, device=q_tensor.device)
            
            # Calculate grid size
            n_elements = q_tensor.numel()
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            
            # Launch kernel
            dequantize_kernel[grid](
                q_tensor,
                dequantized_tensor,
                scale_tensor,
                zero_point_tensor,
                n_elements,
                BLOCK_SIZE,
            )
            
            dequantized_item.append(dequantized_tensor)
        else:
            dequantized_item.append(item)
    
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
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        if not cache:
            return torch.zeros_like(image_features)

        cache_keys = []
        cache_values = []

        # Pre-dequantize cache entries
        for class_index in sorted(cache.keys()):
            for quantized_item in cache[class_index]:
                dequantized_item = dequantize_item(quantized_item)
                feature_vector = dequantized_item[0]

                if feature_vector.dim() == 1:
                    feature_vector = feature_vector.unsqueeze(0)
                cache_keys.append(feature_vector)

                if neg_mask_thresholds:
                    prob_map = dequantized_item[2]
                    if prob_map.dim() == 0:
                        prob_map = prob_map.unsqueeze(0)
                    cache_values.append(prob_map)
                else:
                    cache_values.append(torch.tensor([class_index], device=image_features.device))

        if len(cache_keys) == 0 or len(cache_values) == 0:
            return torch.zeros_like(image_features)

        # Concatenate cache keys and values
        cache_keys = torch.cat(cache_keys, dim=0).to(image_features.device).half()
        image_features = image_features.to(cache_keys.device).half()

        # Ensure correct dimensions
        if cache_keys.dim() == 3:
            cache_keys = cache_keys.squeeze(1)

        if cache_keys.shape[0] != image_features.shape[1]:
            cache_keys = cache_keys.permute(1, 0)

        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0).to(image_features.device)
            cache_values = (((cache_values > neg_mask_thresholds[0]) &
                           (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
        else:
            cache_values = F.one_hot(
                torch.cat(cache_values, dim=0), num_classes=clip_weights.size(1)
            ).cuda().half()

        # Compute affinity and logits
        affinity = torch.matmul(image_features, cache_keys)
        if cache_values.dim() == 1:
            cache_values = cache_values.unsqueeze(1)
        if affinity.shape[1] != cache_values.shape[0]:
            cache_values = cache_values.T

        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

        return alpha * cache_logits.squeeze(0)

# def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
#     """Compute logits using positive/negative cache with Triton optimization."""
    
#     with torch.no_grad():
#         if not cache:
#             return torch.zeros_like(image_features), 0.0, 0.0  # Return early if cache is empty

#         cache_keys = []
#         cache_values = []

#         # Pre-dequantize cache BEFORE timing lookup
#         for class_index in sorted(cache.keys()):
#             for quantized_item in cache[class_index]:
#                 dequantized_item = dequantize_item(quantized_item)
#                 feature_vector = dequantized_item[0]  # Extract dequantized feature vector

#                 # Ensure correct dimensions
#                 if feature_vector.dim() == 1:
#                     feature_vector = feature_vector.unsqueeze(0)
#                 cache_keys.append(feature_vector)

#                 if neg_mask_thresholds:
#                     prob_map = dequantized_item[2]
#                     if prob_map.dim() == 0:
#                         prob_map = prob_map.unsqueeze(0)
#                     cache_values.append(prob_map)
#                 else:
#                     cache_values.append(torch.tensor([class_index], device=image_features.device))


#         if len(cache_keys) == 0 or len(cache_values) == 0:
#             return torch.zeros_like(image_features), 0.0, 0.0

#         # Convert to CUDA tensors
#         cache_keys = torch.cat(cache_keys, dim=0).to(image_features.device).half()
#         image_features = image_features.to(cache_keys.device).half()

#         if cache_keys.dim() == 3:
#             cache_keys = cache_keys.squeeze(1)

#         if cache_keys.shape[0] != image_features.shape[1]:
#             cache_keys = cache_keys.permute(1, 0)  # Ensure alignment

#         if neg_mask_thresholds:
#             cache_values = torch.cat(cache_values, dim=0).to(image_features.device)
#             cache_values = (((cache_values > neg_mask_thresholds[0]) &
#                              (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
#         else:
#             cache_values = F.one_hot(
#                 torch.cat(cache_values, dim=0), num_classes=clip_weights.size(1)
#             ).cuda().half()

#         # **TRITON MATRIX MULTIPLICATION**
#         M, K = image_features.shape
#         K, N = cache_keys.shape
#         stride_am, stride_ak = image_features.stride()  # Row-major layout
#         stride_bk, stride_bn = cache_keys.stride()
#         BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 16  # Optimize based on GPU

#         grid = lambda META: (
#         triton.cdiv(M, META['BLOCK_M']),  # Number of row-blocks
#         triton.cdiv(N, META['BLOCK_N'])   # Number of column-blocks
#         )
        
#         affinity = torch.empty((M, N), device=image_features.device, dtype=torch.float16)
#         stride_cm, stride_cn = affinity.stride()
        
#         matmul_kernel[grid](
#             image_features, cache_keys, affinity,
#             stride_am, stride_ak,  # A strides
#             stride_bk, stride_bn,  # B strides
#             stride_cm, stride_cn,  # C strides
#             M=M, N=N, K=K,
#             BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
#             GROUP_SIZE_M=4
#             )

#         # Ensure `cache_values` has correct dimensions
#         if cache_values.dim() == 1:
#             cache_values = cache_values.unsqueeze(1)

#         if affinity.shape[1] != cache_values.shape[0]:  
#             cache_values = cache_values.T  # Fix dimension mismatch


#         # Compute logits
#         cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values


#         return alpha * cache_logits.squeeze(0)
    

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
            start_time = time.time()
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images, clip_model, clip_weights)
            target = target.cuda()

            if pos_cfg['enabled']:
                update_cache(pos_cache, pred, [image_features, loss], pos_cfg['shot_capacity'], quant_mode)

            if neg_cfg['enabled']:
                update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_cfg['shot_capacity'], quant_mode, include_prob_map=True)

            final_logits = clip_logits.clone()

            # Measure lookup time for cache  
            lookup_start =  time.time()      
            if pos_cache:
                pos_logits = compute_cache_logits(image_features, pos_cache, pos_cfg['alpha'], pos_cfg['beta'], clip_weights)
                final_logits += pos_logits  # Add positive cache logits

            if neg_cache:
                neg_mask_thresholds = (neg_cfg['mask_threshold']['lower'], neg_cfg['mask_threshold']['upper'])
                neg_logits = compute_cache_logits(image_features, neg_cache, neg_cfg['alpha'], neg_cfg['beta'], clip_weights, neg_mask_thresholds)
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
