import os
import yaml
import torch
import math
import numpy as np
import clip
from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader, AugMixAugmenter
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def get_entropy(loss, clip_weights):
    """Calculate entropy of prediction distribution."""
    num_classes = clip_weights.size(1)
    max_entropy = np.log(float(num_classes))  
    
    # Ensure we're calculating a single scalar value
    if loss.numel() > 1:
        mean_loss = loss.mean()
    else:
        mean_loss = loss
    
    #scaling = 1.0 / max_entropy
    raw_entropy = float(mean_loss / max_entropy)
    return raw_entropy

def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def cls_acc(output, target):
    """Calculate classification accuracy."""
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).sum().item() / target.size(0) * 100.0


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def get_clip_logits(images, clip_model, clip_weights):
    """Get CLIP logits with explicit device handling."""
    with torch.no_grad():
        # Get device from model
        device = next(clip_model.parameters()).device
        
        # Ensure images are on the same device
        images = images.to(device)
        
        # Ensure clip_weights are on the same device
        clip_weights = clip_weights.to(device)
        
        # Forward pass
        image_features_result = clip_model.encode_image(images)
        
        # Normalize
        image_features = image_features_result / image_features_result.norm(dim=-1, keepdim=True)
        
        # Calculate logits
        clip_logits = 100.0 * image_features @ clip_weights
        
        # Calculate loss
        loss = 1.0 / clip_logits.softmax(dim=-1)
        
        # Get prediction
        pred = clip_logits.argmax(dim=-1).item()
        
        # Create probability map
        prob_map = clip_logits.softmax(dim=-1).squeeze(0)
        
        return image_features, clip_logits, loss, prob_map, pred

def get_ood_preprocess():
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)

    return aug_preprocess


def get_config_file(config_path, dataset_name):
    if dataset_name == "I":
        config_name = "imagenet.yaml"
    elif dataset_name in ["A", "V", "R", "S"]:
        config_name = f"imagenet_{dataset_name.lower()}.yaml"
    else:
        config_name = f"{dataset_name}.yaml"
    
    config_file = os.path.join(config_path, config_name)
    
    with open(config_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")

    return cfg


def build_test_data_loader(dataset_name, root_path, preprocess):
    if dataset_name == 'I':
        dataset = ImageNet(root_path, preprocess)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=1, num_workers=8, shuffle=True)
    
    elif dataset_name in ['A','V','R','S']:
        preprocess = get_ood_preprocess()
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)

    elif dataset_name in ['caltech101','dtd','eurosat','fgvc','food101','oxford_flowers','oxford_pets','stanford_cars','sun397','ucf101']:
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)
    
    else:
        raise "Dataset is not from the chosen list"
    
    return test_loader, dataset.classnames, dataset.template

import triton
import triton.language as tl
import torch

# Define the kernel
@triton.jit
def int8_matmul_kernel(
    A, B, C,
    A_scale, B_scale,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """Pure INT8 matrix multiplication kernel."""
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Calculate program ID for different dimensions
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Calculate offsets for blocks
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Create pointers to memory
    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    
    # Matmul loop
    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        b_mask = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)
        
        # Load blocks with mask
        a = tl.load(a_ptrs, mask=a_mask, other=0)
        b = tl.load(b_ptrs, mask=b_mask, other=0)
        
        # Compute matrix multiplication and accumulate
        acc += tl.dot(a, b)
        
        # Update pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Apply scaling
    a_scale_val = tl.load(A_scale)
    b_scale_val = tl.load(B_scale)
    acc_float = acc.to(tl.float32) * (a_scale_val * b_scale_val)
    
    # Store results
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc_float, mask=c_mask)

# Main function - no fallbacks
def triton_int8_matmul(A_int8, A_scale, B_int8, B_scale):
    """Pure INT8 matrix multiplication without fallbacks."""
    device = A_int8.device
    
    # Ensure inputs are on the correct device
    B_int8 = B_int8.to(device)
    
    # Convert scales to tensors if needed
    if not isinstance(A_scale, torch.Tensor):
        A_scale = torch.tensor([float(A_scale)], device=device)
    else:
        A_scale = A_scale.to(device)
    
    if not isinstance(B_scale, torch.Tensor):
        B_scale = torch.tensor([float(B_scale)], device=device)
    else:
        B_scale = B_scale.to(device)
    
    # Get dimensions
    M, K = A_int8.shape
    
    # Handle different shapes of B
    if B_int8.dim() == 1:
        B_int8 = B_int8.unsqueeze(0)
    
    if B_int8.shape[-1] != K:  # Ensure dimensions are compatible
        if B_int8.shape[0] == K:
            B_int8 = B_int8.t().contiguous()
        else:
            raise ValueError(f"Incompatible dimensions: A={A_int8.shape}, B={B_int8.shape}")
    
    N = B_int8.shape[0]
    
    # Create output tensor
    C_float = torch.empty((M, N), dtype=torch.float32, device=device)
    
    # Use these block sizes consistently
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 64
    
    # Compute grid
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    # Prepare B for matmul - transpose once
    B_transposed = B_int8.t().contiguous()
    
    # Launch kernel - no exception handling for pure execution path
    int8_matmul_kernel[grid](
        A_int8, B_transposed, C_float,
        A_scale, B_scale,
        M, N, K,
        A_int8.stride(0), A_int8.stride(1),
        B_transposed.stride(0), B_transposed.stride(1),
        C_float.stride(0), C_float.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C_float

def precompile_triton_kernels():
    """Precompile triton kernels for common sizes to avoid JIT overhead."""
    print("Precompiling Triton INT8 kernels...")
    
    # Test different sizes of matrices
    sizes = [(1, 512), (1, 1024), (1, 2048), (2, 1024)]
    
    device = torch.device('cuda')
    for m, k in sizes:
        for n in [1, 4, 16, 64, 512]:
            A = torch.randint(-127, 127, (m, k), dtype=torch.int8, device=device)
            B = torch.randint(-127, 127, (n, k), dtype=torch.int8, device=device)
            A_scale = torch.tensor([0.1], device=device)
            B_scale = torch.tensor([0.1], device=device)
            
            # Run once to compile
            _ = triton_int8_matmul(A, A_scale, B, B_scale)
    
    print("Triton kernels precompiled successfully")

# import triton
# import triton.language as tl

# # Global cache for kernels and buffers
# _KERNEL_CACHE = {}
# _BUFFER_CACHE = {}

# @triton.jit
# def int8_matmul_kernel(
#     A, B, C,
#     A_scale, B_scale,
#     M, N, K,
#     stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
#     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
# ):
#     """Simplified INT8 matrix multiplication kernel - focuses on performance."""
#     # Program ID
#     pid = tl.program_id(0)
#     num_pid_m = tl.cdiv(M, BLOCK_M)
#     num_pid_n = tl.cdiv(N, BLOCK_N)
    
#     # Calculate program ID for different dimensions
#     pid_m = pid // num_pid_n
#     pid_n = pid % num_pid_n
    
#     # Calculate offsets for blocks
#     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
#     offs_k = tl.arange(0, BLOCK_K)
    
#     # Create pointers to memory
#     a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
#     b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
#     # Initialize accumulator
#     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    
#     # Matmul loop
#     for k in range(0, K, BLOCK_K):
#         a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
#         b_mask = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)
        
#         # Load blocks with mask
#         a = tl.load(a_ptrs, mask=a_mask, other=0)
#         b = tl.load(b_ptrs, mask=b_mask, other=0)
        
#         # Compute matrix multiplication and accumulate
#         acc += tl.dot(a, b)
        
#         # Update pointers
#         a_ptrs += BLOCK_K * stride_ak
#         b_ptrs += BLOCK_K * stride_bk
    
#     # Load and apply scaling
#     a_scale_val = tl.load(A_scale)
#     b_scale_val = tl.load(B_scale)
#     acc_float = acc.to(tl.float32) * (a_scale_val * b_scale_val)
    
#     # Store results
#     c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
#     c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
#     tl.store(c_ptrs, acc_float, mask=c_mask)

# def triton_int8_matmul(A_int8, A_scale, B_int8, B_scale):
#     """Optimized INT8 matrix multiplication with proper error handling."""
#     device = A_int8.device
    
#     # Ensure inputs are on the same device
#     if B_int8.device != device:
#         B_int8 = B_int8.to(device)
    
#     if isinstance(A_scale, torch.Tensor) and A_scale.device != device:
#         A_scale = A_scale.to(device)
    
#     if isinstance(B_scale, torch.Tensor) and B_scale.device != device:
#         B_scale = B_scale.to(device)
    
#     # Get dimensions
#     M, K = A_int8.shape
    
#     # Handle different shapes of B
#     if B_int8.dim() == 1:
#         B_int8 = B_int8.unsqueeze(0)
#         N = 1
#     else:
#         N = B_int8.shape[0]
    
#     # Use PyTorch for small matrices (optimize threshold based on your hardware)
#     if M * N * K < 4096:
#         # Simple matmul with tensor operations
#         A_scale_val = A_scale if isinstance(A_scale, torch.Tensor) else torch.tensor(A_scale, device=device)
#         B_scale_val = B_scale if isinstance(B_scale, torch.Tensor) else torch.tensor(B_scale, device=device)
        
#         # Efficient matmul with minimal conversion
#         return torch.matmul(A_int8.to(torch.float32), B_int8.transpose(0, 1).to(torch.float32)) * (A_scale_val * B_scale_val)
    
#     # Create output tensor
#     C_float = torch.empty((M, N), dtype=torch.float32, device=device)
    
#     # Prepare scale tensors
#     A_scale_tensor = A_scale if isinstance(A_scale, torch.Tensor) else torch.tensor([A_scale], device=device)
#     B_scale_tensor = B_scale if isinstance(B_scale, torch.Tensor) else torch.tensor([B_scale], device=device)
    
#     # Use optimal block sizes for current hardware
#     BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 64
    
#     # Compute grid
#     grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
#     # Prepare B for matmul (transpose)
#     B_transposed = B_int8.transpose(0, 1).contiguous() if B_int8.dim() > 1 else B_int8.contiguous()
    
#     try:
#         # Launch kernel
#         int8_matmul_kernel[grid](
#             A_int8, B_transposed, C_float,
#             A_scale_tensor, B_scale_tensor,
#             M, N, K,
#             A_int8.stride(0), A_int8.stride(1),
#             B_transposed.stride(0), B_transposed.stride(1),
#             C_float.stride(0), C_float.stride(1),
#             BLOCK_M, BLOCK_N, BLOCK_K
#         )
#         return C_float
#     except Exception as e:
#         # Fallback to PyTorch
#         print(f"Triton kernel failed: {e}. Using PyTorch fallback.")
#         return torch.matmul(A_int8.to(torch.float32), B_int8.transpose(0, 1).to(torch.float32)) * (A_scale_val * B_scale_val)