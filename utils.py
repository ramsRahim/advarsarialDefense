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
import torch.nn.functional as F


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def get_entropy(loss, clip_weights):
    """Calculate the entropy normalized by the maximum possible entropy."""
    import math
    
    try:
        # Determine number of classes - multiple options for robustness
        if loss.dim() > 0 and loss.size(1) > 1:
            num_classes = loss.size(1)
        elif isinstance(clip_weights, list):
            num_classes = len(clip_weights)
        elif hasattr(clip_weights, 'size') and clip_weights.dim() > 1:
            num_classes = clip_weights.size(1)
        else:
            # Fallback
            num_classes = 1000  # Default for ImageNet
        
        max_entropy = math.log2(num_classes)
        
        # Check for NaNs and infinities
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            # Sanitize the tensor - replace NaNs and Infs with small values
            sanitized_loss = torch.where(
                torch.isnan(loss) | torch.isinf(loss),
                torch.full_like(loss, 1e-6),
                loss
            )
        else:
            sanitized_loss = loss
            
        # Calculate the entropy: -sum(p * log2(p))
        eps = 1e-10  # Small constant to avoid log(0)
        entropy = -torch.sum(sanitized_loss * torch.log2(sanitized_loss + eps), dim=1)
        
        # Return the mean entropy normalized by maximum entropy
        result = float(entropy.mean() / max_entropy)
        
        # Safety check for NaN or infinite result
        if math.isnan(result) or math.isinf(result):
            return 0.5  # Default middle value
            
        return result
    except Exception as e:
        print(f"Error calculating entropy: {e}")
        return 0.5  # Default middle value


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


# def cls_acc(output, target, topk=1):
#     pred = output.topk(topk, 1, True, True)[1].t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#     acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
#     acc = 100 * acc / target.shape[0]
#     return acc

def cls_acc(output, target, topk=1):
    # Get batch sizes from output and target.
    output_batch = output.size(0)
    target = target.view(-1)
    target_batch = target.size(0)
    
    # If output has one element but target has multiple, replicate output.
    if output_batch == 1 and target_batch > 1:
        output = output.expand(target_batch, *output.shape[1:])
        output_batch = target_batch
    # If target has one element but output has multiple, replicate target.
    elif target_batch == 1 and output_batch > 1:
        target = target.expand(output_batch)
        target_batch = output_batch
    # If both have multiple but their sizes don't match, raise an error.
    elif output_batch != target_batch:
        raise ValueError(f"Mismatch: output batch size {output_batch} does not match target batch size {target_batch}")
    
    batch_size = output_batch

    # Get the top-k predictions along the class dimension.
    _, pred = output.topk(topk, dim=1, largest=True, sorted=True)
    # Transpose pred so its shape becomes [topk, batch_size]
    pred = pred.t()
    
    # Expand target to [1, batch_size] to compare with pred.
    target_expanded = target.unsqueeze(0).expand_as(pred)
    correct = pred.eq(target_expanded)
    
    correct_total = correct.float().sum().item()
    acc = 100 * correct_total / batch_size
    return acc


def clip_classifier(classnames, template, clip_model):
    """Create a classifier using CLIP's text encoder."""
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            
            # Get text embeddings (these should be normalized tensors)
            class_embeddings = clip_model.encode_text(texts)
            
            # If the output is a tuple (features, scale, zero_point), take the features
            if isinstance(class_embeddings, tuple):
                class_embeddings = class_embeddings[0]
                
            # Take the mean and normalize
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            
            clip_weights.append(class_embedding)
        
        return clip_weights

def get_clip_logits(images, clip_model, clip_weights):
    """Get CLIP logits with quantized features."""
    images = images.cuda()
    
    try:
        # Get image features from the model
        image_features = clip_model.encode_image(images)
        
        # Determine if the features are quantized (returned as a tuple)
        if isinstance(image_features, tuple):
            # Unpack quantized tensor, scale, and zero_point
            if len(image_features) >= 3:
                img_features_int8, img_scale, img_zero_point = image_features
                
                # Check if we need to dequantize
                if hasattr(img_features_int8, 'dequantize'):
                    try:
                        image_features_fp = img_features_int8.dequantize()
                    except RuntimeError as e:
                        print(f"Dequantization error: {e}, falling back to raw features")
                        image_features_fp = img_features_int8
                else:
                    # Already floating point
                    image_features_fp = img_features_int8
            else:
                # Not a standard quantized return, use first element
                image_features_fp = image_features[0]
                img_features_int8 = image_features[0]
                img_scale = 1.0
                img_zero_point = 0
        else:
            # Not quantized, use the features directly
            image_features_fp = image_features
            # Create placeholders for the quantized version
            img_features_int8 = image_features
            img_scale = torch.tensor(1.0, device=image_features.device)
            img_zero_point = torch.tensor(0, device=image_features.device)
        
        # Normalize the features
        norm = image_features_fp.norm(dim=-1, keepdim=True)
        # Handle potential zeros in norm
        norm = torch.where(norm > 1e-6, norm, torch.ones_like(norm) * 1e-6)
        image_features_norm = image_features_fp / norm
        
        # Convert clip_weights list to tensor if needed
        if isinstance(clip_weights, list):
            class_weights = torch.stack(clip_weights, dim=1).to(image_features_fp.device)
        else:
            class_weights = clip_weights
            
        # Ensure both tensors are of the same type before matrix multiplication
        if image_features_norm.dtype != class_weights.dtype:
            class_weights = class_weights.to(image_features_norm.dtype)
        
        # Compute logits with the class weights
        logits = 100. * image_features_norm @ class_weights
        
        # Get softmax for loss
        loss = F.softmax(logits, dim=1)
        
        # Get probability map
        prob_map = loss.clone()
        
        # Get prediction
        pred = torch.argmax(logits, dim=1)
        
        # Return appropriate values
        return (img_features_int8, img_scale, img_zero_point), logits, loss, prob_map, pred
    
    except Exception as e:
        # If anything fails, log the error and provide a fallback response
        print(f"Error in get_clip_logits: {e}")
        # Create placeholder outputs
        device = images.device
        placeholder = torch.zeros(1, 512, device=device)
        placeholder_logits = torch.zeros(1, len(clip_weights) if isinstance(clip_weights, list) else clip_weights.size(1), device=device) 
        placeholder_loss = F.softmax(placeholder_logits, dim=1)
        placeholder_pred = torch.zeros(1, dtype=torch.long, device=device)
        
        return placeholder, placeholder_logits, placeholder_loss, placeholder_loss.clone(), placeholder_pred


def compute_quantized_similarity(img_feats, img_scale, img_zp, text_feats, text_scales, text_zero_points):
    """Compute similarity between quantized features."""
    # For proper INT8 matmul, we need to either:
    # 1. Dequantize both tensors, compute matmul, then quantize result
    # 2. Use a specialized INT8 matmul kernel (e.g., from PyTorch 2.0 or Triton)
    
    # For simplicity, we'll use option 1 first
    img_feats_fp = img_feats.dequantize()
    img_feats_norm = img_feats_fp / img_feats_fp.norm(dim=-1, keepdim=True)
    
    all_logits = []
    for i, text_feat in enumerate(text_feats):
        text_feat_fp = text_feat.dequantize()
        text_feat_norm = text_feat_fp / text_feat_fp.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        similarity = img_feats_norm @ text_feat_norm.transpose(0, 1)
        all_logits.append(similarity)
    
    # Stack all logits
    logits = torch.stack(all_logits, dim=1).squeeze()
    return logits

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