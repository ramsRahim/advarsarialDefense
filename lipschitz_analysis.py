# lipschitz_analysis.py
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from utils import build_test_data_loader, clip_classifier, estimate_lipschitz_for_clip

def analyze_lipschitz_distribution(dataset, model_name="RN50", attack_type=None, epsilon=0.03, sample_size=1000):
    """
    Analyze the distribution of Lipschitz constants in a dataset.
    
    Args:
        dataset: Dataset name (e.g., 'I' for ImageNet)
        model_name: CLIP model name
        attack_type: Type of attack to apply (None, 'fgsm', 'pgd')
        epsilon: Attack strength
        sample_size: Number of samples to analyze
    """
    # Load CLIP model
    model, preprocess = clip.load(model_name)
    model.eval()
    device = next(model.parameters()).device
    
    # Load dataset
    data_loader, classnames, template = build_test_data_loader(dataset, './dataset/', preprocess)
    
    # Get classifier weights
    text_features = clip_classifier(classnames, template, model)
    
    # Initialize lists to store values
    clean_lip_values = []
    adv_lip_values = []
    
    # Optional attack setup
    if attack_type:
        from clip_attack import fgsm_attack, pgd_attack
    
    # Collect Lipschitz values
    print(f"Calculating Lipschitz constants for {sample_size} samples...")
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(data_loader)):
            if i >= sample_size:
                break
                
            # Move to device
            images = images.to(device)
            target = target.to(device)
            
            # Get clean image features
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate clean Lipschitz constant
            clean_lip = estimate_lipschitz_for_clip(image_features, text_features)
            clean_lip_values.append(clean_lip)
            
            # If attack is specified, also calculate adversarial Lipschitz
            if attack_type:
                with torch.enable_grad():
                    if attack_type == 'fgsm':
                        adv_images = fgsm_attack(model, images, target, epsilon, text_features)
                    elif attack_type == 'pgd':
                        adv_images = pgd_attack(model, images, target, epsilon, epsilon/5, 10, text_features)
                
                # Get adversarial features
                adv_features = model.encode_image(adv_images)
                adv_features = adv_features / adv_features.norm(dim=-1, keepdim=True)
                
                # Calculate adversarial Lipschitz constant
                adv_lip = estimate_lipschitz_for_clip(adv_features, text_features)
                adv_lip_values.append(adv_lip)
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Plot clean distribution
    plt.subplot(2, 1, 1)
    sns.histplot(clean_lip_values, bins=50, kde=True, color='blue')
    plt.title('Distribution of Lipschitz Constants (Clean Images)')
    plt.xlabel('Lipschitz Constant')
    plt.ylabel('Frequency')
    
    # Add percentile lines
    percentiles = [50, 75, 90, 95, 99]
    colors = ['green', 'orange', 'red', 'purple', 'black']
    for p, color in zip(percentiles, colors):
        val = np.percentile(clean_lip_values, p)
        plt.axvline(x=val, color=color, linestyle='--', 
                    label=f'{p}th percentile: {val:.3f}')
    
    plt.legend()
    
    # If we have adversarial data, create comparison plot
    if attack_type:
        plt.subplot(2, 1, 2)
        
        # Overlay histogram
        sns.histplot(clean_lip_values, bins=50, alpha=0.5, color='blue', label='Clean')
        sns.histplot(adv_lip_values, bins=50, alpha=0.5, color='red', label='Adversarial')
        
        # Calculate optimal threshold using ROC analysis
        all_values = clean_lip_values + adv_lip_values
        all_labels = [0] * len(clean_lip_values) + [1] * len(adv_lip_values)
        
        # Sort by lipschitz value
        sorted_indices = np.argsort(all_values)
        sorted_values = [all_values[i] for i in sorted_indices]
        sorted_labels = [all_labels[i] for i in sorted_indices]
        
        # Calculate TPR and FPR for different thresholds
        best_threshold = 0
        best_f1 = 0
        
        for threshold in sorted_values:
            predictions = [1 if v >= threshold else 0 for v in all_values]
            
            # Calculate metrics
            tp = sum(1 for i in range(len(predictions)) if predictions[i] == 1 and all_labels[i] == 1)
            fp = sum(1 for i in range(len(predictions)) if predictions[i] == 1 and all_labels[i] == 0)
            fn = sum(1 for i in range(len(predictions)) if predictions[i] == 0 and all_labels[i] == 1)
            
            # Avoid division by zero
            if tp + fp == 0 or tp + fn == 0:
                continue
                
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        # Add suggested threshold line
        plt.axvline(x=best_threshold, color='black', linestyle='-', 
                   label=f'Suggested threshold: {best_threshold:.3f}')
        
        plt.title(f'Comparison of Lipschitz Constants ({attack_type.upper()} Attack, ε={epsilon})')
        plt.xlabel('Lipschitz Constant')
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'lipschitz_distribution_{dataset}_{attack_type if attack_type else "clean"}.png')
    plt.show()
    
    # Print recommendations
    print("\n--- Lipschitz Constant Analysis ---")
    print(f"Mean (Clean): {np.mean(clean_lip_values):.4f}")
    print(f"Std Dev (Clean): {np.std(clean_lip_values):.4f}")
    
    for p in [50, 75, 90, 95, 99]:
        print(f"{p}th Percentile (Clean): {np.percentile(clean_lip_values, p):.4f}")
    
    if attack_type:
        print(f"\nMean (Adversarial): {np.mean(adv_lip_values):.4f}")
        print(f"Std Dev (Adversarial): {np.std(adv_lip_values):.4f}")
        print(f"\nRecommended threshold: {best_threshold:.4f}")
        
        # Calculate detection performance
        predictions = [1 if v >= best_threshold else 0 for v in all_values]
        tp = sum(1 for i in range(len(predictions)) if predictions[i] == 1 and all_labels[i] == 1)
        fp = sum(1 for i in range(len(predictions)) if predictions[i] == 1 and all_labels[i] == 0)
        tn = sum(1 for i in range(len(predictions)) if predictions[i] == 0 and all_labels[i] == 0)
        fn = sum(1 for i in range(len(predictions)) if predictions[i] == 0 and all_labels[i] == 1)
        
        print(f"\nDetection performance at threshold {best_threshold:.4f}:")
        print(f"True Positives: {tp} ({tp/len(adv_lip_values)*100:.1f}% of adversarial)")
        print(f"False Positives: {fp} ({fp/len(clean_lip_values)*100:.1f}% of clean)")
        print(f"True Negatives: {tn} ({tn/len(clean_lip_values)*100:.1f}% of clean)")
        print(f"False Negatives: {fn} ({fn/len(adv_lip_values)*100:.1f}% of adversarial)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Lipschitz constant distribution")
    parser.add_argument("--dataset", type=str, default="I", help="Dataset name (e.g., I for ImageNet)")
    parser.add_argument("--model", type=str, default="RN50", help="CLIP model name")
    parser.add_argument("--attack", type=str, default=None, choices=[None, "fgsm", "pgd"], 
                        help="Attack type to analyze")
    parser.add_argument("--epsilon", type=float, default=0.03, help="Attack strength")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to analyze")
    
    args = parser.parse_args()
    
    analyze_lipschitz_distribution(args.dataset, args.model, args.attack, args.epsilon, args.samples)