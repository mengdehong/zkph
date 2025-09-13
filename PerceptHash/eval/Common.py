import os
import sys
import torch
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset, DataLoader
import math
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt

# --- Setup and Path Imports ---
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..')) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import necessary modules ---
from model.ModelFactory import create_hashing_model
from utils.tools import get_pihd_transforms

# ===================================================================
# Image Dataset
# ===================================================================
class ImageDataset(Dataset):
    """Simple image dataset for evaluation."""
    def __init__(self, file_list, root_dir, transform):
        self.file_list = file_list
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        fname = self.file_list[idx]
        img_path = os.path.join(self.root_dir, fname)
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img)
        tensor = self.transform(image=arr)['image']
        return fname, tensor

# ===================================================================
# Hash Generation
# ===================================================================
def generate_hashes(model, config):
    """
    Generates hashes for a model with caching and OOM fallback.
    """
    cache_path = config['cache_path']
    if os.path.exists(cache_path):
        logger.info(f"Loading hashes from cache: {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        return cache['binary_hashes'].item(), cache['real_hashes'].item()

    logger.info("Generating hashes for test set...")
    binary_hashes, real_hashes = {}, {}
    file_list = sorted(os.listdir(config['dataset_path']))
    _, val_transform = get_pihd_transforms({'crop_size': config['crop_size']})
    current_bs = config['batch_size']
    min_bs = 1

    def build_loader(bs):
        ds = ImageDataset(file_list, config['dataset_path'], val_transform)
        return DataLoader(ds, batch_size=bs, shuffle=False,
                          num_workers=config.get('num_workers', 4), pin_memory=True,
                          drop_last=False)

    dataloader = build_loader(current_bs)
    model.eval()

    while True:
        try:
            pbar = tqdm(dataloader, desc=f"Inference (bs={current_bs})")
            with torch.no_grad():
                for names, batch in pbar:
                    batch = batch.to(config['device'], non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=config.get('amp', True)):
                        batch_real = model(batch)
                    batch_real_cpu = batch_real.float().cpu().numpy()
                    batch_bin_cpu = np.sign(batch_real_cpu)
                    for i, fname in enumerate(names):
                        real_hashes[fname] = batch_real_cpu[i]
                        binary_hashes[fname] = batch_bin_cpu[i]
            break
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and current_bs > min_bs:
                torch.cuda.empty_cache()
                current_bs = max(min_bs, current_bs // 2)
                logger.warning(f"[OOM] Reducing batch_size -> {current_bs}")
                dataloader = build_loader(current_bs)
            else:
                raise

    logger.info(f"Saving hashes to cache: {cache_path}")
    np.savez_compressed(cache_path, binary_hashes=binary_hashes, real_hashes=real_hashes)
    return binary_hashes, real_hashes

# ===================================================================
# Distance Functions
# ===================================================================
def hamming_distance(h1, h2): 
    return 0.5 * (h1.shape[0] - np.dot(h1, h2))

def mse_distance(h1, h2): 
    return np.mean((h1 - h2)**2)

def imagehash_distance(h1, h2): return h1 - h2

# ===================================================================
# ROC Metrics
# ===================================================================
def compute_roc_metrics(binary_hashes, real_hashes, config, tpr_targets=None):
    """Computes ROC metrics (AUC, FPR, TPR) for Hamming and MSE distances."""
    tpr_targets = tpr_targets or []
    file_list = sorted(os.listdir(config['dataset_path']))
    num_groups = len(file_list) // config['group_size']
    roc_results = {}
    
    for dist_name, hashes_dict, dist_func in [
        ("Hamming", binary_hashes, hamming_distance),
        ("MSE", real_hashes, mse_distance)
    ]:
        labels, distances = [], []
        for i in range(num_groups):
            start_idx = i * config['group_size']
            h_original = hashes_dict[file_list[start_idx]]
            for j in range(1, config['group_size']):
                dist = dist_func(h_original, hashes_dict[file_list[start_idx + j]])
                distances.append(dist)
                labels.append(1 if j < 49 else 0)
        
        scores = -np.array(distances)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        result = {'auc': roc_auc, 'fpr': fpr, 'tpr': tpr}
        if tpr_targets:
            result['fpr_at_tpr'] = {t: fpr[np.argmax(tpr >= t)] for t in tpr_targets}
        roc_results[dist_name] = result
        
    return roc_results

# ===================================================================
# Model Configuration & Evaluation
# ===================================================================
def get_model_config_from_name(exp_name):
    """Infers model config from experiment name."""
    config = {
        'bit_list': [64], 'crop_size': 224, 'device': 'cuda:0',
        'groupmamba_tiny_params': {
            "num_classes": -1, "embed_dims": [64, 128, 348, 448], 
            "depths": [3, 4, 9, 3], "mlp_ratios": [8, 8, 4, 4], 
            "stem_hidden_dim": 32, "k_size": [3, 3, 5, 5]
        }
    }
    
    backbone_map = {
        'ResNet': 'resnet50', 'ViT': 'vit', 'MambaOut': 'mambaout_tiny',
        'GroupMamba': 'groupmamba', 'ConvNeXtV2': 'convnextv2', 'SwinTiny': 'swin_tiny'
    }
    
    config['backbone'] = next((v for k, v in backbone_map.items() if k in exp_name), None)
    config['strategy'] = 'enhanced' if 'Enhanced' in exp_name else 'traditional'
    if not config['backbone']:
        raise ValueError(f"Cannot infer model type from: {exp_name}")
    
    config['bit_list'] = [32] if 'b32' in exp_name else [64]
    return config

def run_full_evaluation(binary_hashes, real_hashes, config, hash_type='deep'):
    """
    Runs a full evaluation suite including robustness, discriminability, and performance.
    """
    model_label = config['model_label']
    output_dir = config['output_dir']
    
    file_list = sorted(os.listdir(config['dataset_path']))
    num_groups = len(file_list) // config['group_size']
    
    # --- Task 1: Robustness Analysis ---
    print(f"\n--- [{model_label}] Running Robustness Analysis ---")
    manipulation_types = ["Flip_Horizontal", "Flip_Vertical", "Rotate", "Crop", "Scale", "Resize", "Clip", "Aspect_ratio", "Blur", "Brightness", "Contrast", "Color_jitter", "Encoding", "Grayscale", "Opacity", "Pixelization", "Saturation", "Sharpen", "Shuffle_pixel", "Random_noise", "Masking", "Padding", "Detail_filter", "Edge_filter", "Smooth_filter", "Kernel_filter", "Rank_filter", "Max_filter", "Min_filter", "Median_filter", "Mode_filter", "Background", "Overlay_stripe", "Overlay_text", "Meme"]
    dist_func = imagehash_distance if hash_type == 'traditional' else hamming_distance
    primary_hashes = binary_hashes
    
    robustness_results = []
    for manip_idx, manip_name in enumerate(manipulation_types):
        distances = [dist_func(primary_hashes[file_list[i*config['group_size']]], 
                              primary_hashes[file_list[i*config['group_size'] + manip_idx + 1]]) 
                     for i in range(num_groups)]
        robustness_results.append({'Manipulation': manip_name, 'Mean': np.mean(distances), 'Max': np.max(distances), 'Min': np.min(distances)})
    
    combined_distances = [dist_func(primary_hashes[file_list[i*config['group_size']]], 
                                   primary_hashes[file_list[i*config['group_size'] + j]]) 
                          for i in range(num_groups) for j in range(36, 49)]
    robustness_results.append({'Manipulation': 'Combination', 'Mean': np.mean(combined_distances), 'Max': np.max(combined_distances), 'Min': np.min(combined_distances)})

    df_robust = pd.DataFrame(robustness_results)
    df_robust.to_csv(os.path.join(output_dir, 'robustness_report.csv'), index=False, float_format='%.4f')
    print(f"Robustness report for {model_label} saved.")

    # --- Task 2 & 3: Discriminability & Performance ---
    roc_data = {}
    dist_configs = [("ImageHash", binary_hashes, imagehash_distance)] if hash_type == 'traditional' else [("Hamming", binary_hashes, hamming_distance), ("MSE", real_hashes, mse_distance)]
    
    for dist_name, hashes_dict, dist_func in dist_configs:
        print(f"\n--- [{model_label}] Analyzing Discrimination & Performance ({dist_name}) ---")
        
        labels, distances, positive_dists, negative_dists = [], [], [], []
        
        for i in tqdm(range(num_groups), desc=f"Building pairs ({dist_name})"):
            start_idx = i * config['group_size']
            h_original = hashes_dict[file_list[start_idx]]
            for j in range(1, config['group_size']):
                dist = dist_func(h_original, hashes_dict[file_list[start_idx + j]])
                is_positive = 1 if j < 49 else 0
                distances.append(dist); labels.append(is_positive)
                if is_positive: positive_dists.append(dist)

        original_hashes = [hashes_dict[file_list[i*config['group_size']]] for i in range(num_groups)]
        for i in range(num_groups):
            for j in range(i + 1, num_groups):
                negative_dists.append(dist_func(original_hashes[i], original_hashes[j]))
        
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(positive_dists, bins=50, alpha=0.7, label=f'Identical Pairs', color='blue', density=True)
        plt.hist(negative_dists, bins=50, alpha=0.7, label=f'Distinct Pairs', color='red', density=True)
        plt.xlabel(f'{dist_name} Distance'); plt.ylabel('Probability Density')
        plt.title(f'Distance Distribution for {model_label} ({dist_name})'); plt.legend()
        plt.savefig(os.path.join(output_dir, f'discrimination_plot_{dist_name.lower()}.png'), dpi=300)
        plt.close()
        
        scores = -np.array(distances)
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        roc_data[dist_name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

    return df_robust, roc_data