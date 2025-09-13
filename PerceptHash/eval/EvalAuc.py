import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import imagehash
from eval.Common import *

# --- Global Settings & Path Imports ---
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

def _imagehash_to_vec32(img: Image.Image, hash_func) -> np.ndarray:
    """Computes a 32-bit hash vector from an image, mapping to {-1, +1}."""
    h = hash_func(img)
    bits = np.array(h.hash, dtype=np.uint8).flatten()[:32]
    return bits.astype(np.int8) * 2 - 1

def _imagehash_to_vec64(img: Image.Image, hash_func) -> np.ndarray:
    """Computes a 64-bit hash vector from an image, mapping to {-1, +1}."""
    h = hash_func(img)
    bits = np.array(h.hash, dtype=np.uint8).flatten()
    if bits.size > 64: bits = bits[:64]
    if bits.size < 64: bits = np.pad(bits, (0, 64 - bits.size), mode='constant', constant_values=0)
    return bits.astype(np.int8) * 2 - 1

def _compute_trad_hash_for_file(args):
    """Worker function to compute a traditional hash for a single image file."""
    fname, dataset_path, hash_func = args
    try:
        img_path = os.path.join(dataset_path, fname)
        img = Image.open(img_path).convert('RGB')
        vec = _imagehash_to_vec32(img, hash_func)
        return fname, vec
    except Exception as e:
        logger.warning(f"Traditional hash failed for {fname}: {e}")
        return fname, None

def generate_traditional_hashes_32(config, hash_type='phash', max_workers=8):
    """Generates 32-bit traditional hashes (phash/whash) using a thread pool."""
    file_list = sorted(os.listdir(config['dataset_path']))
    hash_func = partial(imagehash.phash if hash_type == 'phash' else imagehash.whash, hash_size=8)
    hashes = {}
    thread_args = [(fname, config['dataset_path'], hash_func) for fname in file_list]
    with ThreadPoolExecutor(max_workers=max_workers) as ex, tqdm(total=len(file_list), desc=f"Computing {hash_type.upper()}-32") as pbar:
        futures = {ex.submit(_compute_trad_hash_for_file, a): a[0] for a in thread_args}
        for fut in as_completed(futures):
            fname, vec = fut.result()
            if vec is not None: hashes[fname] = vec
            pbar.update(1)
    logger.info(f"Computed {len(hashes)}/{len(file_list)} {hash_type.upper()} 32-bit hashes")
    return hashes

def generate_traditional_hashes_64(config, hash_type='phash', max_workers=8):
    """Generates 64-bit traditional hashes (phash/whash) using a thread pool."""
    file_list = sorted(os.listdir(config['dataset_path']))
    hash_func = partial(imagehash.phash if hash_type == 'phash' else imagehash.whash, hash_size=8)
    
    def compute64(args):
        fname, dataset_path, hash_func = args
        try:
            img_path = os.path.join(dataset_path, fname)
            img = Image.open(img_path).convert('RGB')
            vec = _imagehash_to_vec64(img, hash_func)
            return fname, vec
        except Exception as e:
            logger.warning(f"Traditional 64-bit hash failed for {fname}: {e}")
            return fname, None

    hashes = {}
    thread_args = [(fname, config['dataset_path'], hash_func) for fname in file_list]
    with ThreadPoolExecutor(max_workers=max_workers) as ex, tqdm(total=len(file_list), desc=f"Computing {hash_type.upper()}-64") as pbar:
        futures = {ex.submit(compute64, a): a[0] for a in thread_args}
        for fut in as_completed(futures):
            fname, vec = fut.result()
            if vec is not None: hashes[fname] = vec
            pbar.update(1)
    logger.info(f"Computed {len(hashes)}/{len(file_list)} {hash_type.upper()} 64-bit hashes")
    return hashes

def compute_roc_metrics_traditional(hashes_dict, config, dist_name='ImageHash32', tpr_targets=[0.95]):
    """Computes ROC metrics using only Hamming distance."""
    file_list = sorted(os.listdir(config['dataset_path']))
    num_groups = len(file_list) // config['group_size']
    labels, distances = [], []
    for i in tqdm(range(num_groups), desc=f"Building pairs ({dist_name})"):
        start_idx = i * config['group_size']
        h_original = hashes_dict[file_list[start_idx]]
        for j in range(1, config['group_size']):
            dist = hamming_distance(h_original, hashes_dict[file_list[start_idx + j]])
            distances.append(dist)
            labels.append(1 if j < 49 else 0)
    scores = -np.array(distances)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    fpr_at_tpr = {target: fpr[np.argmin(np.abs(tpr - target))] for target in tpr_targets}
    logger.info(f"{dist_name} AUC: {roc_auc:.4f}")
    return {'auc': roc_auc, 'fpr': fpr, 'tpr': tpr, 'fpr_at_tpr': fpr_at_tpr}

# ==============================================================
# TPR@95% Hamming Distance Threshold Calculation
# ==============================================================
def compute_tpr95_hamming_threshold(binary_hashes, config, target_tpr=0.95):
    """Calculates the Hamming distance threshold for a target TPR (default 0.95)."""
    file_list = sorted(os.listdir(config['dataset_path']))
    num_groups = len(file_list) // config['group_size']
    labels, distances = [], []
    for i in tqdm(range(num_groups), desc="Building pairs (Hamming for Thr)"):
        start_idx = i * config['group_size']
        h_original = binary_hashes[file_list[start_idx]]
        for j in range(1, config['group_size']):
            dist = hamming_distance(h_original, binary_hashes[file_list[start_idx + j]])
            distances.append(dist)
            labels.append(1 if j < 49 else 0)
    scores = -np.array(distances)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    indices = np.where(tpr >= target_tpr)[0]
    if not indices.any():
        logger.warning(f"Cannot achieve TPR >= {target_tpr}")
        return None, None, None
    idx = indices[0]
    hamming_threshold = -thresholds[idx]
    actual_tpr = tpr[idx]
    actual_fpr = fpr[idx]
    logger.info(f"TPR@{target_tpr*100:.1f}%: {actual_tpr:.4f}, FPR: {actual_fpr:.6f}, Hamming Thr: {hamming_threshold:.4f}")
    return hamming_threshold, actual_tpr, actual_fpr

def get_target_models_for_threshold():
    """Returns a list of enhanced models for TPR@95% evaluation."""
    models = [
        ("ResNet50_Enhanced", "ResNet-50+", "save/TrainAll/ResNet+/ResNet50_Enhanced_best.pth"),
        ("MambaOut_Enhanced", "MambaOut+", "save/TrainAll/MambaOut+/best_model.pth"),
        ("ViT_Enhanced", "ViT-S+", "save/TrainAll/ViT+/best_model.pth"),
        ("GroupMamba_Enhanced", "GroupMamba+", "save/TrainAll/GroupMamba+/GroupMamba_Enhanced_best.pth"),
        ("ConvNeXtV2_Enhanced", "ConvNeXtV2+", "save/TrainAll/ConvNeXtV2+/ConvNeXtV2_Enhanced_best.pth"),
        ("SwinTiny_Enhanced", "SwinTiny+", "save/TrainAll/SwinTiny+/SwinTiny_Enhanced_best.pth"),
    ]
    target_models = []
    for exp_name, model_short, rel_path in models:
        weight_path = os.path.join(project_root, rel_path)
        if os.path.exists(weight_path):
            target_models.append({'experiment_name': exp_name, 'model_short': model_short, 'weight_path': weight_path})
            logger.info(f"[Thr] Found model: {model_short} -> {weight_path}")
        else:
            logger.warning(f"[Thr] Weight not found: {weight_path}")
    return target_models

def evaluate_single_model_threshold(model_info, args=None):
    """Evaluates the TPR@95% Hamming distance threshold for a single model."""
    exp_name, weight_path = model_info['experiment_name'], model_info['weight_path']
    logger.info(f"\n{'='*50}\nEvaluating TPR@95% Threshold: {exp_name}\nWeights: {weight_path}\n{'='*50}")
    
    config = get_model_config_from_name(exp_name)
    config.update({
        'dataset_path': os.path.join(project_root, 'datasets/PIHD/test/test_class'),
        'group_size': 97, 'batch_size': 64, 'num_workers': 8, 'amp': True,
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    })
    
    if args:
        config['batch_size'] = args.batch_size or config['batch_size']
        config['num_workers'] = args.num_workers or config['num_workers']
        config['device'] = torch.device(args.device) if args.device else config['device']
        if args.no_amp or config['device'].type == 'cpu': config['amp'] = False

    cache_dir = os.path.join(project_root, 'save', 'eval', 'eval_tpr95', 'eval_cache', exp_name.replace(' ', '_'))
    os.makedirs(cache_dir, exist_ok=True)
    config['cache_path'] = os.path.join(cache_dir, 'hash_cache.npz')
    
    model = create_hashing_model(config['backbone'], config['strategy'], config).to('cpu')
    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    logger.info("Weights loaded.")
    
    if config['device'].type == 'cuda':
        try: model = model.to(config['device'])
        except RuntimeError as e: logger.warning(f"CUDA fallback to CPU: {e}"); config['device'] = torch.device('cpu'); config['amp'] = False; model = model.to('cpu')

    binary_hashes, _ = generate_hashes(model, config)
    hamming_thr, tpr95, fpr95 = compute_tpr95_hamming_threshold(binary_hashes, config, target_tpr=0.95)
    
    if hamming_thr is None:
        return {'experiment_name': exp_name, 'model_short': model_info['model_short'], 'hamming_threshold_tpr95': 'N/A', 'actual_tpr': 'N/A', 'actual_fpr': 'N/A'}
    return {'experiment_name': exp_name, 'model_short': model_info['model_short'], 'hamming_threshold_tpr95': hamming_thr, 'actual_tpr': tpr95, 'actual_fpr': fpr95}

def run_threshold_evaluation(args=None):
    """Runs TPR@95% threshold calculation for all target enhanced models and saves to CSV."""
    logger.info("Starting TPR@95% threshold evaluation...")
    target_models = get_target_models_for_threshold()
    if not target_models:
        logger.error("No target models found for threshold evaluation.")
        return
    results = [evaluate_single_model_threshold(m, args=args) for m in target_models]
    df = pd.DataFrame(results)
    out_dir = os.path.join(project_root, 'save', 'eval', 'eval_tpr95')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'tpr95_hamming_thresholds.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')
    logger.info(f"Threshold results saved: {csv_path}")
    print("\nTPR@95% Hamming Distance Thresholds:")
    print(df.to_string(index=False))

def find_all_trained_models():
    """Returns a hardcoded list of 64-bit and 32-bit models with their weight paths."""
    hardcoded = [
        ("GroupMamba_Baseline", "GroupMamba", "save/TrainAll/GroupMamba/GroupMamba_Baseline_best.pth"),
        ("GroupMamba_Baseline_b32", "GroupMamba", "save/TrainAll/GroupMamba/GroupMamba_Baseline-b32_best.pth"),
        ("GroupMamba_Enhanced", "GroupMamba+", "save/TrainAll/GroupMamba+/GroupMamba_Enhanced_best.pth"),
        ("GroupMamba_Enhanced_b32", "GroupMamba+", "save/TrainAll/GroupMamba+/GroupMamba_Enhanced-b32_best.pth"),
        ("MambaOut_Baseline", "MambaOut", "save/TrainAll/MambaOut/best_model.pth"),
        ("MambaOut_Baseline_b32", "MambaOut", "save/TrainAll/MambaOut/MambaOut_Baseline-b32_best.pth"),
        ("MambaOut_Enhanced", "MambaOut+", "save/TrainAll/MambaOut+/best_model.pth"),
        ("MambaOut_Enhanced_b32", "MambaOut+", "save/TrainAll/MambaOut+/MambaOut_Enhanced-b32_best.pth"),
        ("ResNet50_BsaeLine", "ResNet", "save/TrainAll/ResNet/ResNet50_BsaeLine_best.pth"),
        ("ResNet50_BsaeLine_b32", "ResNet", "save/TrainAll/ResNet/ResNet50_BsaeLine-b32_best.pth"),
        ("ResNet50_Enhanced", "ResNet+", "save/TrainAll/ResNet+/ResNet50_Enhanced_best.pth"),
        ("ResNet50_Enhanced_b32", "ResNet+", "save/TrainAll/ResNet+/ResNet50_Enhanced-b32_best.pth"),
        ("ViT_Baseline", "ViT", "save/TrainAll/ViT/best_model.pth"),
        ("ViT_Baseline_b32", "ViT", "save/TrainAll/ViT/ViT_BsaeLine-b32_best.pth"),
        ("ViT_Enhanced", "ViT+", "save/TrainAll/ViT+/best_model.pth"),
        ("ViT_Enhanced_b32", "ViT+", "save/TrainAll/ViT+/ViT_Enhanced-b32_best.pth"),
        ("ConvNeXtV2_Baseline", "ConvNeXtV2", "save/TrainAll/ConvNeXtV2/ConvNeXtV2_Baseline_best.pth"),
        ("ConvNeXtV2_Baseline_b32", "ConvNeXtV2", "save/TrainAll/ConvNeXtV2/ConvNeXtV2_Baseline-b32_best.pth"),
        ("ConvNeXtV2_Enhanced", "ConvNeXtV2+", "save/TrainAll/ConvNeXtV2+/ConvNeXtV2_Enhanced_best.pth"),
        ("ConvNeXtV2_Enhanced_b32", "ConvNeXtV2+", "save/TrainAll/ConvNeXtV2+/ConvNeXtV2_Enhanced-b32_best.pth"),
        ("SwinTiny_Baseline", "SwinTiny", "save/TrainAll/SwinTiny/SwinTiny_Baseline_best.pth"),
        ("SwinTiny_Baseline_b32", "SwinTiny", "save/TrainAll/SwinTiny/SwinTiny_Baseline-b32_best.pth"),
        ("SwinTiny_Enhanced", "SwinTiny+", "save/TrainAll/SwinTiny+/SwinTiny_Enhanced_best.pth"),
        ("SwinTiny_Enhanced_b32", "SwinTiny+", "save/TrainAll/SwinTiny+/SwinTiny_Enhanced-b32_best.pth"),
    ]
    bit64_models, bit32_models = [], []
    for exp_name, model_short, rel_path in hardcoded:
        weight_path = os.path.join(project_root, rel_path)
        if os.path.exists(weight_path):
            model_info = {'experiment_name': exp_name, 'model_short': model_short, 'weight_path': weight_path, 'best_metric': {}}
            (bit32_models if 'b32' in exp_name else bit64_models).append(model_info)
            logger.info(f"Using hardcoded model: {exp_name} -> {weight_path}")
        else:
            logger.warning(f"Hardcoded weight not found, skipping: {weight_path}")
    logger.info(f"Total 64-bit models: {len(bit64_models)}, Total 32-bit models: {len(bit32_models)}")
    return bit64_models, bit32_models

def evaluate_single_model(model_info, args=None):
    """Evaluates a single model's AUC and FPR@95%."""
    exp_name, weight_path = model_info['experiment_name'], model_info['weight_path']
    logger.info(f"\n{'='*50}\nEvaluating: {exp_name}\nWeights: {weight_path}\n{'='*50}")
    
    config = get_model_config_from_name(exp_name)
    config.update({'dataset_path': os.path.join(project_root, 'datasets/PIHD/test/test_class'), 'group_size': 97, 'batch_size': 512, 'num_workers': 12, 'amp': True})
    if args:
        config['batch_size'] = args.batch_size or config['batch_size']
        config['num_workers'] = args.num_workers or config['num_workers']
        config['device'] = torch.device(args.device) if args.device else config['device']
        if args.no_amp: config['amp'] = False

    cache_dir = os.path.join(project_root, 'save', 'eval','eval_all','eval_cache', exp_name.replace(' ', '_'))
    os.makedirs(cache_dir, exist_ok=True)
    config['cache_path'] = os.path.join(cache_dir, 'hash_cache.npz')
    
    model = create_hashing_model(config['backbone'], config['strategy'], config).to(config['device'])
    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    logger.info("Weights loaded.")
    
    binary_hashes, real_hashes = generate_hashes(model, config)
    roc_results = compute_roc_metrics(binary_hashes, real_hashes, config, tpr_targets=[0.95])
    
    return {'experiment_name': exp_name, 'model_short': model_info['model_short'], 'auc_hd': roc_results['Hamming']['auc'], 'auc_mse': roc_results['MSE']['auc'], 'fpr95_hd': roc_results['Hamming']['fpr_at_tpr'][0.95], 'fpr95_mse': roc_results['MSE']['fpr_at_tpr'][0.95]}
        
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate AUC and thresholds on PIHD dataset')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--num-workers', type=int, default=None, help='Override dataloader workers')
    parser.add_argument('--device', type=str, default=None, help="Device, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--mode', choices=['64', '32', 'thr'], default=None, help='Run mode')
    return parser.parse_args()

def main():
    args = parse_args()
    bit64_models, bit32_models = find_all_trained_models()
    
    def evaluate_and_save(models, bit_type):
        if not models:
            logger.info(f"No {bit_type} models found!")
            return
        
        all_results = [evaluate_single_model(m, args=args) for m in models]
        
        # Add traditional hashes to 32/64 bit results
        if bit_type == '32bit':
            logger.info("Evaluating traditional hashes (phash/whash) at 32-bit...")
            base_config = {'dataset_path': os.path.join(project_root, 'datasets/PIHD/test/test_class'), 'group_size': 97}
            for ht in ['phash', 'whash']:
                hashes = generate_traditional_hashes_32(base_config, ht, max_workers=12)
                roc = compute_roc_metrics_traditional(hashes, base_config, dist_name=f'{ht.upper()}-32')
                all_results.append({'experiment_name': f'{ht.upper()}_32', 'model_short': 'Traditional', 'auc_hd': roc['auc'], 'auc_mse': np.nan, 'fpr95_hd': roc['fpr_at_tpr'][0.95], 'fpr95_mse': np.nan})
        
        if bit_type == '64bit':
            logger.info("Evaluating traditional hashes (phash/whash) at 64-bit...")
            base_config = {'dataset_path': os.path.join(project_root, 'datasets/PIHD/test/test_class'), 'group_size': 97}
            for ht in ['phash', 'whash']:
                hashes = generate_traditional_hashes_64(base_config, ht, max_workers=12)
                roc = compute_roc_metrics_traditional(hashes, base_config, dist_name=f'{ht.upper()}-64')
                all_results.append({'experiment_name': f'{ht.upper()}_64', 'model_short': 'Traditional', 'auc_hd': roc['auc'], 'auc_mse': np.nan, 'fpr95_hd': roc['fpr_at_tpr'][0.95], 'fpr95_mse': np.nan})
        
        df = pd.DataFrame(all_results).sort_values('experiment_name')
        output_dir = os.path.join(project_root, 'save', 'eval','eval_all')
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f'all_models_roc_evaluation_{bit_type}.csv')
        df.to_csv(csv_path, index=False, float_format='%.6f')
        
        logger.info(f"EVALUATION SUMMARY ({bit_type.upper()})")
        logger.info(f"Total {bit_type} models evaluated: {len(all_results)}")
        logger.info(f"Results saved to: {csv_path}")
        print(f"\n{bit_type.upper()} Results Summary:\n{df[['experiment_name', 'auc_hd', 'auc_mse']].to_string(index=False)}")
        
        valid_results = df[~df['auc_hd'].isna()]
        if not valid_results.empty:
            best_hd = valid_results.loc[valid_results['auc_hd'].astype(float).idxmax()]
            logger.info(f"\nBest AUC(HD) ({bit_type}): {best_hd['auc_hd']:.6f} - {best_hd['experiment_name']}")
            if (~df['auc_mse'].isna()).any():
                best_mse = df.loc[df['auc_mse'].astype(float).idxmax()]
                logger.info(f"Best AUC(MSE) ({bit_type}): {best_mse['auc_mse']:.6f} - {best_mse['experiment_name']}")
    
    if args.mode:
        choice = args.mode
    else:
        choice = input("Select evaluation type (64 / 32 / thr): ").strip().lower()

    if choice == '64':
        evaluate_and_save(bit64_models, '64bit')
    elif choice == '32':
        evaluate_and_save(bit32_models, '32bit')
    elif choice == 'thr':
        run_threshold_evaluation(args=args)
    else:
        print("Invalid input.")

if __name__ == "__main__":
    main()