import os
import sys
import imagehash
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from eval.Common import *

def compute_phash_path(file_path):
    """Computes phash for an image path, returns (basename, hex_hash)."""
    img = Image.open(file_path).convert('RGB')
    ph = imagehash.phash(img)
    return os.path.basename(file_path), str(ph)

def generate_phash_hashes(config):
    """Generates perceptual hashes (phash) for images, using caching."""
    cache_path = config['cache_path']
    file_paths = config.get('file_paths', [])
    
    phash_hashes = {}
    if os.path.exists(cache_path):
        cache = np.load(cache_path, allow_pickle=True)
        if 'phash_hashes' in cache:
            phash_hashes = cache['phash_hashes'].item()
            missing = [os.path.basename(p) for p in file_paths if os.path.basename(p) not in phash_hashes]
            if not missing: return phash_hashes
            logger.warning(f"Cache missing {len(missing)} files, recomputing...")
            file_paths = [p for p in file_paths if os.path.basename(p) in missing] 

    if file_paths:
        max_workers = config.get('max_workers', 4)
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(file_paths), desc="Generating phash") as pbar:
            futures = {executor.submit(compute_phash_path, p): os.path.basename(p) for p in file_paths}
            for fut in as_completed(futures): 
                fname, phash_str = fut.result()
                if phash_str:
                    phash_hashes[fname] = phash_str
                pbar.update(1)
        
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, phash_hashes=phash_hashes)
        logger.info(f"Saved phash hashes to cache: {cache_path}")

    logger.info(f"Generated {len(phash_hashes)} phash hashes.")
    return phash_hashes

def phash_distance(h1, h2):
    """Calculates Hamming distance between two phash hex strings."""
    return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)

def compute_roc_metrics_coco(binary_hashes, real_hashes, phash_hashes, config, tpr_targets=[0.95, 0.99]):
    """Computes ROC metrics using Hamming, MSE, and Phash distances."""
    pairs_csv_dir = config['pairs_csv_dir']
    identical_pairs = pd.read_csv(os.path.join(pairs_csv_dir, 'identical_pairs.csv'))
    distinct_pairs = pd.read_csv(os.path.join(pairs_csv_dir, 'distinct_pairs.csv'))

    dist_configs = []
    if binary_hashes is not None and real_hashes is not None:
        dist_configs.extend([("Hamming", binary_hashes, hamming_distance), ("MSE", real_hashes, mse_distance)])
    if phash_hashes is not None:
        dist_configs.append(("Phash", phash_hashes, phash_distance))

    roc_results = {}
    for dist_name, hashes_dict, dist_func in dist_configs:
        labels, distances = [], []
        for df, label in [(identical_pairs, 1), (distinct_pairs, 0)]:
            for _, row in tqdm(df.iterrows(), desc=f"Processing pairs ({dist_name}, label={label})"):
                img1_fname, img2_fname = os.path.basename(row['image1_path']), os.path.basename(row['image2_path'])
                if img1_fname in hashes_dict and img2_fname in hashes_dict:
                    dist = dist_func(hashes_dict[img1_fname], hashes_dict[img2_fname])
                    distances.append(dist); labels.append(label)

        scores = -np.array(distances)
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        fpr_at_tpr = {target: fpr[np.argmin(np.abs(tpr - target))] for target in tpr_targets}
        targets_str = ", ".join([f"FPR@{int(target*100)}%: {fpr_at_tpr[target]:.6f}" for target in tpr_targets])
        logger.info(f"{dist_name} AUC: {roc_auc:.4f}, {targets_str}")
        roc_results[dist_name] = {'auc': roc_auc, 'fpr': fpr, 'tpr': tpr, 'fpr_at_tpr': fpr_at_tpr}
    return roc_results

def find_models_to_evaluate():
    """Returns hardcoded models to evaluate, filtering by existing weights."""
    models_data = [
        ("phash", "phash", None),
        ("ViT_Small_Baseline", "ViT-S", "save/TrainAll/ViT/best_model.pth"),
        ("ViT_Small_Enhanced", "ViT-S+", "save/TrainAll/ViT+/best_model.pth"),
        ("MambaOut_Baseline", "MambaOut", "save/TrainAll/MambaOut/best_model.pth"),
        ("MambaOut_Enhanced", "MambaOut+", "save/TrainAll/MambaOut+/best_model.pth"),
    ]
    evaluated_models = []
    for exp_name, model_short, rel_path in models_data:
        if rel_path:
            weight_path = os.path.join(project_root, rel_path)
            if os.path.exists(weight_path):
                evaluated_models.append({'experiment_name': exp_name, 'model_short': model_short, 'weight_path': weight_path})
                logger.info(f"Using model: {exp_name} -> {weight_path}")
            else:
                logger.warning(f"Weight not found, skipping: {weight_path}")
        else:
            evaluated_models.append({'experiment_name': exp_name, 'model_short': model_short, 'weight_path': None})
            logger.info(f"Using model: {exp_name}")
    logger.info(f"Total models to evaluate: {len(evaluated_models)}")
    return evaluated_models

def evaluate_single_model(model_info):
    """Evaluates a single model, returning its metrics."""
    exp_name, weight_path = model_info['experiment_name'], model_info['weight_path']
    logger.info(f"\nEvaluating: {exp_name}")

    # Prepare dataset and pairs paths used by both branches
    pairs_csv_dir = os.path.join(project_root, 'datasets/CocoVal2017/pairs_csv')
    origin_dir = os.path.join(project_root, 'datasets/CocoVal2017/origin')
    transformed_dir = os.path.join(project_root, 'datasets/CocoVal2017/transformed')
    all_images = []
    for img_dir in [origin_dir, transformed_dir]:
        if os.path.exists(img_dir):
            all_images.extend([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    file_paths = sorted(list(set(all_images)))

    # Special-case: phash does not require a deep model; avoid get_model_config_from_name
    if exp_name == 'phash':
        phash_config = {
            'pairs_csv_dir': pairs_csv_dir,
            'cache_path': os.path.join(project_root, 'save', 'eval', 'eval_coco', 'eval_cache', exp_name.replace(' ', '_'), 'hash_cache.npz'),
            'file_paths': file_paths,
            'max_workers': 4,
        }
        os.makedirs(os.path.dirname(phash_config['cache_path']), exist_ok=True)
        phash_hashes = generate_phash_hashes(phash_config)
        roc_results = compute_roc_metrics_coco(None, None, phash_hashes, {'pairs_csv_dir': pairs_csv_dir})
        return {'experiment_name': exp_name, 'model_short': model_info['model_short'], 'auc_hd': roc_results['Phash']['auc'], 'saved_at': 'N/A'}

    # For learned models, proceed with standard model config
    config = get_model_config_from_name(exp_name)
    config.update({
        'dataset_path': origin_dir,
        'pairs_csv_dir': pairs_csv_dir,
        'batch_size': 512, 'num_workers': 12, 'amp': True,
        'cache_path': os.path.join(project_root, 'save', 'eval', 'eval_coco', 'eval_cache', exp_name.replace(' ', '_'), 'hash_cache.npz')
    })
    os.makedirs(os.path.dirname(config['cache_path']), exist_ok=True)
    config['file_paths'] = file_paths

    try:
        model = create_hashing_model(config['backbone'], config['strategy'], config).to(config['device'])
        state_dict = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        binary_hashes, real_hashes = generate_hashes(model, config)
        roc_results = compute_roc_metrics_coco(binary_hashes, real_hashes, None, {'pairs_csv_dir': config['pairs_csv_dir']})
        
        return {
            'experiment_name': exp_name, 'model_short': model_info['model_short'],
            'auc_hd': roc_results['Hamming']['auc'], 'fpr95_hd': roc_results['Hamming']['fpr_at_tpr'][0.95],
            'fpr99_hd': roc_results['Hamming']['fpr_at_tpr'][0.99], 'fpr99_mse': roc_results['MSE']['fpr_at_tpr'][0.99],
            'saved_at': model_info.get('saved_at', 'N/A')
        }
    except Exception as e:
        logger.error(f"Error evaluating {exp_name}: {str(e)}")
        return {'experiment_name': exp_name, 'model_short': model_info['model_short'], 'auc_hd': np.nan, 'fpr95_hd': np.nan, 'fpr99_hd': np.nan, 'fpr99_mse': np.nan, 'status': f'Error: {str(e)}'}

def main():
    """Main function to orchestrate model evaluation and save results."""
    logger.info("Starting evaluation on COCO dataset...")
    models_to_eval = find_models_to_evaluate()
    if not models_to_eval:
        logger.error("No models found for evaluation!")
        return

    all_results = [evaluate_single_model(m) for m in models_to_eval]
    df = pd.DataFrame(all_results)
    df = df.drop(columns=[c for c in ['fpr99_hd', 'fpr99_mse', 'saved_at'] if c in df.columns])
    df = df.sort_values('experiment_name')

    output_dir = os.path.join(project_root, 'save', 'eval', 'eval_coco')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'coco_models_evaluation.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')

    logger.info(f"\nEvaluation Summary:")
    logger.info(f"Total models evaluated: {len(all_results)}")
    logger.info(f"Results saved to: {csv_path}")
    print("\nResults Summary:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()