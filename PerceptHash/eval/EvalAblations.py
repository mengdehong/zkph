#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViT ablation study evaluation script.
Evaluates all ablation experiments using FPR@95TPR (Hamming) metric.
"""

import os
import sys
import pandas as pd
import torch
from PIL import ImageFile
from loguru import logger

# Global setup and path imports
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

# Import necessary modules
from model.ModelFactory import create_hashing_model
from eval.Common import generate_hashes, compute_roc_metrics


def get_ablation_models():
    """Returns configurations for all ablation models."""
    ablation_models = [
        {
            'experiment_name': 'ViT_Baseline',
            'ablation_id': '1',
            'backbone': 'vit',
            'strategy': 'traditional',
            'weight_path': 'save/TrainAll/ViT/best_model.pth',
            'description': 'Baseline: Traditional strategy + Tanh head'
        },
        {
            'experiment_name': 'ViT_Ablation_BN',
            'ablation_id': '2', 
            'backbone': 'vit',
            'strategy': 'ablation_bn',
            'weight_path': 'save/ablations/2/ViT_Ablation_BN_best.pth',
            'description': '+BN head: BN head + Traditional loss'
        },
        {
            'experiment_name': 'ViT_Ablation_Angular_NoABL',
            'ablation_id': '3',
            'backbone': 'vit', 
            'strategy': 'ablation_angular_no_abl',
            'weight_path': 'save/ablations/3/ViT_Ablation_Angular_NoABL_best.pth',
            'description': '+BN head + Angular Loss (no ABL): BN head + Angular loss (no ABL)'
        },
        {
            'experiment_name': 'ViT_Enhanced',
            'ablation_id': '4',
            'backbone': 'vit',
            'strategy': 'enhanced', 
            'weight_path': 'save/TrainAll/ViT+/best_model.pth',
            'description': '+BN head + Angular Loss (with ABL): BN head + Full Angular loss'
        }
    ]
    
    # Check if weight files exist
    valid_models = []
    for model_info in ablation_models:
        weight_path = os.path.join(project_root, model_info['weight_path'])
        if os.path.exists(weight_path):
            model_info['weight_path'] = weight_path
            valid_models.append(model_info)
        else:
            logger.warning(f"Weight file not found: {weight_path}")
    
    logger.info(f"Found {len(valid_models)} valid ablation models")
    return valid_models

def get_model_config_from_strategy(strategy):
    """Gets model configuration based on strategy."""
    config = {
        'bit_list': [64],
        'crop_size': 224,
        'device': 'cuda:0'
    }
    config['backbone'] = 'vit'
    config['strategy'] = strategy
    return config

def evaluate_single_ablation(model_info):
    """Evaluates a single ablation model."""
    exp_name = model_info['experiment_name']
    weight_path = model_info['weight_path']
    strategy = model_info['strategy']
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating: {exp_name}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Weights: {weight_path}")
    logger.info(f"Description: {model_info['description']}")
    logger.info(f"{'='*50}")
    
    config = get_model_config_from_strategy(strategy)
    config.update({
        'dataset_path': os.path.join(project_root, 'datasets/PIHD/test/test_class'),
        'group_size': 97,
        'batch_size': 512,
        'num_workers': 12,
    })
    
    cache_dir = os.path.join(project_root, 'save', 'ablations', 'eval_cache', exp_name.replace(' ', '_'))
    os.makedirs(cache_dir, exist_ok=True)
    config['cache_path'] = os.path.join(cache_dir, 'hash_cache.npz')
    
    try:
        model = create_hashing_model(config['backbone'], config['strategy'], config).to(config['device'])
        
        state_dict = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        logger.info("Weights loaded.")
        
        binary_hashes, real_hashes = generate_hashes(model, config)
        roc_results = compute_roc_metrics(binary_hashes, real_hashes, config, tpr_targets=[0.95])

        return {
            'experiment_name': exp_name,
            'ablation_id': model_info['ablation_id'],
            'strategy': strategy,
            'description': model_info['description'],
            'fpr95_hd': roc_results['Hamming']['fpr_at_tpr'][0.95],
            'status': 'Success'
        }
        
    except Exception as e:
        logger.error(f"Error evaluating {exp_name}: {str(e)}")
        return {
            'experiment_name': exp_name,
            'ablation_id': model_info['ablation_id'],
            'strategy': strategy,
            'description': model_info['description'],
            'fpr95_hd': float('nan'),
            'status': f'Error: {str(e)}'
        }

def main():
    """Main function: evaluates all ViT ablation experiments."""
    logger.info("Starting ViT Ablation Evaluation...")
    
    ablation_models = get_ablation_models()
    if not ablation_models:
        logger.error("No valid ablation models found!")
        return
    
    all_results = [evaluate_single_ablation(model_info) for model_info in ablation_models]
    
    df = pd.DataFrame(all_results)
    if 'fpr95_hd' in df.columns:
        df['fpr95_hd_percent'] = df['fpr95_hd'] * 100.0

    df = df.sort_values('ablation_id')
    keep_cols = ['ablation_id', 'experiment_name', 'strategy', 'description', 'fpr95_hd_percent']
    df = df[[c for c in keep_cols if c in df.columns]]

    output_dir = os.path.join(project_root, 'save', 'ablations')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'vit_ablation_evaluation.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    
    logger.info(f"\n{'='*60}")
    logger.info("ViT ABLATION EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total models evaluated: {len(all_results)}")
    logger.info(f"Results saved to: {csv_path}")
    
    print("\nViT Ablation Results Summary:")
    display_df = df[['ablation_id', 'experiment_name', 'fpr95_hd_percent', 'description']].copy()
    display_df.columns = ['ID', 'Experiment', 'FPR@95TPR (%)', 'Description']
    print(display_df.to_string(index=False))

    if 'fpr95_hd' in pd.DataFrame(all_results).columns:
        df_all = pd.DataFrame(all_results)
        valid_results = df_all[df_all['fpr95_hd'].notna()]
        if not valid_results.empty:
            best = valid_results.loc[valid_results['fpr95_hd'].astype(float).idxmin()]
            print(f"\nBest FPR@95TPR (Hamming): {best['experiment_name']} ({best['fpr95_hd']*100:.4f}%)")
    
    logger.info("ViT Ablation Evaluation Completed!")

if __name__ == "__main__":
    main()