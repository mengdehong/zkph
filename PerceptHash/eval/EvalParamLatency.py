import os
import sys
import torch
import numpy as np
import time
from loguru import logger
from thop import profile, clever_format

# --- Setup & Imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from model.ModelFactory import create_hashing_model

def get_model_config_from_name(exp_name):
    """Infers model configuration from the experiment name."""
    config = {
        'bit_list': [64], 'crop_size': 224, 'device': 'cuda:0',
        'groupmamba_tiny_params': {
            "num_classes": -1, "embed_dims": [64, 128, 348, 448], 
            "depths": [3, 4, 9, 3], "mlp_ratios": [8, 8, 4, 4], 
            "stem_hidden_dim": 32, "k_size": [3, 3, 5, 5]
        }
    }
    
    if 'ResNet' in exp_name: backbone = 'resnet50'
    elif 'ViT' in exp_name: backbone = 'vit'
    elif 'MambaOut' in exp_name: backbone = 'mambaout_tiny'
    elif 'GroupMamba' in exp_name: backbone = 'groupmamba'
    elif 'ConvNeXtV2' in exp_name: backbone = 'convnextv2'
    elif 'SwinTiny' in exp_name: backbone = 'swin_tiny'
    else: raise ValueError(f"Cannot infer model type from: {exp_name}")
    
    strategy = 'enhanced' if 'Enhanced' in exp_name else 'traditional'
    
    config['backbone'] = backbone
    config['strategy'] = strategy
    return config

def count_parameters(model):
    """Counts model parameters."""
    return sum(p.numel() for p in model.parameters())

def calculate_flops(model, input_size=(1, 3, 224, 224), device='cuda:0'):
    """Calculates model FLOPs."""
    try:
        if device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU for FLOPs")
            device = 'cpu'
        
        model = model.to(device).eval()
        dummy_input = torch.randn(input_size).to(device)
        
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
        
        logger.info(f"FLOPs calculation successful: {flops_formatted}")
        return flops, params
        
    except Exception as e:
        logger.warning(f"FLOPs calculation failed: {e}")
        return 0, 0

def measure_latency(model, input_size=(1, 3, 224, 224), device='cuda:0', num_runs=100, warm_up_runs=20):
    """Measures model inference latency."""
    if device.startswith('cuda') and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    model.to(device).eval()
    dummy_input = torch.randn(input_size).to(device)
    latencies = []
    
    with torch.no_grad():
        # Warm-up runs
        for _ in range(warm_up_runs):
            if device.startswith('cuda'): torch.cuda.synchronize()
            _ = model(dummy_input)
            if device.startswith('cuda'): torch.cuda.synchronize()
        
        # Latency measurement
        for _ in range(num_runs):
            if device.startswith('cuda'): torch.cuda.synchronize()
            start_time = time.time()
            _ = model(dummy_input)
            if device.startswith('cuda'): torch.cuda.synchronize()
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)
    
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    return avg_latency, std_latency

def evaluate_model_complexity(exp_name, weight_path, model_short):
    """Evaluates a single model's complexity (params, FLOPs, latency)."""
    logger.info(f"Evaluating: {exp_name}")
    
    try:
        config = get_model_config_from_name(exp_name)
        model = create_hashing_model(config['backbone'], config['strategy'], config)
        
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Weights loaded from: {weight_path}")
        else:
            logger.warning(f"Weight file not found: {weight_path}")
        
        params = count_parameters(model)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Calculate FLOPs
        model_for_flops = create_hashing_model(config['backbone'], config['strategy'], config)
        flops, _ = calculate_flops(model_for_flops, device=device)
        del model_for_flops

        # Measure latency
        model_for_latency = create_hashing_model(config['backbone'], config['strategy'], config)
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location='cpu')
            model_for_latency.load_state_dict(state_dict, strict=False)
        
        avg_latency, std_latency = measure_latency(model_for_latency, device=device)
        del model_for_latency
        
        params_formatted = f"{params / 1e6:.2f}M"
        flops_formatted = f"{flops / 1e9:.2f}G" if flops > 0 else "Error"
        latency_formatted = f"{avg_latency:.2f}ms ± {std_latency:.2f}ms"
        
        return {
            'model_short': model_short,
            'experiment_name': exp_name,
            'params': params,
            'params_formatted': params_formatted,
            'flops': flops,
            'flops_formatted': flops_formatted,
            'avg_latency': avg_latency,
            'std_latency': std_latency,
            'latency_formatted': latency_formatted
        }
        
    except Exception as e:
        logger.error(f"Error evaluating {exp_name}: {str(e)}")
        return {
            'model_short': model_short,
            'experiment_name': exp_name,
            'params': 0, 'params_formatted': 'Error',
            'flops': 0, 'flops_formatted': 'Error',
            'avg_latency': 0, 'std_latency': 0, 'latency_formatted': 'Error'
        }

def main():
    """Main function to evaluate model complexity and latency."""
    logger.info("Starting model complexity and latency evaluation...")
    
    models_info = [
        ("ResNet50_Baseline", "ResNet", "save/TrainAll/ResNet/ResNet50_BsaeLine_best.pth"),
        ("MambaOut_Baseline", "MambaOut", "save/TrainAll/MambaOut/best_model.pth"),
        ("ViT_Baseline", "ViT", "save/TrainAll/ViT/best_model.pth"),
        ("GroupMamba_Baseline", "GroupMamba", "save/TrainAll/GroupMamba/GroupMamba_Baseline_best.pth"),
        ("ConvNeXtV2_Baseline", "ConvNeXtV2", "save/TrainAll/ConvNeXtV2/ConvNeXtV2_Baseline_best.pth"),
        ("SwinTiny_Baseline", "SwinTiny", "save/TrainAll/SwinTiny/SwinTiny_Baseline_best.pth"),
        ("ResNet50_Enhanced", "ResNet", "save/TrainAll/ResNet+/ResNet50_Enhanced_best.pth"),
        ("MambaOut_Enhanced", "MambaOut", "save/TrainAll/MambaOut+/best_model.pth"),
        ("ViT_Enhanced", "ViT", "save/TrainAll/ViT+/best_model.pth"),
        ("GroupMamba_Enhanced", "GroupMamba", "save/TrainAll/GroupMamba+/GroupMamba_Enhanced_best.pth"),
        ("ConvNeXtV2_Enhanced", "ConvNeXtV2", "save/TrainAll/ConvNeXtV2+/ConvNeXtV2_Enhanced_best.pth"),
        ("SwinTiny_Enhanced", "SwinTiny", "save/TrainAll/SwinTiny+/SwinTiny_Enhanced_best.pth"),
    ]
    
    results = [evaluate_model_complexity(exp_name, os.path.join(project_root, rel_path), model_short) for exp_name, model_short, rel_path in models_info]
    
    print(f"\n{'='*100}")
    print("MODEL COMPLEXITY AND LATENCY EVALUATION RESULTS")
    print(f"{'='*100}")
    print(f"{'Type':<10} {'Model':<12} {'Params':<12} {'FLOPs':<12} {'Latency':<18}")
    print("-" * 100)
    
    def print_results(group_type, models):
        print(group_type)
        for model_short in models:
            result = next((r for r in results if r['model_short'] == model_short and group_type.replace(' ', '') in r['experiment_name']), None)
            if result:
                print(f"{'':^10} {result['model_short']:<12} {result['params_formatted']:<12} {result['flops_formatted']:<12} {result['latency_formatted']:<18}")
    
    baseline_models = ["ResNet", "MambaOut", "ViT", "GroupMamba", "ConvNeXtV2", "SwinTiny"]
    print_results("Baseline", baseline_models)
    print_results("Enhanced", baseline_models)
    
    print(f"{'='*100}")
    
    output_dir = os.path.join(project_root, 'save', 'eval', 'eval_all')
    os.makedirs(output_dir, exist_ok=True)
    
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'model_complexity_latency_evaluation.csv')
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Detailed results saved to: {csv_path}")

if __name__ == "__main__":
    main()