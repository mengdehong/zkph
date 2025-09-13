import torch
import optuna
import os
import sys
import traceback
from loguru import logger

# Ensure project path is in sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import factory functions and utilities
from model.ModelFactory import create_hashing_model
from utils.LossFactory import create_loss_function
from utils.tools import config_dataset, get_data, validate_pihd

def objective(trial, experiment_config, global_config, data_loaders):
    """
    The Optuna objective function.
    Trains and evaluates for a short period with suggested hyperparameters.
    """
    exp_name = experiment_config['name']
    backbone_name = experiment_config['backbone']
    strategy_name = experiment_config['strategy']
    
    device = torch.device(global_config['device'])
    train_loader, test_loader, db_loader = data_loaders

    try:
        # --- 1. Suggest hyperparameters ---
        trial_config = global_config.copy()
        trial_config['lr'] = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        trial_config['weight_decay'] = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
        trial_config['alpha'] = trial.suggest_float('alpha', 1e-3, 5e-2, log=True)
        trial_config['beta'] = trial.suggest_float('beta', 1e-6, 1e-4, log=True)

        # Suggest margin based on strategy
        if strategy_name == 'BaseLine':
            trial_config['margin'] = trial.suggest_float('margin', 0.2, 1.0)
        elif strategy_name == 'enhanced':
            trial_config['angular_margin'] = trial.suggest_float('angular_margin', 0.05, 0.5)

        # --- 2. Build model and loss with suggested parameters ---
        model = create_hashing_model(backbone_name, strategy_name, trial_config).to(device)
        loss_fn = create_loss_function(strategy_name, trial_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=trial_config['lr'], weight_decay=trial_config['weight_decay'])
        
        # --- 3. Run a short training and validation loop ---
        best_roc_auc_in_trial = 0.0
        opt_epochs = global_config['opt_epochs']
        ver_freq = opt_epochs // 3 # Validate 3 times

        for epoch in range(opt_epochs):
            # Train for one epoch
            model.train()
            for (anchor, positive, negative) in train_loader:
                anchor, pos, neg = anchor.to(device), positive.to(device), negative.to(device)
                optimizer.zero_grad()
                h_a, h_p, h_n = model(anchor), model(pos), model(neg)
                total_loss, _, _, _ = loss_fn(h_a, h_p, h_n)
                total_loss.backward()
                optimizer.step()

            # Validate at specified frequency
            if (epoch + 1) % ver_freq == 0 or (epoch + 1) == opt_epochs:
                model.eval()
                with torch.no_grad():
                    _, roc_auc, _ = validate_pihd(
                        trial_config, trial_config['bit_list'][0], 0.0,
                        test_loader, db_loader, model, epoch
                    )
                best_roc_auc_in_trial = max(best_roc_auc_in_trial, roc_auc)
        
        logger.info(f"Trial {trial.number} for '{exp_name}' finished. Best ROC-AUC: {best_roc_auc_in_trial:.4f}")
        
        # --- 4. Return final performance metric ---
        return best_roc_auc_in_trial

    except Exception as e:
        logger.error(f"Trial {trial.number} for '{exp_name}' failed: {e}")
        logger.error(traceback.format_exc())
        raise optuna.exceptions.TrialPruned()

def run_optimization(config):
    """Main function to iterate and run Optuna optimization for all experiments."""
    
    # 1. Load dataset once for all experiments
    logger.info("Loading dataset once for all optimization trials...")
    config = config_dataset(config)
    train_loader, test_loader, db_loader, _, _, _ = get_data(config)
    data_loaders = (train_loader, test_loader, db_loader)
    
    # 2. Iterate through all experiments to optimize
    for exp_config in config['experiments_to_optimize']:
        exp_name = exp_config['name']
        logger.info(f"\n{'='*20} Starting Hyperparameter Optimization for: {exp_name} {'='*20}")

        # 3. Create a separate study and database file for each experiment
        study_name = exp_name
        db_dir = "./save/optuna/All/"
        os.makedirs(db_dir, exist_ok=True)
        storage_path = f"sqlite:///{os.path.join(db_dir, exp_name)}.db"
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            direction='maximize', 
            load_if_exists=True
        )

        completed = sum(1 for t in study.get_trials() if t.state == optuna.trial.TrialState.COMPLETE)
        if completed >= config['opt_trials'] and getattr(study, "best_trial", None) is not None:
            logger.info(f"Study for '{exp_name}' already has {completed} completed trials. Skipping optimization.")
            logger.info(f"Existing best trial: #{study.best_trial.number}, Best ROC-AUC: {study.best_value:.4f}")
            continue

        # 4. Run optimization
        study.optimize(
            lambda trial: objective(trial, exp_config, config, data_loaders),
            n_trials=config['opt_trials']
        )
        
        # 5. Print best result
        logger.info(f"Optimization finished for '{exp_name}'.")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info("Best parameters found:")
        for key, value in study.best_trial.params.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"Best ROC-AUC: {study.best_value:.4f}")

if __name__ == "__main__":
    # --- Global Configuration ---
    config = {
        # Experiments to optimize
        'experiments_to_optimize': [
            {'name': 'ResNet50_BaseLine', 'backbone': 'resnet50', 'strategy': 'traditional'},
            {'name': 'ResNet50_Enhanced', 'backbone': 'resnet50', 'strategy': 'enhanced'},
            {'name': 'ViT_Small_BaseLine', 'backbone': 'vit', 'strategy': 'traditional'},
            {'name': 'ViT_Small_Enhanced', 'backbone': 'vit', 'strategy': 'enhanced'},
            {'name': 'MambaOut_BaseLine', 'backbone': 'mambaout_tiny', 'strategy': 'traditional'},
            {'name': 'MambaOut_Enhanced', 'backbone': 'mambaout_tiny', 'strategy': 'enhanced'},
            {'name': 'GroupMamba_BaseLine', 'backbone': 'groupmamba', 'strategy': 'traditional'},
            {'name': 'GroupMamba_Enhanced', 'backbone': 'groupmamba', 'strategy': 'enhanced'},
            {'name': 'ConvNeXtV2_BaseLine', 'backbone': 'convnextv2', 'strategy': 'traditional'},
            {'name': 'ConvNeXtV2_Enhanced', 'backbone': 'convnextv2', 'strategy': 'enhanced'},
            {'name': 'SwinTiny_BaseLine', 'backbone': 'swin_tiny', 'strategy': 'traditional'},
            {'name': 'SwinTiny_Enhanced', 'backbone': 'swin_tiny', 'strategy': 'enhanced'},
        ],
        # Optuna configuration
        'opt_trials': 10,
        'opt_epochs': 15,
        
        # Fixed model and data configuration
        'groupmamba_tiny_params': {
            "num_classes": -1, "embed_dims": [64, 128, 348, 448], "depths": [3, 4, 9, 3],
            "mlp_ratios": [8, 8, 4, 4], "stem_hidden_dim": 32, "k_size": [3, 3, 5, 5]
        },
        'bit_list': [64],
        'batch_size': 64, # RTXA6000: 48 for GroupMamba, 64 for others
        'resize_size': 256, 'crop_size': 224, 'num_workers': 12, 
        'dataset': "pihd",
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'preload_data': True, 'group_size': 97
    }
    
    # Configure logging
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    logger.add(f"{log_dir}/hyperparam_optimization.log", rotation="10 MB")

    run_optimization(config)