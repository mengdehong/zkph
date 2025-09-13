import torch
import torch.nn as nn
import sys
import os
import traceback
import time
import json
from loguru import logger
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.ModelFactory import create_hashing_model
from utils.LossFactory import create_loss_function
from utils.tools import config_dataset, get_data, validate_pihd

class ExperimentTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.all_results = {}

    def get_model_short_name(self, backbone, strategy):
        b = backbone.lower()
        plus = '+' if strategy and strategy.lower() == 'enhanced' else ''
        if 'vit' in b:
            return f"ViT{plus}"
        if 'resnet' in b:
            return f"ResNet{plus}"
        if 'mambaout' in b:
            return f"MambaOut{plus}"
        if 'groupmamba' in b:
            return f"GroupMamba{plus}"
        if 'convnextv2' in b:
            return f"ConvNeXtV2{plus}"
        if 'swin' in b:
            return f"SwinTiny{plus}"
        return f"{backbone.capitalize()}{plus}"

    def run_all_experiments(self, train_loader, test_loader, db_loader):
        """Iterates and runs all experiments defined in the config file."""
        for exp_config in self.config['experiments']:
            exp_name = exp_config['name']
            backbone = exp_config['backbone']
            strategy = exp_config['strategy']
            
            logger.info(f"\n{'='*20} Starting Experiment: {exp_name} {'='*20}")
            logger.info(f"Backbone: {backbone}, Strategy: {strategy}")
            
            try:
                self.run_single_experiment(exp_name, backbone, strategy, train_loader, test_loader, db_loader)
            except Exception as e:
                logger.error(f"Experiment '{exp_name}' failed with error: {e}")
                logger.error(traceback.format_exc())
        
        self.save_all_results()

    def run_single_experiment(self, exp_name, backbone_name, strategy_name, train_loader, test_loader, db_loader):
        """Runs a single complete experiment."""

        # Get experiment-specific config
        exp_config = next(exp for exp in self.config['experiments'] if exp['name'] == exp_name)

        # Create full config by merging global and experiment-specific settings
        current_config = self.config.copy()
        current_config.update(exp_config)

        logger.info(
            f"Using experiment-specific config: lr={current_config.get('lr')}, "
            f"weight_decay={current_config.get('weight_decay')}, "
            f"alpha={current_config.get('alpha')}, beta={current_config.get('beta')}"
        )
        bit = int(current_config['bit_list'][0])
        bit_suffix = f"-b{bit}"

        # 1. Create model and loss function
        model = create_hashing_model(backbone_name, strategy_name, current_config).to(self.device)
        loss_fn = create_loss_function(strategy_name, current_config)

        # 2. Create optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=current_config['lr'],
            weight_decay=current_config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=current_config['epoch'],
            eta_min=1e-6
        )

        # 3. Training loop
        best_metric = {'roc_auc': 0.0, 'epoch': 0, 'mAP': 0.0, 'eer': 1.0}
        model_short = self.get_model_short_name(backbone_name, strategy_name)
        all_dir = os.path.join('tmp', 'TrainAll', model_short)
        os.makedirs(all_dir, exist_ok=True)

        for epoch in range(current_config['epoch']):
            logger.info(f"\n--- Experiment '{exp_name}' | Epoch {epoch+1}/{current_config['epoch']} ---")

            # Training
            model.train()
            for (anchor, positive, negative) in train_loader:
                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
                optimizer.zero_grad()
                h_anchor, h_positive, h_negative = model(anchor), model(positive), model(negative)
                total_loss, _, _, _ = loss_fn(h_anchor, h_positive, h_negative)
                total_loss.backward()
                optimizer.step()

            scheduler.step()

            # Evaluation
            if (epoch + 1) % current_config['ver_roc'] == 0:
                model.eval()
                mAP, roc_auc, eer = validate_pihd(
                    current_config,
                    current_config['bit_list'][0],
                    best_metric['roc_auc'],
                    test_loader,
                    db_loader,
                    model,
                    epoch
                )

                if roc_auc > best_metric['roc_auc']:
                    logger.info(
                        f"🎉 New best performance for '{exp_name}'! ROC-AUC: {roc_auc:.4f} at epoch {epoch+1}"
                    )
                    best_metric.update({'roc_auc': roc_auc, 'epoch': epoch+1, 'mAP': mAP, 'eer': eer})
                    # Save best model checkpoint
                    self.save_checkpoint(
                        model.state_dict(),
                        f"{model_short}",
                        f"{exp_name}{bit_suffix}_best.pth",
                        base_save_dir=os.path.join('save', 'TrainAll')
                    )

        self.all_results[f"{exp_name}{bit_suffix}"] = best_metric
        logger.info(f"--- Experiment '{exp_name}' Finished ---")
        logger.info(f"Best Result: {best_metric}")
        try:
            # Save experiment summary
            result_file = os.path.join(all_dir, f"results_{exp_name}{bit_suffix}.json")
            with open(result_file, 'w') as rf:
                json.dump(
                    {
                        'experiment': exp_name,
                        'model_short': model_short,
                        'best_metric': best_metric,
                        'config': current_config
                    },
                    rf,
                    indent=2
                )
            logger.info(f"Saved experiment summary to {result_file}")
        except Exception:
            logger.exception("Failed to save per-model result summary")

        # Update aggregated results file
        try:
            agg_file = os.path.join(all_dir, "aggregated_results.json")
            if os.path.exists(agg_file):
                with open(agg_file, 'r') as af:
                    aggregated = json.load(af)
            else:
                aggregated = {}

            aggregated[f"{exp_name}{bit_suffix}"] = {
                'best_metric': best_metric,
                'saved_model': f"{exp_name}{bit_suffix}_best.pth",
                'saved_at': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(agg_file, 'w') as af:
                json.dump(aggregated, af, indent=2)
            logger.info(f"Updated aggregated results at {agg_file}")
        except Exception:
            logger.exception("Failed to update aggregated results")

    def save_checkpoint(self, state, exp_name, filename, base_save_dir=None):
        """Saves a model checkpoint."""
        save_base = base_save_dir if base_save_dir else "save"
        save_dir = os.path.join(save_base, exp_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        torch.save(state, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
    
    def save_all_results(self):
        """Saves a summary of all experiments after all are completed."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join('save', 'TrainAll')
        os.makedirs(save_dir, exist_ok=True)
        results_path = os.path.join(save_dir, f"summary_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump({
                'summary': self.all_results,
                'config': self.config
            }, f, indent=2)
        logger.info(f"All experiment results saved to {results_path}")

def main():
    config = {
        # Global/Default Hyperparameters
        'bit_list': [32], 'lr': 0.0001, 'weight_decay': 0.1,
        'alpha': 0.05, 'beta': 1.5e-05,
        'margin': 0.4, 'angular_margin': 0.1,
        'epoch': 100, 'batch_size': 64, 'preload_data': True,
        'dataset': 'pihd', 'num_workers': 12,
        'resize_size': 256, 'crop_size': 224,
        'ver_roc': 10, 'device': 'cuda:0',
        'group_size': 97,

        # GroupMamba specific parameters
        'groupmamba_tiny_params': {
            "num_classes": -1, 
            "embed_dims": [64, 128, 348, 448], 
            "depths": [3, 4, 9, 3],
            "mlp_ratios": [8, 8, 4, 4], 
            "stem_hidden_dim": 32, 
            "k_size": [3, 3, 5, 5]
        },
        # Experiment definitions
        'experiments': [
            # ResNet50
            {
                'name': 'ResNet50_BsaeLine', 'backbone': 'resnet50', 'strategy': 'traditional',
                'lr': 0.0006132182827356257, 'weight_decay': 0.026317191542302183,
                'alpha': 0.004231986802516284, 'beta': 0.0000031534139022998077
            },
            {
                'name': 'ResNet50_Enhanced', 'backbone': 'resnet50', 'strategy': 'enhanced',
                'lr': 0.0006900195283236928, 'weight_decay': 0.00042638571909434813,
                'alpha': 0.002468251840564691, 'beta': 0.0000028399129357275125,
                'angular_margin': 0.28052743580034506
            },
            
            # ViT 
            {
                'name': 'ViT_BsaeLine', 'backbone': 'vit', 'strategy': 'traditional',
                'lr': 0.00019326156234761747, 'weight_decay': 0.0047147366253532,
                'alpha': 0.004558159919267386, 'beta': 0.00008362777306500706
            },
            {
                'name': 'ViT_Enhanced', 'backbone': 'vit', 'strategy': 'enhanced',
                'lr': 0.00007389810743676917, 'weight_decay': 0.0007018810304527054,
                'alpha': 0.01278513204637169, 'beta': 0.0000026479455942463863,
                'angular_margin': 0.4789583374729169
            },
            # MambaOut
            {
                'name': 'MambaOut_Baseline', 'backbone': 'mambaout_tiny', 'strategy': 'traditional',
                'lr': 0.0004960218886876176, 'weight_decay': 0.001305755358855032,
                'alpha': 0.03787065647176286, 'beta': 0.0000022992392997488816
            },
            {
            'name': 'MambaOut_Enhanced', 'backbone': 'mambaout_tiny', 'strategy': 'enhanced',
            'lr': 0.00032205475549689573, 'weight_decay': 0.0007329804283540057,
            'alpha': 0.003705005866833663, 'beta': 0.00005743859846316311,
            'angular_margin': 0.46779082730806576
            },           
            {
                'name': 'GroupMamba_Baseline', 'backbone': 'groupmamba', 'strategy': 'traditional',
                'lr': 0.00027211388342281324, 'weight_decay': 0.045973516941508884,
                'alpha': 0.0025718497258628067, 'beta': 0.00000348221411268456
            },
            {
                'name': 'GroupMamba_Enhanced', 'backbone': 'groupmamba', 'strategy': 'enhanced',
                'lr': 0.00011475317337994807, 'weight_decay': 0.0002821610761685168,
                'alpha': 0.010381936765669298, 'beta': 0.00001613987168109618,
                'angular_margin': 0.31540664834035514
            },
            
            # ConvNeXtV2 Experiments
            {
                'name': 'ConvNeXtV2_Baseline', 'backbone': 'convnextv2', 'strategy': 'traditional',
                'lr': 0.00014982852411569812, 'weight_decay': 0.0016505075824962178,
                'alpha': 0.0025998002918414475, 'beta': 0.000008708554759477607
            },
            {
                'name': 'ConvNeXtV2_Enhanced', 'backbone': 'convnextv2', 'strategy': 'enhanced',
                'lr': 0.0003282364270850507, 'weight_decay': 0.08784182492745979,
                'alpha': 0.0010102417972613217, 'beta': 0.0000011183409443437897,
                'angular_margin': 0.23974127774856374
            },
            
            # Swin Transformer Tiny Experiments
            {
                'name': 'SwinTiny_Baseline', 'backbone': 'swin_tiny', 'strategy': 'traditional',
                'lr': 0.00008614622887324078, 'weight_decay': 0.004703168354174607,
                'alpha': 0.004024654147809965, 'beta': 0.000021825869369064363
            },
            {
                'name': 'SwinTiny_Enhanced', 'backbone': 'swin_tiny', 'strategy': 'enhanced',
                'lr': 0.0002015097946566373, 'weight_decay': 0.0012613336614463757,
                'alpha': 0.01612185530008903, 'beta': 0.000012873613688695439,
                'angular_margin': 0.3877570235500098
            }
        ],
    }
    
    config = config_dataset(config)
    train_loader, test_loader, db_loader, _, _, _ = get_data(config)
    
    trainer = ExperimentTrainer(config)
    trainer.run_all_experiments(train_loader, test_loader, db_loader)

if __name__ == "__main__":
    main()