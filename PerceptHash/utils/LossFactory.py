from loguru import logger

from utils.HashLoss import TripletLossForHashing, AngularTripletLoss, AngularTripletLossNoABL


def create_loss_function(strategy_name, config):
    """
    Loss function factory.
    Creates and returns the appropriate loss function instance based on the strategy name.
    """
    logger.info(f"Creating loss function for strategy: '{strategy_name}'")
    
    if strategy_name == 'traditional':
        return TripletLossForHashing(
            alpha=config['alpha'],
            beta=config['beta'],
            margin=config.get('margin', 0.5)
        )
        
    elif strategy_name == 'enhanced':
        return AngularTripletLoss(
            alpha=config['alpha'],
            beta=config['beta'],
            hash_bit=config['bit_list'][0],
            margin=config.get('angular_margin', 0.1)
        )
    
    elif strategy_name == 'ablation_bn':
        return TripletLossForHashing(
            alpha=config['alpha'],
            beta=config['beta'],
            margin=config.get('margin', 0.5)
        )
    
    elif strategy_name == 'ablation_angular_no_abl':
        return AngularTripletLossNoABL(
            alpha=config['alpha'],
            beta=config['beta'],
            hash_bit=config['bit_list'][0],
            margin=config.get('angular_margin', 0.1)
        )
        
    else:
        raise ValueError(f"Unknown training strategy for loss creation: '{strategy_name}'")