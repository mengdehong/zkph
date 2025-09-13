# new_generic_model.py (或者放在训练脚本顶部)
import torch,os
import torch.nn as nn
import torch.nn.functional as F
import timm
import importlib
from loguru import logger
from model.MambaHash import GroupMamba 

def create_hashing_model(backbone_name, hash_head_type, model_config):
    """
    顶层模型构建函数：组装 骨干网络 + 哈希头。
    
    Args:
        backbone_name (str): 骨干网络的名称 (e.g., 'resnet50', 'vit').
        hash_head_type (str): 哈希头的类型 ('traditional' or 'enhanced').
        model_config (dict): 包含模型参数的配置字典.
        
    Returns:
        nn.Module: 一个完整的、可用于训练的哈希模型.
    """
    logger.info(f"Assembling full hashing model: Backbone='{backbone_name}', HashHead='{hash_head_type}'")
    
    # 1. 使用已有的工厂函数构建骨干网络
    backbone_model, feature_dim = build_backbone_model(backbone_name, model_config)
    
    # 2. 检查特征维度是否成功推断
    if feature_dim is None:
        raise RuntimeError(f"Could not determine the output feature dimension for backbone '{backbone_name}'.")
    
    # 3. 使用通用包装器将骨干网络和哈希头组合起来
    full_model = GenericHashingModel(
        backbone_model=backbone_model,
        input_feature_dim=feature_dim,
        hash_bits=model_config['bit_list'][0],
        hash_head_type=hash_head_type
    )
    
    return full_model
class GenericHashingModel(nn.Module):
    """
    一个通用的哈希模型包装器。
    它接收一个骨干网络，并在其上附加指定的哈希头。
    """
    def __init__(self, backbone_model, input_feature_dim, hash_bits, hash_head_type='enhanced'):
        super().__init__()
        self.backbone_model = backbone_model
        self.hash_bits = hash_bits
        self.hash_head_type = hash_head_type
        
        # 根据哈希头类型创建哈希头
        if self.hash_head_type == 'traditional':
            self.hash_head = self._build_traditional_hash_head(input_feature_dim)
        elif self.hash_head_type == 'enhanced':
            self.hash_head = self._build_enhanced_hash_head(input_feature_dim)
        elif self.hash_head_type == 'ablation_bn':
            # 消融实验：BN头但使用传统损失
            self.hash_head = self._build_enhanced_hash_head(input_feature_dim)
        elif self.hash_head_type == 'ablation_angular_no_abl':
            # 消融实验：BN头 + 角度损失（不含ABL）
            self.hash_head = self._build_enhanced_hash_head(input_feature_dim)
        else:
            raise ValueError(f"Unknown hash_head_type: '{hash_head_type}'")

    def _build_traditional_hash_head(self, input_dim):
        """创建传统哈希头 (Linear -> Tanh)"""
        logger.info(f"Building Traditional Hash Head: input_dim={input_dim}, hash_bits={self.hash_bits}")
        return nn.Sequential(
            nn.Linear(input_dim, self.hash_bits),
            nn.Tanh()
        )

    def _build_enhanced_hash_head(self, input_dim):
        """创建增强哈希头 (Linear -> BN)"""
        logger.info(f"Building Enhanced Hash Head: input_dim={input_dim}, hash_bits={self.hash_bits}")
        return nn.Sequential(
            nn.Linear(input_dim, self.hash_bits),
            nn.BatchNorm1d(self.hash_bits)
        )

    def forward(self, x):
        # 1. 通过骨干网络提取特征
        #    注意：骨干网络需要返回一个 (B, D) 的特征向量
        extracted_features = self.backbone_model(x)
        
        # 2. 通过哈希头生成哈希码
        if self.hash_head_type == 'traditional':
            hash_codes = self.hash_head(extracted_features)
        elif self.hash_head_type in ['enhanced', 'ablation_bn', 'ablation_angular_no_abl']:
            hash_features = self.hash_head(extracted_features)
            hash_codes = F.normalize(hash_features, p=2, dim=1)
        
        return hash_codes

def build_backbone_model(backbone_name, model_config):
    """
    骨干网络工厂函数。
    返回 (骨干网络模型, 输出特征维度)。
    """
    logger.info(f"Building backbone model: {backbone_name}")
    
    if backbone_name == 'groupmamba':
        return _build_groupmamba_backbone(model_config)
    elif backbone_name == 'vit':
        return _build_vit_small_backbone()
    elif backbone_name in ['mambaout_tiny', 'mambaout']:
        return _build_mambaout_backbone()
    elif backbone_name == 'resnet50':
        return _build_resnet50_backbone()
    elif backbone_name == 'convnextv2':
        return _build_convnextv2_backbone()
    elif backbone_name == 'swin_tiny':
        return _build_swin_tiny_backbone()
    else:
        raise ValueError(f"Unsupported backbone model: '{backbone_name}'")

def _build_groupmamba_backbone(model_config):
    """构建 GroupMamba 骨干网络"""
    logger.info("Building GroupMamba backbone with pretrained weights")
    
    # 如果配置中缺失 groupmamba_tiny_params，使用合理的默认值并给出警告
    default_params = {
        "num_classes": -1, "embed_dims": [64, 128, 348, 448], "depths": [3, 4, 9, 3],
        "mlp_ratios": [8, 8, 4, 4], "stem_hidden_dim": 32, "k_size": [3, 3, 5, 5]
    }
    if 'groupmamba_tiny_params' not in model_config:
        logger.warning("model_config does not contain 'groupmamba_tiny_params'. Using default GroupMamba params.")
    params = model_config.get('groupmamba_tiny_params', default_params)

    # 允许用户提供 embed_dims，但确保存在并可用
    if 'embed_dims' not in params or not isinstance(params['embed_dims'], (list, tuple)) or len(params['embed_dims']) == 0:
        logger.warning("Invalid or missing 'embed_dims' in groupmamba_tiny_params. Falling back to default embed_dims.")
        params['embed_dims'] = default_params['embed_dims']

    model = GroupMamba(
        final_output_dim=params['embed_dims'][-1],
        **params
    )
    output_feature_dim = params['embed_dims'][-1]
    
    # 加载 GroupMamba 的预训练权重（若存在）
    pretrained_checkpoint_path = "checkpoint/groupmamba_tiny_ema.pth"
    _load_pretrained_weights(model, pretrained_checkpoint_path, "GroupMamba")
    
    return model, output_feature_dim

def _build_vit_small_backbone():
    """构建 ViT Small 骨干网络"""
    logger.info("Building ViT Small backbone with custom pretrained weights")
    
    # 创建 ViT Small 模型，不使用 timm 默认预训练权重
    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=0)
    output_feature_dim = model.num_features
    
    # 加载自定义预训练权重
    pretrained_checkpoint_path = "checkpoint/vit_small_patch16_224.pth"
    _load_pretrained_weights(model, pretrained_checkpoint_path, "ViT Small")
    
    return model, output_feature_dim

def _build_mambaout_backbone():
    """构建 MambaOut (门控CNN) 骨干网络（兼容 model/MambaOut.py 注册的 mambaout_tiny）"""
    logger.info("Building MambaOut (Gated CNN) backbone with pretrained weights")

    # 确保 model.MambaOut 被导入，从而让 timm 注册器中存在 'mambaout_tiny'
    try:
        importlib.import_module('model.MambaOut')
    except Exception as e:
        logger.error(f"Failed to import model.MambaOut: {e}")
        raise

    # 通过 timm 创建已注册的模型（num_classes=0 移除分类头）
    model = timm.create_model('mambaout_tiny', pretrained=False, num_classes=0)

    # 尝试推断输出特征维度
    output_feature_dim = getattr(model, 'num_features', None)
    if output_feature_dim is None:
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                # 优先使用 forward_features（MambaOut 中存在），否则尝试直接前向
                if hasattr(model, 'forward_features'):
                    feat = model.forward_features(dummy)
                else:
                    feat = model(dummy)
                output_feature_dim = feat.shape[1]
        except Exception as e:
            logger.warning(f"Could not infer output feature dim for MambaOut: {e}")
            output_feature_dim = None

    # 加载预训练权重（若存在）
    pretrained_checkpoint_path = "checkpoint/mambaout_tiny.pth"
    _load_pretrained_weights(model, pretrained_checkpoint_path, "MambaOut")

    return model, output_feature_dim

def _build_resnet50_backbone():
    """构建 ResNet-50 骨干网络"""
    logger.info("Building ResNet-50 backbone with LOCAL weights (no online download)")

    # 禁用在线下载，纯本地构建
    model = timm.create_model('resnet50', pretrained=False, num_classes=0)
    output_feature_dim = model.num_features

    # 尝试加载本地 checkpoint/resnet50.pth
    pretrained_checkpoint_path = "checkpoint/resnet50.pth"
    _load_pretrained_weights(model, pretrained_checkpoint_path, "ResNet-50")

    logger.info(f"ResNet-50 loaded (offline) with feature dimension: {output_feature_dim}")
    return model, output_feature_dim

def _build_convnextv2_backbone():
    """构建 ConvNeXtV2 骨干网络"""
    logger.info("Building ConvNeXtV2 backbone with pretrained weights")
    
    # 导入本地的 ConvNeXtV2 模型
    try:
        from model.ConvNeXtV2 import convnextv2_tiny
        model = convnextv2_tiny(num_classes=0)  # 不带分类头
        output_feature_dim = model.num_features
        
        # 加载自定义预训练权重
        pretrained_checkpoint_path = "checkpoint/convnextv2.pt"
        _load_pretrained_weights(model, pretrained_checkpoint_path, "ConvNeXtV2")
        
        logger.info(f"ConvNeXtV2 loaded successfully with feature dimension: {output_feature_dim}")
        return model, output_feature_dim
        
    except Exception as e:
        logger.error(f"Failed to build ConvNeXtV2 backbone: {e}")
        raise e

def _build_swin_tiny_backbone():
    """构建 Swin Transformer Tiny 骨干网络"""
    logger.info("Building Swin Transformer Tiny backbone with pretrained weights")
    
    # 导入本地的 Swin Transformer 模型
    try:
        from model.SwinTransformer import swin_tiny_patch4_window7_224
        model = swin_tiny_patch4_window7_224(num_classes=0)  # 不带分类头
        output_feature_dim = model.num_features
        
        # 加载自定义预训练权重
        pretrained_checkpoint_path = "checkpoint/swin_transformer_tiny.pth"
        _load_pretrained_weights(model, pretrained_checkpoint_path, "Swin Transformer Tiny")
        
        logger.info(f"Swin Transformer Tiny loaded successfully with feature dimension: {output_feature_dim}")
        return model, output_feature_dim
        
    except Exception as e:
        logger.error(f"Failed to build Swin Transformer Tiny backbone: {e}")
        raise e

def _load_pretrained_weights(model, checkpoint_path, model_name):
    """通用的预训练权重加载函数"""
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading pretrained weights for {model_name} from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 处理不同的权重格式
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            model_state_dict = model.state_dict()
            # 过滤掉不匹配的键 (例如分类头或哈希头)
            compatible_weights = {
                k: v for k, v in state_dict.items() 
                if k in model_state_dict and model_state_dict[k].shape == v.shape
            }
            
            if compatible_weights:
                model_state_dict.update(compatible_weights)
                model.load_state_dict(model_state_dict, strict=False)
                logger.info(f"Successfully loaded {len(compatible_weights)} compatible weight tensors for {model_name}")
            else:
                logger.warning(f"No compatible weights found for {model_name}")
                
        except Exception as e:
            logger.error(f"Error loading pretrained weights for {model_name}: {e}")
    else:
        logger.warning(f"Pretrained weights for {model_name} not found at {checkpoint_path}")

# 别名函数，保持向后兼容
def create_backbone(backbone_name, config):
    """向后兼容的别名函数"""
    return build_backbone_model(backbone_name, config)