"""
MambaOut 图像分类模型。
部分实现改自：
timm (https://github.com/rwightman/pytorch-image-models),
MetaFormer (https://github.com/sail-sg/metaformer),
InceptionNeXt (https://github.com/sail-sg/inceptionnext)
"""
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'mambaout_tiny': _cfg(
        url=''),  # 离线版：不包含在线 url
}


class StemLayer(nn.Module):
    r""" 代码改自 InternImage """

    def __init__(self,
                 in_channels=3,
                 out_channels=96,
                 act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = norm_layer(out_channels // 2)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(out_channels // 2,
                               out_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        return x


class DownsampleLayer(nn.Module):
    r""" 代码改自 InternImage """
    def __init__(self, in_channels=96, out_channels=198, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class MlpHead(nn.Module):
    """ MLP 分类头 """
    def __init__(self, dim, num_classes=1000, act_layer=nn.GELU, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


class GatedCNNBlock(nn.Module):
    r""" Gated CNN 块实现
    参数说明：
        conv_ratio: 控制参与深度可分离卷积的通道比例
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm,eps=1e-6), 
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x  # [B, H, W, C] (批, 高, 宽, 通道)
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2)  # -> [B, C, H, W]
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1)  # -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        return x + shortcut

r"""
下采样说明（简要）：
第一阶段的 stem = 两层 k3 s2 p1 的卷积
之后每个阶段前的下采样 = 一层 k3 s2 p1 的卷积
DOWNSAMPLE_LAYERS_FOUR_STAGES 格式： [StemLayer, DownsampleLayer, DownsampleLayer, DownsampleLayer]
"""
DOWNSAMPLE_LAYERS_FOUR_STAGES = [StemLayer] + [DownsampleLayer]*3


class MambaOut(nn.Module):
    r""" MetaFormer 风格骨干（简要中文说明）
    参数（简要）：
        in_chans: 输入通道数
        num_classes: 分类数
        depths: 每个 stage 的块数
        dims: 每个 stage 的通道数
        drop_path_rate: 随机深度比率
        head_dropout: 分类头 dropout
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 576],
                 downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,
                 conv_ratio=1.0,
                 kernel_size=7,
                 drop_path_rate=0.,
                 output_norm=partial(nn.LayerNorm, eps=1e-6), 
                 head_fn=MlpHead,
                 head_dropout=0.0, 
                 **kwargs,
                 ):
        super().__init__()
        self.num_classes = num_classes

        if not isinstance(depths, (list, tuple)):
            depths = [depths] # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i+1]) for i in range(num_stage)]
        )

        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[GatedCNNBlock(dim=dims[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                kernel_size=kernel_size,
                conv_ratio=conv_ratio,
                drop_path=dp_rates[cur + j],
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = output_norm(dims[-1])

        if head_dropout > 0.0:
            self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
        else:
            self.head = head_fn(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward_features(self, x):
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([1, 2])) # (B, H, W, C) -> (B, C)

    def forward(self, x):
        x = self.forward_features(x)
        if self.num_classes == 0 or self.num_classes is None:
            return x
        x = self.head(x)
        return x



###############################################################################
# 一组 MambaOut 模型（仅保留 tiny）
@register_model
def mambaout_tiny(pretrained=False, **kwargs):
    model = MambaOut(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 576],
        **kwargs)
    model.default_cfg = default_cfgs['mambaout_tiny']

    # 离线加载逻辑：
    # - pretrained == False: 不加载权重
    # - pretrained 为字符串：被视为本地权重文件路径，使用 torch.load 加载
    # - pretrained == True: 抛出错误，提示必须提供本地路径以便离线加载
    if pretrained:
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location="cpu")
            # 兼容保存时可能带有 'model' 或 'state_dict' 的情况
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            model.load_state_dict(state_dict)
        else:
            raise ValueError("Offline mode: please provide a local checkpoint path string for `pretrained`, or set pretrained=False.")
    return model