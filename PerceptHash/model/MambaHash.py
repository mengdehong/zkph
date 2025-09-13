import torch
import torch.nn as nn
from functools import partial
import torch.fft
import pywt

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from loguru import logger
import math
from torch.nn import functional as F
from einops import rearrange

try:
    from .ss2d import SS2D
    from .csms6s import CrossScan_1, CrossScan_2, CrossScan_3, CrossScan_4
    from .csms6s import CrossMerge_1, CrossMerge_2, CrossMerge_3, CrossMerge_4
except:
    from ss2d import SS2D
    from csms6s import CrossScan_1, CrossScan_2, CrossScan_3, CrossScan_4
    from csms6s import CrossMerge_1, CrossMerge_2, CrossMerge_3, CrossMerge_4


class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        

        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        
        x = self.fc2(x)

        return x



class GroupMambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, k_size, d_state=1, d_conv=3, expand=1, reduction=16):
        super().__init__()
        
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)

        self.mamba_g1 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )
        self.mamba_g2 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )
        self.mamba_g3 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )
        self.mamba_g4 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )

        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

        self.act = nn.ReLU(inplace=True)

        # num_channels_reduced = input_dim // reduction
        # self.fc_in = nn.Linear(input_dim, num_channels_reduced, bias=True)
        # self.fc_out = nn.Linear(num_channels_reduced, output_dim, bias=True)

    def forward(self, x, H, W):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, N, C = x.shape
        x = self.norm(x)

        # Channel Affinity
        z_avg = x.permute(0, 2, 1).mean(dim=2)
        
        # ciam
        y = self.conv(z_avg.unsqueeze(-1).transpose(-1, -2)).transpose(-1, -2).squeeze(-1)
        fc_out_2 = self.sigmoid(self.fc(z_avg)+y)     


        x = rearrange(x, 'b (h w) c -> b h w c', b=B, h=H, w=W, c=C)
        
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=-1)
        # Four scans applied to 4 different directions, each is applied for N/4 channels
        x_mamba1 = self.mamba_g1(x1, CrossScan=CrossScan_1, CrossMerge=CrossMerge_1)
        x_mamba2 = self.mamba_g2(x2, CrossScan=CrossScan_2, CrossMerge=CrossMerge_2)
        x_mamba3 = self.mamba_g3(x3, CrossScan=CrossScan_3, CrossMerge=CrossMerge_3)
        x_mamba4 = self.mamba_g4(x4, CrossScan=CrossScan_4, CrossMerge=CrossMerge_4)

        # Combine all feature maps
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=-1) * self.skip_scale * x

        x_mamba = rearrange(x_mamba, 'b h w c -> b (h w) c', b=B, h=H, w=W, c=C)

        # Channel Modulation
        x_mamba = x_mamba * fc_out_2.unsqueeze(1)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)

        return x_mamba


class Block_mamba(nn.Module):
    def __init__(self, 
        dim, 
        mlp_ratio,
        drop_path=0., 
        norm_layer=nn.LayerNorm,
        ken_size=3
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)

        self.attn = GroupMambaLayer(dim, dim,ken_size)
        self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(x, H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class DownSamples(nn.Module):
    def __init__(self, in_channels, out_channels,hash_bit):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
       
        
        return x, H, W

class Stem(nn.Module):
    def __init__(self, in_channels, stem_hidden_dim, out_channels):
        super().__init__()
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2,
                      padding=3, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_dim,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.norm = nn.LayerNorm(out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Enhanced_feature(nn.Module):
    def __init__(self, dim, mlp_ratio=2, drop_path=0.):
        super().__init__()
        
        self.dwconv2 = ConvBN(mlp_ratio * dim, mlp_ratio * dim, 3, 1, (3 - 1) // 2, groups=dim, with_bn=False)
        self.dwconv3 = ConvBN(mlp_ratio * dim, mlp_ratio * dim, 5, 1, (5 - 1) // 2, groups=dim, with_bn=False)
        self.dwconv1 = ConvBN(mlp_ratio * dim, mlp_ratio * dim, 1, 1, (1 - 1) // 2, groups=dim, with_bn=False)


        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.act = nn.ReLU()
        self.f2 = ConvBN(mlp_ratio *dim, dim, 1, with_bn=False)

    def forward(self, x):
        x = self.f1(x)
        x = self.dwconv1(x)+self.dwconv2(x)+self.dwconv3(x)
        x = self.act(x)
        x = self.f2(x)
        return x



class GroupMamba(nn.Module):
    def __init__(self, 
        in_chans=3, 
        hash_bit=16,
        num_classes=1000, 
        stem_hidden_dim = 32,
        embed_dims=[64, 128, 348, 448],
        mlp_ratios=[8, 8, 4, 4], 
        drop_path_rate=0., 
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        num_stages=4,
        k_size = [3,3,5,5],
        distillation=False,
        hash_head_type='enhanced', # <-- 【新增参数】 'traditional' 或 'enhanced'
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.final_output_dim = kwargs.get('final_output_dim')

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(num_stages):
            if i == 0:
                patch_embed = Stem(in_chans, stem_hidden_dim, embed_dims[i])
            else:
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i],hash_bit)
            block = nn.ModuleList([Block_mamba(
                    dim = embed_dims[i],
                    mlp_ratio = mlp_ratios[i],
                    drop_path=dpr[cur + j],
                    norm_layer=norm_layer,
                    ken_size=k_size[i])
                for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

     
        self.code_length = hash_bit if hash_bit is not None else 16 # Default for feature extractor mode
        self._norm_layer = nn.BatchNorm2d
       
        self.feature_module = Enhanced_feature(embed_dims[-1],mlp_ratio=2**(self.code_length//16))
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.hash_head_type = hash_head_type
        if self.final_output_dim is None:
            if self.hash_head_type == 'traditional':
                logger.info("Initializing with Traditional Hash Head (Linear -> Tanh)")
                self.hash_head = self._create_traditional_hash_head(embed_dims[-1])
            elif self.hash_head_type == 'enhanced':
                logger.info("Initializing with Enhanced Hash Head (Linear -> BN -> L2 Norm)")
                self.hash_head = self._create_enhanced_hash_head(embed_dims[-1])
            else:
                raise ValueError(f"Unknown hash_head_type: '{self.hash_head_type}'. Choose 'traditional' or 'enhanced'.")
        
        self.apply(self._init_weights)
    def _create_traditional_hash_head(self, input_dim):
        """函数1：创建传统哈希头"""
        return nn.Sequential(
            nn.Linear(input_dim, self.code_length),
            nn.Tanh()
        )

    def _create_enhanced_hash_head(self, input_dim):
        """函数2：创建增强哈希头 (不含L2 Norm，因其在forward中应用)"""
        return nn.Sequential(
            nn.Linear(input_dim, self.code_length),
            nn.BatchNorm1d(self.code_length) 
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward_features(self, x):
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)

            norm = getattr(self, f"norm{i + 1}")
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        return x

    def forward(self, x):
        # 1. 提取深度特征
        features = self.forward_features(x)
        features = self.feature_module(features)
        pooled_features = torch.flatten(self.avg(features), start_dim=1)
        
        # 如果是作为纯特征提取器，则直接返回
        if hasattr(self, 'final_output_dim') and self.final_output_dim is not None:
            return pooled_features

        # 2. --- 【核心修改】根据哈希头类型应用不同的前向传播逻辑 ---
        if self.hash_head_type == 'traditional':
            # 传统哈希头: 直接通过 Linear -> Tanh
            hash_code = self.hash_head(pooled_features)
        
        elif self.hash_head_type == 'enhanced':
            # 增强哈希头: 通过 Linear -> BN, 然后进行 L2 归一化
            out = self.hash_head(pooled_features)
            hash_code = F.normalize(out, p=2, dim=1)
        
        else:
            # 此处代码理论上不会执行，因为构造函数中已检查
            raise RuntimeError(f"Invalid hash_head_type '{self.hash_head_type}' during forward pass.")
            
        return hash_code
    
    def forward_with_group_embeds(self, x):
        """
        返回 (hash_embedding, group_embeds)
        - hash_embedding: 和原 forward 一致，L2 归一化后的连续向量 [B, K]
        - group_embeds: 分成 4 个通道组后的每组全局池化 embedding -> [B, 4, Dg]
        
        设计说明：
        - 我们在 feature_module 的输出基础上，按通道将 feature map 划分为4组（与 GroupMambaLayer 的组划分语义对齐）
        - 每一组做 AdaptiveAvgPool2d(1) -> flatten 得到向量
        - 该 group_embeds 用于计算组间一致性 loss（clean vs adv）
        """
        # forward_features -> feature map [B, C, H, W]
        feat_map = self.forward_features(x)  # B, C, H, W
        feat_map = self.feature_module(feat_map)  # B, C, H, W (same as in forward)
        B, C, H, W = feat_map.shape

        # compute hash as in forward
        pooled = torch.flatten(self.avg(feat_map), start_dim=1, end_dim=3)  # [B, C]
        if hasattr(self, 'final_output_dim') and self.final_output_dim is not None:
            out = pooled
            # If final_output_dim mode, we don't apply hash_head
            hash_vec = out
        else:
            out = self.hash_head(pooled)  # [B, K]
            hash_vec = F.normalize(out, p=2, dim=1)

        # compute group embeddings by splitting channels into 4 groups
        # if C not divisible by 4, last group gets the remainder
        g = 4
        per_group = C // g
        group_embeds = []
        for i in range(g):
            start = i * per_group
            end = (i + 1) * per_group if i < g - 1 else C
            gm = feat_map[:, start:end, :, :]  # B, Cg, H, W
            pooled_g = F.adaptive_avg_pool2d(gm, 1).view(B, -1)  # B, Cg
            group_embeds.append(pooled_g)
        # stack -> [B, 4, Cg_var]
        # to have uniform dims, pad smaller groups to max dim (rare); for simplicity we keep ragged second dim by projecting each to same dim
        # project each pooled group to a common dim Dg
        Dg = min(128, per_group)  # or tune
        # create small projection layers on-the-fly if not exist
        if not hasattr(self, "_group_proj"):
            # create learnable projection for groups
            self._group_proj = nn.ModuleList([nn.Linear(group_embeds[i].shape[1], Dg).to(group_embeds[i].device) for i in range(g)])
            # init
            for p in self._group_proj:
                # trunc_normal_(p.weight, std=.02)
                torch.nn.init.normal_(p.weight, std=0.02)

                if p.bias is not None:
                    nn.init.constant_(p.bias, 0.0)
        # apply projections
        group_vecs = []
        for i in range(g):
            v = self._group_proj[i](group_embeds[i])  # B, Dg
            group_vecs.append(v)
        group_vecs = torch.stack(group_vecs, dim=1)  # B, 4, Dg

        return hash_vec, group_vecs

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x
