import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat

from einops import rearrange
from einops.layers.torch import Rearrange

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
import json
from scipy.ndimage import rotate
import os
import time
import math
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from data_t import StellarSpectrumDataset
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from odconv import ODConv2d


def convert_log_params(df, log_columns=['teff', 'teff_err']):
    converted_df = df.copy()
    for col in converted_df.columns:
        if any(log_col in col for log_col in log_columns) and col.endswith('_pred'):
            try:
                converted_df[col] = 2 ** converted_df[col]
                converted_df[col] = converted_df[col].replace([np.inf, -np.inf], np.nan)
            except Exception as e:
                print(f"转换列 {col} 时出错: {str(e)}")
    return converted_df

def conv_3x3_bn(inp, oup, image_size, downsample=False, use_odconv=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4, use_odconv=False):
        super().__init__()
        self.downsample = downsample
        stride = 1

        hidden_dim = int(inp * expansion)
        conv_layer = ODConv2d if use_odconv else nn.Conv2d

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = conv_layer(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                conv_layer(hidden_dim, hidden_dim, 3, stride,
                           1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                conv_layer(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                conv_layer(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                conv_layer(hidden_dim, hidden_dim, 3, 1, 1,
                           groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                conv_layer(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            orig_shape = x.shape
            pooled = self.pool(x)
            proj = self.proj(pooled)
            conv_out = self.conv(pooled)

            # 尺寸验证
            if proj.shape != conv_out.shape:
                raise RuntimeError(
                    f"尺寸不匹配！输入尺寸：{orig_shape}\n"
                    f"池化后：{pooled.shape}\n"
                    f"投影后：{proj.shape}\n"
                    f"卷积后：{conv_out.shape}"
                )

            return proj + conv_out
        else:
            return x + self.conv(x)
    @staticmethod
    def validate_size(in_size, expected_out):
        return (in_size[0]+2 * 1-3)//2 + 1 == expected_out[0]


class Attention(nn.Module):
    def __init__(self, inp, oup, base_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        self.in_features = inp
        self.out_features = oup
        self.base_ih, self.base_iw = base_size
        self.register_buffer('current_ih', torch.tensor(base_size[0]))
        self.register_buffer('current_iw', torch.tensor(base_size[1]))

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if (heads > 1 or dim_head != inp) else nn.Identity()

        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.base_ih - 1) * (2 * self.base_iw - 1), heads))
        self.register_buffer('max_index', torch.tensor((2 * self.base_ih - 1) * (2 * self.base_iw - 1) - 1))
        self._generate_relative_index()

    def forward(self, x):
        # 维度校验
        assert x.size(-1) == self.in_features, \
            f"Attention维度不匹配! 输入维度:{x.size(-1)} 期望:{self.in_features}"

        b, n, _ = x.shape
        current_size = int(math.sqrt(n))
        assert current_size ** 2 == n, f"非法输入形状: {x.shape}无法转换为正方形"
        self.current_ih.fill_(current_size)
        self.current_iw.fill_(current_size)
        self._generate_relative_index()

        safe_index = torch.clamp(self.relative_index, 0, self.max_index)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        safe_index = safe_index.view(-1, 1).expand(-1, self.heads)

        # 获取相对位置偏置并调整形状
        relative_bias = self.relative_bias_table.gather(0, safe_index)
        relative_bias = relative_bias.view(current_size * current_size,
                                           current_size * current_size,
                                           self.heads).permute(2, 0, 1).unsqueeze(0)

        dots = dots + relative_bias
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

    def _create_relative_index(self):
        """生成并注册相对位置索引"""
        coords = torch.stack(torch.meshgrid(
            [torch.arange(self.ih), torch.arange(self.iw)], indexing='ij'
        )).flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords += torch.tensor([self.ih - 1, self.iw - 1])[:, None, None]
        relative_coords[0] *= 2 * self.iw - 1
        self.register_buffer("relative_index", relative_coords.sum(0).flatten())

    def _generate_relative_index(self):
        """动态生成相对位置索引"""
        device = self.relative_bias_table.device
        h, w = self.current_ih.item(), self.current_iw.item()
        coords = torch.stack(torch.meshgrid(
            [torch.arange(h, device=device),
             torch.arange(w, device=device)],
            indexing='ij'
        )).flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[0] += h - 1
        relative_coords[1] += w - 1
        relative_coords[0] *= 2 * w - 1
        self.register_buffer("relative_index", relative_coords.sum(0).flatten())

    def _resize_relative_table(self, new_size):
        """动态调整相对位置编码表（修复版）"""
        new_max_rel = 2 * new_size - 1
        old_max_rel = 2 * int(math.sqrt(self.relative_bias_table.size(0))) - 1  # 关键修复：正确计算旧尺寸

        new_table = nn.Parameter(torch.zeros((new_max_rel) ** 2, self.heads))

        def map_coord(old_coord):
            i, j = old_coord // old_max_rel, old_coord % old_max_rel
            return (i - new_size + 1) * new_max_rel + (j - new_size + 1)

        for x in range(new_max_rel):
            for y in range(new_max_rel):
                old_idx = (x + old_max_rel // 2 - new_size + 1) * old_max_rel + (y + old_max_rel // 2 - new_size + 1)
                if 0 <= old_idx < self.relative_bias_table.size(0):
                    new_table.data[x * new_max_rel + y] = self.relative_bias_table.data[old_idx]

        self.relative_bias_table = new_table
        self.max_index = torch.tensor(new_table.size(0) - 1)

        self.ih = self.iw = new_size
        self._generate_relative_index()


class Transformer(nn.Module):
    def __init__(self, inp, oup, base_size, heads=8, dim_head=32, downsample=False, use_odconv=False, dropout=0.):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.in_channels = inp
        self.out_channels = oup

        hidden_dim = int(oup * 4)
        self.ih, self.iw = base_size
        self.dynamic_size = None
        self.base_size = base_size

        self.downsample = downsample
        self.register_buffer('current_size', torch.tensor(base_size))

        if downsample:
            self.size_adjust = nn.Sequential(
                nn.Conv2d(inp, oup, 1) if inp != oup else nn.Identity(),
                nn.AvgPool2d(2, 2)
            )
        else:
            self.size_adjust = nn.Conv2d(inp, oup, 1) if inp != oup else nn.Identity()
        self._record_channels()

        self.attn_module = Attention(
            oup, oup,
            base_size=base_size,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )

        self.attn = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            PreNorm(inp, self.attn_module, nn.LayerNorm),
            Rearrange('b (h w) c -> b c h w', h=self.ih, w=self.iw)  # 使用记录的尺寸
        )
        self.attn_norm = PreNorm(oup, self.attn_module, nn.LayerNorm)  # 关键修改：inp -> oup
        self.ff_module = FeedForward(oup, hidden_dim, dropout)
        self.ff_norm = PreNorm(oup, self.ff_module, nn.LayerNorm)
        self.ff = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            PreNorm(oup, FeedForward(oup, hidden_dim, dropout), nn.LayerNorm),
            Rearrange('b (h w) c -> b c h w', h=base_size[0], w=base_size[1])
        )

    def forward(self, x):
        x_adjusted = self.size_adjust(x)
        b, c, h, w = x_adjusted.shape
        self.attn_module.current_ih.fill_(h)
        self.attn_module.current_iw.fill_(w)
        # 下采样处理（如果需要）
        if self.downsample:
            x_adjusted = F.avg_pool2d(x_adjusted, kernel_size=2, stride=2)
            h, w = h // 2, w // 2
        attn_in = rearrange(x_adjusted, 'b c h w -> b (h w) c')
        assert attn_in.size(-1) == self.attn_module.in_features, "注意力输入维度不匹配"
        attn_out = self.attn_norm(attn_in)
        attn_out = self.attn_module(attn_out)
        attn_out = rearrange(attn_out, 'b (h w) c -> b c h w', h=h, w=w)

        # 前馈分支
        ff_in = rearrange(x_adjusted, 'b c h w -> b (h w) c')
        ff_out = self.ff_norm(ff_in)
        ff_out = self.ff_module(ff_out)
        ff_out = rearrange(ff_out, 'b (h w) c -> b c h w', h=h, w=w)
        return x_adjusted + attn_out + ff_out
        # return x + self.attn(x) + self.ff(x)

    @staticmethod
    def validate_size(in_size, expected_out):
        return (in_size[0]//2, in_size[1]//2) == expected_out

    def _record_channels(self):
        """通道记录方法"""
        if isinstance(self.size_adjust, nn.Identity):
            self.adjusted_channels = self.in_channels
        elif isinstance(self.size_adjust, nn.Sequential):  # 处理下采样情况
            conv_layer = self.size_adjust[0]
            if isinstance(conv_layer, nn.Identity):
                self.adjusted_channels = self.in_channels
            else:
                self.adjusted_channels = conv_layer.out_channels
        else:  # 单个卷积层情况
            self.adjusted_channels = self.size_adjust.out_channels


class CoAtNetCore(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, block_types, od=False):
        super().__init__()
        assert len(block_types) == 4, "block_types必须有4个元素对应stage2-5"
        self.od=od
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size_tracker = {
            'current': image_size,
            'stages': [image_size]
        }
        self.stages = nn.ModuleList()
        self.current_size = image_size

        self.stages.append(
            nn.Sequential(
                conv_3x3_bn(in_channels, channels[0], self.current_size),
                *[MBConv(channels[0], channels[0], self.current_size)
                  for _ in range(num_blocks[0] - 1)]  # 可选的重复块
            )
        )
        current_size = (self.current_size[0] // 2, self.current_size[1] // 2)
        for idx in range(1, 5):
            downsample = idx < 4
            # if idx == 4:
            #     # 强制阶段4的下采样为False
            #     downsample = False
            assert 0 <= idx - 1 < len(block_types), f"block_types索引越界: idx={idx} block_types长度={len(block_types)}"
            block_char = block_types[idx - 1]
            block_cls = {'C': MBConv, 'T': Transformer}.get(block_char, None)
            assert block_cls is not None, f"无效的block类型: {block_char}"
            stage_input_size = self.current_size
            stage = self._build_stage(
                block_type=block_cls,
                base_size=stage_input_size,
                in_ch=channels[idx - 1],
                out_ch=channels[idx],
                num_blocks=num_blocks[idx],
                img_size=stage_input_size,
                stage_idx=idx,
                #有时结构为CCCC或TTTT报错时，可将下面这个注释替换downsample = True解决
                # downsample=(idx < 4),
                downsample = True
            )
            if downsample:
                self.size_tracker['current'] = (
                    self.size_tracker['current'][0] // 2,
                    self.size_tracker['current'][1] // 2
                )
            if idx < 4:
                self.current_size = (
                    stage_input_size[0] // 2,
                    stage_input_size[1] // 2
                )
            self.stages.append(stage)
            current_size = (current_size[0] // 2, current_size[1] // 2) if idx < 4 else current_size

    def _build_stage(self, block_type, base_size, in_ch, out_ch, num_blocks, img_size, stage_idx, downsample):
        """构建单个网络阶段"""
        expected_output = (img_size[0] // 2, img_size[1] // 2) if downsample else img_size
        blocks = []
        for i in range(num_blocks):
            current_inp = in_ch if i == 0 else out_ch
            if block_type == Transformer:
                block = block_type(
                    inp=in_ch if i == 0 else out_ch,
                    oup=out_ch,
                    base_size=base_size,
                    heads=8,
                    dim_head=32,
                    downsample=downsample and (i == 0),
                )
            else:
                block = block_type(
                    inp=in_ch if i == 0 else out_ch,
                    oup=out_ch,
                    image_size=img_size,
                    downsample=downsample and (i == 0),
                    use_odconv=(self.od==True and stage_idx == 3 and i >= num_blocks - 2 )
                )
            blocks.append(block)

        return nn.Sequential(*blocks)

    def forward_features(self, x):
        current_size = (x.shape[2], x.shape[3])
        for i, stage in enumerate(self.stages):
            x = stage(x)
            current_size = (x.shape[2], x.shape[3])
        return x

class FeaturePyramid(nn.Module):
    """特征金字塔模块"""
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        assert len(in_channels_list) >= 3, "需要至少3个阶段输入"
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, 256, 1) for ch in in_channels_list[-3:]
        ])
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256 * 3, 768, 3, padding=1),
            nn.BatchNorm2d(768),
            nn.GELU()
        )
        self.channel_adjust = nn.Conv2d(
            768,
            in_channels_list[-1],
            1
        )

    def forward(self, features):
        # 特征图尺寸校验
        for i, f in enumerate(features[-3:]):
            assert f.shape[2] == f.shape[3], "非方形特征图输入"
        p3, p4, p5 = [conv(f) for conv, f in zip(self.lateral_convs, features[-3:])]
        p4 = F.interpolate(p4, size=p3.shape[-2:], mode='bilinear', align_corners=False)
        p5 = F.interpolate(p5, size=p3.shape[-2:], mode='bilinear', align_corners=False)
        # 特征融合
        fused = self.fusion_conv(torch.cat([p3, p4, p5], dim=1))
        return self.channel_adjust(fused)


class RegressionHead(nn.Module):
    """通用回归头"""
    def __init__(self, in_features, out_features, hidden_dims=[512, 256]):
        super().__init__()
        layers = []
        for dim in hidden_dims:
            layers += [
                nn.Linear(in_features, dim),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                # nn.Dropout(0.5)
            ]
            in_features = dim
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(in_features, out_features)

    def forward(self, x):
        if x.dim() == 4:
            x = x.flatten(1)
        assert x.dim() == 2, f"非法输入维度: {x.shape}"
        return self.output(self.mlp(x))

class TokenExpansion(nn.Module):
    def __init__(self, num_tokens=4, dim=128):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, dim))
        nn.init.normal_(self.tokens, mean=0, std=0.02)

    def forward(self, x):
        """
        x: [B, D]
        返回: [B, num_tokens, D]
        """
        return x.unsqueeze(1) + self.tokens

class StellarParameterPredictor(CoAtNetCore):
    """参数预测模型"""
    def __init__(self, image_size, in_channels, num_blocks, channels, block_types, od=False):
        super().__init__(image_size, in_channels, num_blocks, channels, block_types, od)
        #初始化校验
        assert MBConv.validate_size((64, 64), (32, 32)), "MBConv尺寸计算错误"
        assert Transformer.validate_size((16, 16), (8, 8)), "Transformer尺寸计算错误"

        self.param_feat_adjust = nn.Linear(256, 512)
        # self.error_feat_adjust = nn.Linear(128, 128)
        # 特征金字塔
        self.fpn = FeaturePyramid(channels[2:])
        # 自适应池化
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 双预测头
        self.param_head = RegressionHead(channels[-1], 256)
        # self.error_head = RegressionHead(channels[-1], 128)
        # 注意力掩码（控制参数内部交互）
        self.register_buffer('attn_mask', self._create_attention_mask())
        # 维度投影层
        # self.error_proj = nn.Linear(256, 128)
        # 参数输出层
        self._build_output_layers()

        #交叉注意力
        self.cross_attn = nn.Sequential(
            nn.LayerNorm(256),
            nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True, dropout=0.1),
            nn.Dropout(0.1),
            nn.LayerNorm(256)
        )
        # 动态token生成
        self.token_generator = nn.ModuleDict({
            'teff': TokenExpansion(num_tokens=2, dim=256),
            'logg': TokenExpansion(num_tokens=2, dim=256),
            'feh': TokenExpansion(num_tokens=2, dim=256)
        })
        # 每个参数对应一个token
        self.token_expander = TokenExpansion(num_tokens=3, dim=128)  # 修改为3个token
        self.od=od

    def _create_attention_mask(self):
        """参数内部注意力掩码"""
        mask = torch.zeros(6, 6, dtype=torch.bool)
        # Teff内部交互
        mask[0:2, 0:2] = True
        # Logg内部交互
        mask[2:4, 2:4] = True
        # FeH内部交互
        mask[4:6, 4:6] = True
        return mask

    def _build_output_layers(self):
        """输出层"""
        # 预测值层
        self.teff_pred = nn.Linear(512, 1)
        self.logg_pred = nn.Linear(512, 1)
        self.feh_pred = nn.Linear(512, 1)
        # 误差估计层
        self.teff_err = nn.Sequential(
            nn.Linear(512, 1),
            nn.Softplus()
        )
        self.logg_err = nn.Sequential(
            nn.Linear(512, 1),
            nn.Softplus()
        )
        self.feh_err = nn.Sequential(
            nn.Linear(512, 1),
            nn.Softplus()
        )
        # self.rv_layer = nn.Linear(128, 2)

    def forward(self, x):
        # 骨干网络
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        # MFF融合
        fused = self.fpn(features)
        target_size = features[-1].shape[-2:]

        # x = features[-1] + fused_upsampled

        # # 调整上采样倍数计算
        # scale_factor = features[-1].shape[2] // fused.shape[2]
        x = features[-1] + nn.functional.interpolate(
            fused,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        # 全局特征提取
        global_feat = self.pool(x).flatten(1)
        param_feat = self.param_head(global_feat)
        # 生成参数特征对
        teff_feats = self.token_generator['teff'](param_feat)  # [B,2,256]
        logg_feats = self.token_generator['logg'](param_feat)
        feh_feats = self.token_generator['feh'](param_feat)
        # 合并所有特征对
        all_feats = torch.cat([teff_feats, logg_feats, feh_feats], dim=1)  # [B,6,256]
        # error_feat = self.token_expander(error_feat)

        # 参数内部交叉注意力
        norm_feats = self.cross_attn[0](all_feats)
        attn_out, _ = self.cross_attn[1](
            query=norm_feats,
            key=norm_feats,
            value=norm_feats,
            attn_mask=self.attn_mask
        )
        attn_feats = all_feats + self.cross_attn[2](attn_out)
        attn_feats = self.cross_attn[3](attn_feats)
        # 特征增强
        enhanced_feats = self.param_feat_adjust(attn_feats)  # [B,6,512]
        # 分离各参数特征
        teff_pred_feat = enhanced_feats[:, 0, :]
        teff_err_feat = enhanced_feats[:, 1, :]
        logg_pred_feat = enhanced_feats[:, 2, :]
        logg_err_feat = enhanced_feats[:, 3, :]
        feh_pred_feat = enhanced_feats[:, 4, :]
        feh_err_feat = enhanced_feats[:, 5, :]
        # 最终输出
        return torch.cat([
            self.teff_pred(teff_pred_feat),
            self.teff_err(teff_err_feat),
            self.logg_pred(logg_pred_feat),
            self.logg_err(logg_err_feat),
            self.feh_pred(feh_pred_feat),
            self.feh_err(feh_err_feat)
        ], dim=1)


    def _format_output(self, param_feat):
        """格式化输出"""
        # 分离各参数的特征
        teff_feat = param_feat[:, 0, :]  # [B, 256]
        logg_feat = param_feat[:, 1, :]
        feh_feat = param_feat[:, 2, :]
        # error_feat = self.error_feat_adjust(error_feat)
        outputs = [
            self.teff_layer(teff_feat),
            self.logg_layer(logg_feat),
            self.feh_layer(feh_feat),
            # self.rv_layer(error_feat)
        ]
        # 格式化输出
        results = []
        for pred in outputs:
            results.extend([pred[:, 0:1], torch.exp(pred[:, 1:2])])
        # 验证所有输出维度
        for i, t in enumerate(results):
            assert t.dim() == 2, f"输出{i}维度错误: {t.shape}"

        return torch.cat(results, dim=1)

    @classmethod
    def default_spectral_config(cls):
        """测试配置"""
        return cls(
            image_size=(64, 64),
            in_channels=1,
            num_blocks=[2, 2, 6, 14, 2],
            channels=[64, 128, 256, 512, 1024],
            block_types=['C', 'C', 'T', 'T'],
            od=False
        )

def coatnet_0():
    num_blocks = [2, 2, 3, 5, 2]  # L
    channels = [64, 96, 192, 384, 768]  # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_1():
    num_blocks = [2, 2, 6, 14, 2]  # L
    channels = [64, 96, 192, 384, 768]  # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_2():
    num_blocks = [2, 2, 6, 14, 2]  # L
    channels = [128, 128, 256, 512, 1026]  # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_3():
    num_blocks = [2, 2, 6, 14, 2]  # L
    channels = [192, 192, 384, 768, 1536]  # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_4():
    num_blocks = [2, 2, 12, 28, 2]  # L
    channels = [192, 192, 384, 768, 1536]  # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)

    net = coatnet_0()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_1()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_2()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_3()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_4()
    out = net(img)
    print(out.shape, count_parameters(net))

    # import os
    #
    # save_path = 'trained/coatnet.pth'
    # save_dir = os.path.dirname(save_path)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
