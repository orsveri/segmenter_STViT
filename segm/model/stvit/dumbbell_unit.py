"""
Modidfication of Swin-based STViT module Deit from repository:
https://github.com/changsn/STViT-R
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from segm.model.stvit.layers import STViT_Block, STViT_SemanticAttentionBlock, STViT_Block_simple
from segm.model.stvit.multi_scale import *
import math

from segm.model.blocks import Block


# TODO: naming for weights initialization? Will it be consistent for the global transformer case?
class Dumbbell_unit_global(nn.Module):
    """ Vision Transformer
    A dumbbell unit for global transformers
    use_layer_scale: bool, whether to use original Segmenter transformer blocks without LayerScale or modified ones
    """
    def __init__(self, input_resolution, window_size, sample_window_size, multi_scale, embed_dim=768, depth=6,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=None, act_layer=None, fuse_loc=0, split_loc=2, semantic_key_concat=False,
                 downsample=None, use_conv_pos=False, shortcut=False, use_global=True,
                 use_layer_scale=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.multi_scale = multi_scale
        num_windows = (input_resolution[0] // window_size) * (input_resolution[1] // window_size)
        self.num_samples = sample_window_size * sample_window_size * num_windows
        self.fuse_loc = fuse_loc
        self.split_loc = split_loc
        self.semantic_key_concat = semantic_key_concat
        self.input_resolution = input_resolution
        self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.use_global = use_global
        if self.use_global:
            self.semantic_token2 = nn.Parameter(torch.zeros(1, self.num_samples, embed_dim))
            trunc_normal_(self.semantic_token2, std=.02)
        self.use_layer_scale = use_layer_scale
        # self.up_dim = nn.Linear(self.embed_dim, 2*self.embed_dim)
        # self.semantic_token_ = nn.Parameter(torch.zeros(1, self.num_samples, embed_dim))
        # trunc_normal_(self.semantic_token_, std=.02)

        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i in [0, 6, 12]:
                if self.use_layer_scale:
                    self.blocks.append(
                        STViT_Block_simple(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                           attn_drop=attn_drop_rate, drop=drop_rate, drop_path=drop_path_rate[i],
                                           norm_layer=norm_layer, act_layer=act_layer)
                        )
                else:
                    self.blocks.append(
                        Block(dim=self.d_model, heads=self.n_heads, mlp_dim=mlp_ratio*self.d_model,
                              dropout=self.dropout_value, drop_path=drop_path_rate[i])
                    )

            elif i in [1, 7, 13]:
                self.blocks.append(STViT_SemanticAttentionBlock(
                    dim=embed_dim, window_size=window_size, sample_window_size=sample_window_size,
                    num_heads=num_heads, multi_scale=self.multi_scale, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate[i],
                    norm_layer=norm_layer, act_layer=act_layer,
                    use_conv_pos=use_conv_pos, shortcut=shortcut)
                    )
            elif i in [2, 5, 8, 11, 14, 17]:
                # TODO: these layers take semantic tokens as second input and they use layer_scale
                self.blocks.append(
                    STViT_Block_simple(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                       attn_drop=attn_drop_rate, drop=drop_rate, drop_path=drop_path_rate[i],
                                       norm_layer=norm_layer, act_layer=act_layer)
                    )
            else:
                if self.use_layer_scale:
                    self.blocks.append(
                        STViT_Block_simple(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                           attn_drop=attn_drop_rate, drop=drop_rate, drop_path=drop_path_rate[i],
                                           norm_layer=norm_layer, act_layer=act_layer)
                    )
                else:
                    self.blocks.append(
                        Block(dim=self.d_model, heads=self.n_heads, mlp_dim=mlp_ratio * self.d_model,
                              dropout=self.dropout_value, drop_path=drop_path_rate[i])
                    )
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=self.embed_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for i, blk in enumerate(self.blocks):
            if i == 0:
                x = blk(x)
            elif i == 1:
                semantic_token = blk(x)
            elif i == 2:
                if self.use_global:
                    semantic_token = blk(semantic_token+self.semantic_token2, torch.cat([semantic_token, x], dim=1))
                else:
                    semantic_token = blk(semantic_token, torch.cat([semantic_token, x], dim=1))
            elif i > 2 and i < 5:
                semantic_token = blk(semantic_token)
            elif i == 5:
                x = blk(x, semantic_token)
            elif i == 6:
                x = blk(x)
            elif i == 7:
                semantic_token = blk(x)
            elif i == 8:
                semantic_token = blk(semantic_token, torch.cat([semantic_token, x], dim=1))
            elif i > 8 and i < 11:
                semantic_token = blk(semantic_token)
            elif i == 11:
                x = blk(x, semantic_token)
            elif i == 12:
                x = blk(x)
            elif i == 13:
                semantic_token = blk(x)
            elif i == 14:
                semantic_token = blk(semantic_token, torch.cat([semantic_token, x], dim=1))
            elif i > 14 and i < 17:
                semantic_token = blk(semantic_token)
            else:
                x = blk(x, semantic_token)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def flops(self):
        flops = 0
        return flops