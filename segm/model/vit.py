"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn

from segm.model.utils import init_weights, resize_pos_embed
from segm.model.blocks import Block
from segm.model.stvit import Deit, PatchMerging

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _load_weights


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im):
        B, C, H, W = im.shape
        x = self.proj(im).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        n_cls,
        dropout=0.1,
        drop_path_rate=0.0,
        distilled=False,
        channels=3,
        backbone=None,
        stvitr_cfg=None
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            d_model,
            channels,
        )
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout_value = dropout
        self.dropout = nn.Dropout(dropout)
        self.drop_path_rate = drop_path_rate
        self.n_cls = n_cls

        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.distilled = distilled
        if self.distilled:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.num_patches + 2, d_model)
            )
            self.head_dist = nn.Linear(d_model, n_cls)
        else:
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.num_patches + 1, d_model)
            )

        self.build_tranformer_blocks(backbone, extra_cfg=stvitr_cfg)

        # output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_cls)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if self.distilled:
            trunc_normal_(self.dist_token, std=0.02)
        self.pre_logits = nn.Identity()

        self.apply(init_weights)

    def build_tranformer_blocks(self, backbone, extra_cfg=None):
        # TODO: what will it be for STViT-R?
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.n_layers)]
        if backbone.startswith("stvitr"):
            # first some checks
            if not(self.n_layers == 12 or self.n_layer == 24):
                raise ValueError(f"STViT-R model is implemented only with 12 or 24 layers. "
                                 f"Given: {self.n_layers} layers.")
            if self.patch_embed.image_size[0] != self.patch_embed.image_size[1]:
                raise ValueError(f"For STViT-R block to work in this implementation, image size must be square. "
                                 f"Actual image size: {self.patch_embed.image_size}")

            blocks = [Block(self.d_model, self.n_heads, self.d_ff, self.dropout_value, dpr[i]) for i in range(4)]
            blocks += [Deit(
                input_resolution= self.patch_embed.image_size, #(self.patch_embed.grid_size[0] // 4, self.patch_embed.grid_size[1] // 4),  # TODO: It was like that in STViT-Swin, I'm not sure..
                embed_dim=self.d_model,
                depth=(self.n_layers - 6) // 6,  # 1 if 12 layers overall, 3 if 24 layers overall.
                num_heads=self.n_heads,
                window_size=self.patch_embed.image_size[0] // 32, # TODO: why is this hardcoded in STViT? Maybe I'm wrong..
                sample_window_size=3,  # TODO: paper section 3.2, w_s
                mlp_ratio=extra_cfg.MODEL.SWIN.MLP_RATIO,  # Segmenter: 4, STViT: seems to have 4 too
                qkv_bias=extra_cfg.MODEL.SWIN.QKV_BIAS, # TODO: check models, maybe it's True for all of them
                drop_rate=extra_cfg.MODEL.DROP_RATE,
                attn_drop_rate=0.,
                drop_path_rate=dpr[4:-2],  # TODO: check this
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging,
                multi_scale=extra_cfg.MULTI_SCALE,
                relative_pos=extra_cfg.RELATIVE_POS,
                use_conv_pos=extra_cfg.USE_CONV_POS,
                shortcut=extra_cfg.SHORTCUT,
                use_global=extra_cfg.USE_GLOBAL
            )]
            blocks += [Block(self.d_model, self.n_heads, self.d_ff, self.dropout_value, dpr[i]) for i in
                      range(self.n_layers - 2, self.n_layers)]
            self.blocks = nn.ModuleList(blocks)
        else:
            self.blocks = nn.ModuleList(
                [Block(self.d_model, self.n_heads, self.d_ff, self.dropout_value, dpr[i]) for i in range(self.n_layers)]
            )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward(self, im, return_features=False):
        B, _, H, W = im.shape
        PS = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + self.distilled
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // PS, W // PS),
                num_extra_tokens,
            )
        x = x + pos_embed
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if return_features:
            return x

        if self.distilled:
            x, x_dist = x[:, 0], x[:, 1]
            x = self.head(x)
            x_dist = self.head_dist(x_dist)
            x = (x + x_dist) / 2
        else:
            x = x[:, 0]
            x = self.head(x)
        return x

    def get_attention_map(self, im, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        B, _, H, W = im.shape
        PS = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + self.distilled
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // PS, W // PS),
                num_extra_tokens,
            )
        x = x + pos_embed

        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)
