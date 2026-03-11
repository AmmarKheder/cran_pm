import torch
import torch.nn as nn

from .global_branch import GlobalBranch
from .local_branch import LocalBranch
from .cross_attention import CrossAttentionBridge
from .decoder import CNNDecoder


class CranPM(nn.Module):

    def __init__(
        self,
        era5_channels: int = 70,
        global_img_size: tuple = (168, 280),
        global_patch_size: int = 8,
        global_embed_dim: int = 768,
        global_depth: int = 8,
        global_num_heads: int = 12,
        local_channels: int = 5,
        local_img_size: tuple = (512, 512),
        local_patch_size: int = 16,
        local_embed_dim: int = 512,
        local_depth: int = 6,
        local_num_heads: int = 8,
        cross_num_heads: int = 8,
        cross_layers: int = 2,
        decoder_depth: int = 2,
        out_channels: int = 1,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        drop_path: float = 0.1,
        global_region_h: int = 7,
        global_region_w: int = 7,
    ):
        super().__init__()

        self.global_branch = GlobalBranch(
            in_channels=era5_channels,
            img_size=global_img_size,
            patch_size=global_patch_size,
            embed_dim=global_embed_dim,
            depth=global_depth,
            num_heads=global_num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path=drop_path,
            region_h=global_region_h,
            region_w=global_region_w,
        )

        self.local_branch = LocalBranch(
            in_channels=local_channels,
            img_size=local_img_size,
            patch_size=local_patch_size,
            embed_dim=local_embed_dim,
            depth=local_depth,
            num_heads=local_num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path=drop_path,
        )

        global_grid_h = global_img_size[0] // global_patch_size
        global_grid_w = global_img_size[1] // global_patch_size

        self.cross_attention = CrossAttentionBridge(
            local_dim=local_embed_dim,
            global_dim=global_embed_dim,
            num_heads=cross_num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=drop_rate,
            drop_path=drop_path,
            num_layers=cross_layers,
            global_grid_h=global_grid_h,
            global_grid_w=global_grid_w,
        )

        local_grid_h = local_img_size[0] // local_patch_size
        local_grid_w = local_img_size[1] // local_patch_size

        self.prediction_head = CNNDecoder(
            embed_dim=local_embed_dim,
            grid_h=local_grid_h,
            grid_w=local_grid_w,
            out_channels=out_channels,
            skip_channels=local_embed_dim,
        )

    def forward(self, era5, elevation_coarse, ghap_patch, elevation_hires, lead_time,
                patch_center=None, wind_at_patch=None):
        global_feats = self.global_branch(era5, elevation_coarse, lead_time)
        local_feats, skip = self.local_branch(ghap_patch, elevation_hires)
        fused = self.cross_attention(
            local_feats, global_feats,
            patch_center=patch_center,
            wind_at_patch=wind_at_patch,
        )
        delta = self.prediction_head(fused, skip=skip)
        ghap_today = ghap_patch[:, 0:1, :, :]
        return ghap_today + delta
