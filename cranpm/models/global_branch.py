import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from timm.layers import trunc_normal_

from .topoflow_block import TopoFlowBlock, compute_patch_coords, compute_patch_elevations
from .wind_scan import RegionalWindScanner
from ..utils.pos_embed import get_2d_sincos_pos_embed


class GlobalBranch(nn.Module):

    def __init__(
        self,
        in_channels: int = 70,
        img_size: tuple = (168, 280),
        patch_size: int = 8,
        embed_dim: int = 768,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        drop_path: float = 0.1,
        u_channel: int = 0,
        v_channel: int = 1,
        region_h: int = 7,
        region_w: int = 7,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.u_channel = u_channel
        self.v_channel = v_channel
        self.region_h = region_h
        self.region_w = region_w

        H, W = img_size
        self.grid_h = H // patch_size
        self.grid_w = W // patch_size
        self.num_patches = self.grid_h * self.grid_w

        self.patch_embed = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        self.norm_embed = nn.LayerNorm(embed_dim)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )
        self.lead_time_embed = nn.Linear(1, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList()
        self.blocks.append(
            TopoFlowBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=drop_rate,
                drop_path=dpr[0],
            )
        )
        for i in range(1, depth):
            self.blocks.append(
                Block(
                    embed_dim, num_heads, mlp_ratio,
                    qkv_bias=True, drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                )
            )
        self.norm = nn.LayerNorm(embed_dim)
        self.wind_scanner = None
        self._init_weights()

    def _init_weights(self):
        pos = get_2d_sincos_pos_embed(self.embed_dim, self.grid_h, self.grid_w)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)
        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _ensure_wind_scanner(self, device):
        if self.wind_scanner is None:
            self.wind_scanner = RegionalWindScanner(
                self.grid_h, self.grid_w,
                region_h=self.region_h,
                region_w=self.region_w,
                num_sectors=16,
                device=device,
            )

    def _crop_to_grid(self, x):
        _, _, H, W = x.shape
        tgt_h, tgt_w = self.img_size
        if H > tgt_h or W > tgt_w:
            dh = (H - tgt_h) // 2
            dw = (W - tgt_w) // 2
            x = x[:, :, dh:dh + tgt_h, dw:dw + tgt_w]
        return x

    def forward(self, era5, elevation, lead_time):
        era5 = self._crop_to_grid(era5)
        B = era5.shape[0]

        u_wind = era5[:, self.u_channel]
        v_wind = era5[:, self.v_channel]

        patches = F.unfold(era5, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.transpose(1, 2)
        x = self.patch_embed(patches)
        x = self.norm_embed(x)

        x = x + self.pos_embed
        lt = self.lead_time_embed(lead_time.unsqueeze(-1).float())
        x = x + lt.unsqueeze(1)
        x = self.pos_drop(x)

        elevation = self._crop_to_grid(elevation.unsqueeze(1)).squeeze(1)
        coords_2d = compute_patch_coords(self.img_size, self.patch_size, x.device)
        coords_2d = coords_2d.expand(B, -1, -1)
        elev_patches = compute_patch_elevations(elevation, self.patch_size)

        self._ensure_wind_scanner(x.device)
        x, sectors = self.wind_scanner.reorder(x, u_wind, v_wind)
        coords_2d = self.wind_scanner.reorder_like(coords_2d, sectors)
        elev_patches = self.wind_scanner.reorder_like(elev_patches, sectors)

        for i, blk in enumerate(self.blocks):
            if i == 0:
                x = blk(x, coords_2d, elev_patches)
            else:
                x = blk(x)

        x = self.norm(x)
        x = self.wind_scanner.inverse_reorder(x, sectors)
        return x
