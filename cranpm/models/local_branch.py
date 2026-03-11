import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from timm.layers import trunc_normal_

from .topoflow_block import TopoFlowBlock, compute_patch_coords, compute_patch_elevations
from ..utils.pos_embed import get_2d_sincos_pos_embed


class LocalBranch(nn.Module):

    def __init__(
        self,
        in_channels: int = 5,
        img_size: tuple = (512, 512),
        patch_size: int = 16,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        H, W = img_size
        self.grid_h = H // patch_size
        self.grid_w = W // patch_size
        self.num_patches = self.grid_h * self.grid_w

        self.patch_embed = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        self.norm_embed = nn.LayerNorm(embed_dim)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )
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
                elevation_scale=500.0,
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

    def forward(self, local_input, elevation_hires):
        B = local_input.shape[0]

        patches = F.unfold(local_input, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.transpose(1, 2)
        x = self.patch_embed(patches)
        skip = x.transpose(1, 2).reshape(B, self.embed_dim, self.grid_h, self.grid_w)
        x = self.norm_embed(x)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        coords_2d = compute_patch_coords(self.img_size, self.patch_size, x.device)
        coords_2d = coords_2d.expand(B, -1, -1)
        elev_patches = compute_patch_elevations(elevation_hires, self.patch_size)

        for i, blk in enumerate(self.blocks):
            if i == 0:
                x = blk(x, coords_2d, elev_patches)
            else:
                x = blk(x)

        return self.norm(x), skip
