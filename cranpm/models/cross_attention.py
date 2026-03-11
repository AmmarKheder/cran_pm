import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
from timm.layers import DropPath


class CrossAttentionBridge(nn.Module):

    def __init__(
        self,
        local_dim: int = 512,
        global_dim: int = 768,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        num_layers: int = 2,
        global_grid_h: int = 21,
        global_grid_w: int = 35,
    ):
        super().__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.num_heads = num_heads

        self.global_proj = (
            nn.Linear(global_dim, local_dim)
            if global_dim != local_dim
            else nn.Identity()
        )

        self.layers = nn.ModuleList([
            WindGuidedCrossAttentionLayer(
                dim=local_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
            )
            for _ in range(num_layers)
        ])

        self.global_grid_h = global_grid_h
        self.global_grid_w = global_grid_w
        self._register_global_coords()

    def _register_global_coords(self):
        gh, gw = self.global_grid_h, self.global_grid_w
        patch_deg = 8 * 0.25
        lats = 72.0 - (torch.arange(gh).float() + 0.5) * patch_deg
        lons = -25.0 + (torch.arange(gw).float() + 0.5) * patch_deg
        lat_grid = lats.unsqueeze(1).expand(gh, gw).reshape(-1)
        lon_grid = lons.unsqueeze(0).expand(gh, gw).reshape(-1)
        global_coords = torch.stack([lat_grid, lon_grid], dim=-1)
        self.register_buffer("global_coords", global_coords, persistent=False)

    def forward(self, local_tokens, global_tokens, patch_center=None, wind_at_patch=None):
        global_proj = self.global_proj(global_tokens)

        wind_bias = None
        if patch_center is not None and wind_at_patch is not None:
            wind_bias = self._compute_wind_bias(patch_center, wind_at_patch)

        x = local_tokens
        for layer in self.layers:
            x = layer(x, global_proj, wind_bias=wind_bias)
        return x

    def _compute_wind_bias(self, patch_center, wind_at_patch):
        B = patch_center.shape[0]
        gc = self.global_coords.unsqueeze(0).expand(B, -1, -1)
        pc = patch_center.unsqueeze(1)

        direction = pc - gc
        dir_norm = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = direction / dir_norm

        wind_dir = torch.stack([wind_at_patch[:, 1], wind_at_patch[:, 0]], dim=-1)
        wind_speed = wind_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        wind_dir = wind_dir / wind_speed

        alignment = (direction * wind_dir.unsqueeze(1)).sum(dim=-1)
        return alignment.unsqueeze(1).unsqueeze(1)


class WindGuidedCrossAttentionLayer(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)

        self.beta = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_ffn = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop,
        )

    def forward(self, query_tokens, context_tokens, wind_bias=None):
        B, N_q, D = query_tokens.shape
        N_c = context_tokens.shape[1]

        q = self.q_proj(self.norm_q(query_tokens))
        kv_input = self.norm_kv(context_tokens)
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)

        q = q.reshape(B, N_q, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N_c, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N_c, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if wind_bias is not None:
            attn = attn + self.beta * wind_bias

        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, D)
        out = self.out_proj(out)

        query_tokens = query_tokens + self.drop_path(out)
        query_tokens = query_tokens + self.drop_path(self.mlp(self.norm_ffn(query_tokens)))
        return query_tokens
