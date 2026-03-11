import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
from timm.layers import DropPath


class RelativePositionBias2D(nn.Module):

    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bias_table = nn.Parameter(torch.zeros(num_buckets * num_buckets, num_heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def _bucket_positions(self, relative_position, max_distance):
        num_buckets = self.num_buckets
        ret = 0
        n = -relative_position
        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / torch.log(torch.tensor(max_distance / max_exact, dtype=torch.float32))
            * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        return ret + torch.where(is_small, n, val_if_large)

    def forward(self, coords_2d: torch.Tensor):
        B, N, _ = coords_2d.shape
        coords_int = (coords_2d * self.max_distance).long()
        rel_x = coords_int[:, :, None, 0] - coords_int[:, None, :, 0]
        rel_y = coords_int[:, :, None, 1] - coords_int[:, None, :, 1]
        bucket_x = self._bucket_positions(rel_x, self.max_distance)
        bucket_y = self._bucket_positions(rel_y, self.max_distance)
        bucket_idx = bucket_x * self.num_buckets + bucket_y
        rel_bias = self.bias_table[bucket_idx]
        return rel_bias.permute(0, 3, 1, 2).contiguous()


class TopoFlowAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        elevation_scale: float = 1000.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.elevation_scale = elevation_scale

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rel_pos_bias = RelativePositionBias2D(num_heads=num_heads)
        self.alpha = nn.Parameter(torch.ones(1) * 2.0)

    def _compute_elevation_bias(self, elevation_patches):
        elev_i = elevation_patches.unsqueeze(2)
        elev_j = elevation_patches.unsqueeze(1)
        elev_diff = (elev_j - elev_i) / self.elevation_scale
        elevation_bias = -self.alpha * F.relu(elev_diff)
        return torch.clamp(elevation_bias, min=-10.0, max=0.0)

    def forward(self, x, coords_2d, elevation_patches):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn + self.rel_pos_bias(coords_2d)

        elev_bias = self._compute_elevation_bias(elevation_patches)
        attn = attn + elev_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(out))


class TopoFlowBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        elevation_scale: float = 1000.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TopoFlowAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            elevation_scale=elevation_scale,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop,
        )

    def forward(self, x, coords_2d, elevation_patches):
        x = x + self.drop_path(self.attn(self.norm1(x), coords_2d, elevation_patches))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def compute_patch_coords(img_size, patch_size, device):
    H, W = img_size
    ph, pw = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
    h, w = H // ph, W // pw
    y = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    x = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return coords.unsqueeze(0)


def compute_patch_elevations(elevation_field, patch_size):
    B, H, W = elevation_field.shape
    ph, pw = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
    elev = elevation_field.unsqueeze(1)
    elev = F.avg_pool2d(elev, kernel_size=(ph, pw), stride=(ph, pw))
    return elev.squeeze(1).reshape(B, -1)
