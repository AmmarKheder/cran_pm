import math
import torch

from .scan_orders import wind_band_hilbert


class RegionalWindScanner:

    def __init__(self, grid_h, grid_w, region_h=7, region_w=7, num_sectors=16, device="cpu"):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.region_h = region_h
        self.region_w = region_w
        self.num_sectors = num_sectors
        self.num_patches = grid_h * grid_w
        self.device = device

        self.regions = self._build_regions()
        self.n_regions = len(self.regions)
        self._scan_cache = {}
        self._precompute_all()

    def _build_regions(self):
        regions = []
        row_starts = list(range(0, self.grid_h, self.region_h))
        col_starts = list(range(0, self.grid_w, self.region_w))

        for i, r0 in enumerate(row_starts):
            r1 = row_starts[i + 1] if i + 1 < len(row_starts) else self.grid_h
            rh = r1 - r0
            for j, c0 in enumerate(col_starts):
                c1 = col_starts[j + 1] if j + 1 < len(col_starts) else self.grid_w
                rw = c1 - c0
                indices = [
                    (r0 + ri) * self.grid_w + (c0 + ci)
                    for ri in range(rh)
                    for ci in range(rw)
                ]
                regions.append({
                    "indices": torch.tensor(indices, dtype=torch.long, device=self.device),
                    "rh": rh, "rw": rw, "r0": r0, "c0": c0,
                })
        return regions

    def _precompute_all(self):
        sizes = set((r["rh"], r["rw"]) for r in self.regions)
        for rh, rw in sizes:
            for s in range(self.num_sectors):
                angle = 2 * math.pi * s / self.num_sectors
                order = wind_band_hilbert(rh, rw, angle)
                order_t = torch.tensor(order, dtype=torch.long, device=self.device)
                inv_t = torch.empty_like(order_t)
                inv_t[order_t] = torch.arange(len(order), dtype=torch.long, device=self.device)
                self._scan_cache[(rh, rw, s)] = (order_t, inv_t)

    def _get_regional_sectors(self, u_wind, v_wind):
        B, H, W = u_wind.shape
        ppH = H / self.grid_h
        ppW = W / self.grid_w
        sectors = torch.zeros(B, self.n_regions, dtype=torch.long, device=u_wind.device)

        for idx, reg in enumerate(self.regions):
            pr0 = int(reg["r0"] * ppH)
            pr1 = min(int((reg["r0"] + reg["rh"]) * ppH), H)
            pc0 = int(reg["c0"] * ppW)
            pc1 = min(int((reg["c0"] + reg["rw"]) * ppW), W)

            u_local = u_wind[:, pr0:pr1, pc0:pc1].mean(dim=(-2, -1))
            v_local = v_wind[:, pr0:pr1, pc0:pc1].mean(dim=(-2, -1))

            angle = torch.atan2(v_local, u_local)
            angle = (angle + 2 * math.pi) % (2 * math.pi)
            sec = (angle / (2 * math.pi) * self.num_sectors).long() % self.num_sectors
            sectors[:, idx] = sec

        return sectors

    def reorder(self, tokens, u_wind, v_wind):
        B, N, D = tokens.shape
        sectors = self._get_regional_sectors(u_wind, v_wind)
        reordered = tokens.clone()
        for b in range(B):
            for r_idx, reg in enumerate(self.regions):
                s = sectors[b, r_idx].item()
                gidx = reg["indices"]
                order, _ = self._scan_cache[(reg["rh"], reg["rw"], s)]
                reordered[b, gidx] = tokens[b, gidx[order]]
        return reordered, sectors

    def reorder_like(self, tensor, sectors):
        is_2d = tensor.dim() == 2
        if is_2d:
            tensor = tensor.unsqueeze(-1)
        B = tensor.shape[0]
        out = tensor.clone()
        for b in range(B):
            for r_idx, reg in enumerate(self.regions):
                s = sectors[b, r_idx].item()
                gidx = reg["indices"]
                order, _ = self._scan_cache[(reg["rh"], reg["rw"], s)]
                out[b, gidx] = tensor[b, gidx[order]]
        return out.squeeze(-1) if is_2d else out

    def inverse_reorder(self, tokens, sectors):
        B, N, D = tokens.shape
        restored = tokens.clone()
        for b in range(B):
            for r_idx, reg in enumerate(self.regions):
                s = sectors[b, r_idx].item()
                gidx = reg["indices"]
                _, inv = self._scan_cache[(reg["rh"], reg["rw"], s)]
                restored[b, gidx] = tokens[b, gidx[inv]]
        return restored

    def to(self, device):
        self.device = device
        for reg in self.regions:
            reg["indices"] = reg["indices"].to(device)
        new_cache = {}
        for key, (order, inv) in self._scan_cache.items():
            new_cache[key] = (order.to(device), inv.to(device))
        self._scan_cache = new_cache
        return self
