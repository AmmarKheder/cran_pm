import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLoss(nn.Module):

    def __init__(
        self,
        alpha_mse: float = 1.0,
        alpha_ssim: float = 0.0,
        alpha_grad: float = 0.0,
        alpha_spectral: float = 0.1,
        alpha_station: float = 0.1,
        ghap_mean: float = 15.0,
        ghap_std: float = 20.0,
        hotspot_alpha: float = 2.0,
        hotspot_threshold: float = 0.25,
        hotspot_scale: float = 0.5,
        hotspot_max_weight: float = 5.0,
        underestimate_penalty: float = 1.0,
        ffl_alpha: float = 1.0,
    ):
        super().__init__()
        self.alpha_mse = alpha_mse
        self.alpha_ssim = alpha_ssim
        self.alpha_grad = alpha_grad
        self.alpha_spectral = alpha_spectral
        self.alpha_station = alpha_station
        self.ffl_alpha = ffl_alpha
        self.land_threshold = -ghap_mean / ghap_std

        self.hotspot_alpha = hotspot_alpha
        self.hotspot_threshold = hotspot_threshold
        self.hotspot_scale = hotspot_scale
        self.hotspot_max_weight = hotspot_max_weight
        self.underestimate_penalty = underestimate_penalty

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _land_mask(self, target):
        return (target > self.land_threshold) & torch.isfinite(target)

    def _pixel_weights(self, target):
        excess = (target - self.hotspot_threshold) / self.hotspot_scale
        excess = excess.clamp(min=0.0, max=self.hotspot_max_weight)
        return 1.0 + self.hotspot_alpha * excess

    def _masked_mse(self, pred, target, mask):
        mask_f = mask.float()
        weights = self._pixel_weights(target) * mask_f
        under = (pred < target).float()
        asym = 1.0 + (self.underestimate_penalty - 1.0) * under
        weights = weights * asym
        denom = weights.sum().clamp(min=1.0)
        diff = (pred - target) ** 2
        return (diff * weights).sum() / denom

    def _spatial_gradient_loss(self, pred, target, mask):
        pred_gx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1)
        tgt_gx = F.conv2d(target, self.sobel_x, padding=1)
        tgt_gy = F.conv2d(target, self.sobel_y, padding=1)
        mask_f = mask.float()
        mask_eroded = F.avg_pool2d(mask_f, 3, stride=1, padding=1)
        mask_eroded = (mask_eroded > 0.99).float()
        denom = mask_eroded.sum().clamp(min=1.0)
        grad_diff = (pred_gx - tgt_gx) ** 2 + (pred_gy - tgt_gy) ** 2
        return (grad_diff * mask_eroded).sum() / denom

    def _spectral_loss(self, pred, target, mask):
        pred_clean = torch.where(mask, pred, torch.zeros_like(pred))
        target_clean = torch.where(mask, target, torch.zeros_like(target))
        pred_clean = torch.nan_to_num(pred_clean, nan=0.0, posinf=0.0, neginf=0.0)
        target_clean = torch.nan_to_num(target_clean, nan=0.0, posinf=0.0, neginf=0.0)
        pred_fft = torch.fft.rfft2(pred_clean.float(), norm="ortho")
        target_fft = torch.fft.rfft2(target_clean.float(), norm="ortho")
        diff = pred_fft - target_fft
        diff_amp = torch.abs(diff)
        weight = diff_amp.detach().clamp(max=10.0) ** self.ffl_alpha
        return (weight * diff_amp ** 2).mean()

    def _station_loss(self, pred, station_pixels, station_values, station_count):
        B, _, H, W = pred.shape
        total_loss = torch.tensor(0.0, device=pred.device)
        total_count = 0

        for b in range(B):
            n = station_count[b].item()
            if n == 0:
                continue
            px = station_pixels[b, :n]
            sv = station_values[b, :n]
            grid_x = (px[:, 1] / (W - 1)) * 2 - 1
            grid_y = (px[:, 0] / (H - 1)) * 2 - 1
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)
            pred_at_stations = F.grid_sample(
                pred[b:b+1], grid.to(pred.dtype),
                mode="bilinear", padding_mode="border", align_corners=True,
            )
            pred_vals = pred_at_stations.squeeze()
            total_loss = total_loss + ((pred_vals - sv.to(pred.dtype)) ** 2).sum()
            total_count += n

        if total_count == 0:
            return total_loss
        return total_loss / total_count

    def forward(self, pred, target, station_pixels=None, station_values=None,
                station_count=None):
        mask = self._land_mask(target)

        mse = self._masked_mse(pred, target, mask)
        loss = self.alpha_mse * mse
        metrics = {"mse": mse.detach()}

        if self.alpha_grad > 0:
            grad_loss = self._spatial_gradient_loss(pred, target, mask)
            loss = loss + self.alpha_grad * grad_loss
            metrics["grad_loss"] = grad_loss.detach()

        if self.alpha_spectral > 0:
            spectral_loss = self._spectral_loss(pred, target, mask)
            loss = loss + self.alpha_spectral * spectral_loss
            metrics["spectral_loss"] = spectral_loss.detach()

        if self.alpha_station > 0 and station_pixels is not None:
            stn_loss = self._station_loss(pred, station_pixels, station_values, station_count)
            loss = loss + self.alpha_station * stn_loss
            metrics["station_loss"] = stn_loss.detach()

        if self.alpha_ssim > 0:
            ssim_loss = self._ssim_loss(pred, target)
            loss = loss + self.alpha_ssim * ssim_loss
            metrics["ssim_loss"] = ssim_loss.detach()

        metrics["loss"] = loss.detach()
        metrics["land_pct"] = mask.float().mean().detach()
        return loss, metrics
