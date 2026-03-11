import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
import zarr


ERA5_NORM = {
    "u10": (0.0, 10.0), "v10": (0.0, 10.0), "t2m": (280.0, 20.0),
    "msl": (101325., 1500.), "sp": (97000., 5000.),
}
PRESSURE_NORM = {
    "t": (260.0, 30.0), "u": (0.0, 15.0), "v": (0.0, 10.0),
    "q": (0.003, 0.004), "z": (50000., 40000.),
}
PRESSURE_LEVELS = [1000, 925, 850, 700, 500]

CAMS_NORM = {
    "no2": (2.15, 3.52), "o3": (71.75, 18.18), "so2": (0.85, 1.69),
    "co": (140.8, 36.7), "pm10": (14.0, 13.7),
}
CAMS_VARS = ["no2", "o3", "so2", "co", "pm10"]
N_CAMS = len(CAMS_VARS)

GHAP_MEAN = 15.0
GHAP_STD = 20.0
ELEV_MEAN = 300.0
ELEV_STD = 500.0
HORIZONS = [1, 2, 3, 4]

LAT_NORTH = 72.0
LON_WEST = -25.0
GHAP_RES = 0.01
ERA5_LAT_NORTH = 72.0
ERA5_LON_WEST = -25.0
ERA5_RES = 0.25


def _build_era5_norm():
    means, stds = [], []
    for var in ["u10", "v10", "t2m", "msl", "sp"]:
        m, s = ERA5_NORM[var]
        means.append(m)
        stds.append(s)
    for var in ["t", "u", "v", "q", "z"]:
        for _ in PRESSURE_LEVELS:
            m, s = PRESSURE_NORM[var]
            means.append(m)
            stds.append(s)
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


ERA5_MEANS, ERA5_STDS = _build_era5_norm()
CAMS_MEANS = np.array([CAMS_NORM[v][0] for v in CAMS_VARS], dtype=np.float32)
CAMS_STDS = np.array([CAMS_NORM[v][1] for v in CAMS_VARS], dtype=np.float32)


class CranPMDataset(Dataset):

    MAX_STATIONS_PER_PATCH = 64

    def __init__(
        self,
        era5_dir: str,
        ghap_dir: str,
        elev_coarse_path: str,
        elev_hires_path: str,
        years: list,
        cams_dir: str = None,
        eea_zarr_path: str = None,
        horizons: list = None,
        patch_size: int = 512,
        normalize: bool = True,
        augment: bool = False,
        hotspot_ratio: float = 0.5,
        hotspot_power: float = 1.0,
    ):
        self.era5_dir = Path(era5_dir)
        self.ghap_dir = Path(ghap_dir)
        self.elev_coarse_path = Path(elev_coarse_path)
        self.elev_hires_path = Path(elev_hires_path)
        self.cams_dir = Path(cams_dir) if cams_dir else None
        self.eea_zarr_path = Path(eea_zarr_path) if eea_zarr_path else None
        self.years = years
        self.horizons = horizons or HORIZONS
        self.patch_size = patch_size
        self.normalize = normalize
        self.augment = augment
        self.use_cams = cams_dir is not None
        self.hotspot_ratio = hotspot_ratio
        self.hotspot_power = hotspot_power
        self.max_horizon = max(self.horizons)

        self._load_data()
        self._load_eea_stations()
        self._build_index()

    def _load_data(self):
        self.era5_stores = {}
        self.ghap_stores = {}
        self.cams_stores = {}
        self.year_days = {}

        for year in self.years:
            era5_path = self.era5_dir / f"{year}.zarr"
            ghap_path = self.ghap_dir / f"{year}.zarr"
            if era5_path.exists() and ghap_path.exists():
                self.era5_stores[year] = zarr.open(str(era5_path), mode="r")
                self.ghap_stores[year] = zarr.open(str(ghap_path), mode="r")
                self.year_days[year] = self.era5_stores[year].shape[0]
                if self.use_cams:
                    cams_path = self.cams_dir / f"{year}.zarr"
                    if cams_path.exists():
                        self.cams_stores[year] = zarr.open(str(cams_path), mode="r")
                    else:
                        self.use_cams = False

        if self.year_days:
            first_year = next(iter(self.era5_stores))
            self._era5_h = self.era5_stores[first_year].shape[2]
            self._era5_w = self.era5_stores[first_year].shape[3]

        self.elev_coarse = (
            zarr.open(str(self.elev_coarse_path), mode="r")
            if self.elev_coarse_path.exists() else None
        )
        self.elev_hires = (
            zarr.open(str(self.elev_hires_path), mode="r")
            if self.elev_hires_path.exists() else None
        )

    def _load_eea_stations(self):
        self.use_stations = False
        if self.eea_zarr_path is None or not self.eea_zarr_path.exists():
            return

        eea = zarr.open(str(self.eea_zarr_path), mode="r")
        coords = np.array(eea["station_coords"])
        self.station_rows = (LAT_NORTH - coords[:, 0]) / GHAP_RES
        self.station_cols = (coords[:, 1] - LON_WEST) / GHAP_RES
        self.station_daily = {}
        for year in self.years:
            key = f"daily/{year}"
            if key in eea:
                self.station_daily[year] = np.array(eea[key])
        self.use_stations = len(self.station_daily) > 0

    def _get_stations_in_patch(self, year, day_target, r, c):
        MAX = self.MAX_STATIONS_PER_PATCH
        pixels = np.zeros((MAX, 2), dtype=np.float32)
        values = np.full(MAX, np.nan, dtype=np.float32)

        if not self.use_stations or year not in self.station_daily:
            return pixels, values, 0

        daily = self.station_daily[year]
        if day_target >= daily.shape[0]:
            return pixels, values, 0

        ps = self.patch_size
        in_patch = (
            (self.station_rows >= r) & (self.station_rows < r + ps) &
            (self.station_cols >= c) & (self.station_cols < c + ps)
        )
        indices = np.where(in_patch)[0]
        if len(indices) == 0:
            return pixels, values, 0

        day_values = daily[day_target, indices]
        valid = ~np.isnan(day_values) & (day_values >= 0)
        indices = indices[valid]
        day_values = day_values[valid]
        if len(indices) == 0:
            return pixels, values, 0

        n = min(len(indices), MAX)
        indices = indices[:n]
        day_values = day_values[:n]
        pixels[:n, 0] = self.station_rows[indices] - r
        pixels[:n, 1] = self.station_cols[indices] - c
        values[:n] = (day_values - GHAP_MEAN) / GHAP_STD if self.normalize else day_values
        return pixels, values, n

    def _build_index(self):
        self.samples = []
        for year in sorted(self.year_days.keys()):
            n_days = self.year_days[year]
            max_t = n_days - self.max_horizon
            for day_t in range(max_t):
                for h in self.horizons:
                    if day_t + h < n_days:
                        self.samples.append((year, day_t, h))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        year, day_t, horizon = self.samples[idx]
        rng = np.random.default_rng(idx)

        era5_raw = np.array(self.era5_stores[year][day_t]).astype(np.float32)
        era5 = era5_raw.copy()
        if self.normalize:
            era5 = (era5 - ERA5_MEANS[:, None, None]) / ERA5_STDS[:, None, None]

        if self.use_cams and year in self.cams_stores:
            cams_store = self.cams_stores[year]
            cams_channels = []
            for var in CAMS_VARS:
                ch = np.nan_to_num(np.array(cams_store[var][day_t]).astype(np.float32))
                cams_channels.append(ch)
            cams = np.stack(cams_channels, axis=0)
            cams_t = F.interpolate(
                torch.from_numpy(cams).unsqueeze(0),
                size=(self._era5_h, self._era5_w), mode="bilinear", align_corners=False
            ).squeeze(0).numpy()
            if self.normalize:
                cams_t = (cams_t - CAMS_MEANS[:, None, None]) / CAMS_STDS[:, None, None]
            era5 = np.concatenate([era5, cams_t], axis=0)

        day_prev = max(day_t - 1, 0)
        era5_prev = np.nan_to_num(np.array(self.era5_stores[year][day_prev]).astype(np.float32))
        if self.normalize:
            era5_prev = (era5_prev - ERA5_MEANS[:, None, None]) / ERA5_STDS[:, None, None]
        era5 = np.concatenate([era5, era5_prev], axis=0)

        if self.use_cams and year in self.cams_stores:
            cams_store = self.cams_stores[year]
            cams_prev_channels = []
            for var in CAMS_VARS:
                ch = np.nan_to_num(np.array(cams_store[var][day_prev]).astype(np.float32))
                cams_prev_channels.append(ch)
            cams_prev = np.stack(cams_prev_channels, axis=0)
            cams_prev_t = F.interpolate(
                torch.from_numpy(cams_prev).unsqueeze(0),
                size=(self._era5_h, self._era5_w), mode="bilinear", align_corners=False
            ).squeeze(0).numpy()
            if self.normalize:
                cams_prev_t = (cams_prev_t - CAMS_MEANS[:, None, None]) / CAMS_STDS[:, None, None]
            era5 = np.concatenate([era5, cams_prev_t], axis=0)

        if self.elev_coarse is not None:
            elev_c = np.array(self.elev_coarse[0]).astype(np.float32)
            if self.normalize:
                elev_c = (elev_c - ELEV_MEAN) / ELEV_STD
        else:
            elev_c = np.zeros((169, 281), dtype=np.float32)

        ghap_day = self.ghap_stores[year][day_t]
        ghap_target_day = self.ghap_stores[year][day_t + horizon]
        ghap_prev_day = self.ghap_stores[year][day_prev]

        if rng.random() < (1.0 - self.hotspot_ratio):
            r = rng.integers(0, 4192 - self.patch_size + 1)
            c = rng.integers(0, 6992 - self.patch_size + 1)
        else:
            block = 256
            nr = (4192 - self.patch_size) // block + 1
            nc = (6992 - self.patch_size) // block + 1
            coarse = np.array(ghap_day[::block, ::block]).astype(np.float32)
            weights = np.zeros(nr * nc, dtype=np.float32)
            for bi in range(nr):
                for bj in range(nc):
                    if bi < coarse.shape[0] and bj < coarse.shape[1]:
                        weights[bi * nc + bj] = max(coarse[bi, bj], 0.0) ** self.hotspot_power
            weights += 1.0
            weights /= weights.sum()
            chosen = rng.choice(nr * nc, p=weights)
            r = min((chosen // nc) * block, 4192 - self.patch_size)
            c = min((chosen % nc) * block, 6992 - self.patch_size)

        ghap_patch = np.nan_to_num(
            np.array(ghap_day[r:r + self.patch_size, c:c + self.patch_size]).astype(np.float32)
        )
        ghap_target = np.array(
            ghap_target_day[r:r + self.patch_size, c:c + self.patch_size]
        ).astype(np.float32)
        ghap_prev_patch = np.nan_to_num(
            np.array(ghap_prev_day[r:r + self.patch_size, c:c + self.patch_size]).astype(np.float32)
        )

        if self.elev_hires is not None:
            scale_h = 20160 / 4192
            scale_w = 33600 / 6992
            er = int(r * scale_h)
            ec = int(c * scale_w)
            ep = int(self.patch_size * scale_h)
            elev_h_raw = np.array(self.elev_hires[er:er + ep, ec:ec + ep]).astype(np.float32)
            elev_h = F.interpolate(
                torch.from_numpy(elev_h_raw).unsqueeze(0).unsqueeze(0),
                size=(self.patch_size, self.patch_size), mode="bilinear", align_corners=False
            ).squeeze().numpy()
            if self.normalize:
                elev_h = (elev_h - ELEV_MEAN) / ELEV_STD
        else:
            elev_h = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)

        if self.normalize:
            ghap_patch = (ghap_patch - GHAP_MEAN) / GHAP_STD
            ghap_target = (ghap_target - GHAP_MEAN) / GHAP_STD
            ghap_prev_patch = (ghap_prev_patch - GHAP_MEAN) / GHAP_STD

        rows = np.arange(r, r + self.patch_size, dtype=np.float32)
        cols = np.arange(c, c + self.patch_size, dtype=np.float32)
        lats = LAT_NORTH - rows * GHAP_RES
        lons = LON_WEST + cols * GHAP_RES
        lat_grid = (lats[:, None] - 30.0) / 42.0 * np.ones((1, self.patch_size), dtype=np.float32)
        lon_grid = (lons[None, :] + 25.0) / 70.0 * np.ones((self.patch_size, 1), dtype=np.float32)

        local_input = np.stack([ghap_patch, elev_h, lat_grid, lon_grid, ghap_prev_patch], axis=0)

        stn_pixels, stn_values, stn_count = self._get_stations_in_patch(
            year, day_t + horizon, r, c)

        if self.augment:
            if rng.random() < 0.5:
                local_input = local_input[:, ::-1, :].copy()
                ghap_target = ghap_target[::-1, :].copy()
                elev_h = elev_h[::-1, :].copy()
                stn_pixels[:stn_count, 0] = (self.patch_size - 1) - stn_pixels[:stn_count, 0]
            if rng.random() < 0.5:
                local_input = local_input[:, :, ::-1].copy()
                ghap_target = ghap_target[:, ::-1].copy()
                elev_h = elev_h[:, ::-1].copy()
                stn_pixels[:stn_count, 1] = (self.patch_size - 1) - stn_pixels[:stn_count, 1]

        patch_center_lat = LAT_NORTH - (r + self.patch_size / 2) * GHAP_RES
        patch_center_lon = LON_WEST + (c + self.patch_size / 2) * GHAP_RES
        era5_row = (ERA5_LAT_NORTH - patch_center_lat) / ERA5_RES
        era5_col = (patch_center_lon - ERA5_LON_WEST) / ERA5_RES
        er_i = int(np.clip(era5_row, 0, self._era5_h - 1))
        ec_i = int(np.clip(era5_col, 0, self._era5_w - 1))
        wind_u = float(era5_raw[0, er_i, ec_i])
        wind_v = float(era5_raw[1, er_i, ec_i])

        return {
            "era5": torch.from_numpy(era5),
            "elevation_coarse": torch.from_numpy(elev_c),
            "local_input": torch.from_numpy(local_input),
            "elevation_hires": torch.from_numpy(elev_h),
            "target": torch.from_numpy(ghap_target).unsqueeze(0),
            "lead_time": torch.tensor(float(horizon), dtype=torch.float32),
            "patch_center": torch.tensor([patch_center_lat, patch_center_lon], dtype=torch.float32),
            "wind_at_patch": torch.tensor([wind_u, wind_v], dtype=torch.float32),
            "station_pixels": torch.from_numpy(stn_pixels),
            "station_values": torch.from_numpy(stn_values),
            "station_count": torch.tensor(stn_count, dtype=torch.long),
            "meta": {"year": year, "day": day_t, "horizon": horizon, "row": r, "col": c},
        }


class CranPMDataModule(pl.LightningDataModule):

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config

    def setup(self, stage=None):
        data = self.cfg["data"]
        common = dict(
            era5_dir=data["era5_dir"],
            ghap_dir=data["ghap_dir"],
            elev_coarse_path=data["elev_coarse_path"],
            elev_hires_path=data["elev_hires_path"],
            cams_dir=data.get("cams_dir"),
            eea_zarr_path=data.get("eea_zarr_path"),
            horizons=data.get("horizons", HORIZONS),
            patch_size=data.get("patch_size", 512),
            normalize=data.get("normalize", True),
        )
        if stage in ("fit", None):
            self.train_ds = CranPMDataset(
                years=data["train_years"],
                augment=data.get("augment", False),
                hotspot_ratio=data.get("hotspot_ratio", 0.5),
                hotspot_power=data.get("hotspot_power", 1.0),
                **common,
            )
            self.val_ds = CranPMDataset(years=data["val_years"], **common)
        if stage in ("test", None):
            self.test_ds = CranPMDataset(years=data["test_years"], **common)

    def train_dataloader(self):
        nw = self.cfg["data"].get("num_workers", 0)
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg["train"]["batch_size"],
            shuffle=True, num_workers=nw,
            pin_memory=True, persistent_workers=(nw > 0), drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg["train"].get("val_batch_size", self.cfg["train"]["batch_size"]),
            shuffle=False, num_workers=self.cfg["data"].get("num_workers", 4), pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg["train"].get("val_batch_size", self.cfg["train"]["batch_size"]),
            shuffle=False, num_workers=self.cfg["data"].get("num_workers", 4), pin_memory=True,
        )
