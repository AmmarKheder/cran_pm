# CRAN-PM

**Cross-Resolution Attention Network for PM₂.₅ forecasting.**

**[Project Website](https://ammarkheder.github.io/cran_pm/)**

Trained on Europe (2017–2021), CRAN-PM downscales coarse ERA5/CAMS reanalysis to 0.01° (~1 km) PM₂.₅ predictions using a two-branch transformer architecture with wind-guided cross-attention.

## Architecture

- **Global branch** — ERA5 + CAMS at 0.25° → 735 tokens (dim 768, depth 8)
- **Local branch** — GHAP PM₂.₅ at 0.01°, 512×512 patches → 1024 tokens (dim 512, depth 6)
- **Cross-attention bridge** — local tokens query global context with wind-direction alignment bias
- **CNN decoder** — PixelShuffle progressive upsampling 32→512 with skip connection
- **Delta prediction** — output = PM₂.₅(t) + Δ, zero-initialized decoder

## Data

| Source | Resolution | Variables |
|--------|-----------|-----------|
| ERA5 | 0.25° | u10, v10, t2m, msl, sp + 5 pressure levels |
| CAMS EAC4 | 0.1° | NO₂, O₃, SO₂, CO, PM₁₀ |
| GHAP | 0.01° | PM₂.₅ (target) |
| GMTED2010 | ~250m | Elevation |

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python scripts/train.py --config configs/default.yaml
```

Edit `configs/default.yaml` to set your data paths before training.

