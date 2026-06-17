# The two models, explained

[← Back to the main README](../README.md)

The repository ships only **Apache-2.0-licensed** weather models. Other Earth2Studio models exist (Pangu-Weather, GraphCast, FuXi, etc.) but carry research-only or non-commercial licenses that are incompatible with publishing this repo as a permissive sample. If you need them in your own fork, double-check each model's license against your use case first.

## DLWP — Deep Learning Weather Prediction

| Property | Value |
|---|---|
| Origin | University of Washington (Karlbauer et al., 2023) |
| License | Apache-2.0 |
| Architecture | Convolutional neural network on a **cubed-sphere** grid (six face-projections of Earth onto a cube; convolutions wrap across face boundaries) |
| Native resolution | **1.0°** (≈ 111 km at equator) |
| Variables | Limited dynamic set — primarily `t2m`, `z500` |
| Time step | 6 hours |
| GPU memory | ~4 GB |
| Instance | `ml.g5.2xlarge` (NVIDIA A10G) |
| Forecast wall time (24 h lead) | **~10 seconds** |

**When to use DLWP**: rapid prototyping, baseline runs, smoke-testing pipelines, and education. It's a small, fast model — ideal for *"I just want a global temperature map for tomorrow"* without paying for a Blackwell GPU.

## FCN3 — FourCastNet v3

| Property | Value |
|---|---|
| Origin | NVIDIA (Pathak et al., 2024) |
| License | Apache-2.0 |
| Architecture | **Spherical Fourier Neural Operator** with a probabilistic ensemble head (uses the spherical-harmonic transform via `torch-harmonics` to handle the sphere properly, unlike rectangular CNNs) |
| Native resolution | **0.25°** (≈ 28 km at equator), 721×1440 grid |
| Variables | **72** — surface fields plus a full 3-D atmosphere on 13 pressure levels |
| Time step | 6 hours |
| GPU memory | ~80 GB (needs a Blackwell-class GPU) |
| Instance | `ml.g7e.2xlarge` (1× RTX PRO 6000 Blackwell, 96 GB VRAM) |
| Forecast wall time (24 h lead) | **~30 seconds** |

**When to use FCN3**: this is the production model. 0.25° matches operational NWP centers; the 72-variable output covers surface fields plus a full 3-D atmosphere on 13 pressure levels, suitable for deriving thousands of downstream products (jet streams, cyclone tracking, energy generation forecasts, agricultural risk, etc.). FCN3 needs a custom container because:

1. PyTorch 2.6 (the latest SageMaker inference DLC) doesn't support Blackwell GPUs (compute capability `sm_120`).
2. FCN3's `torch-harmonics` extension must be **CUDA-compiled** for both Ada Lovelace (`sm_89`) and Blackwell (`sm_120`) to get the fast path.
3. Stock SageMaker inference DLCs ≥ 2.7 don't exist (TorchServe is in maintenance mode), so we ship two **bring-your-own-container** (BYOC) variants:
   - **`container_fcn3/`** (NGC base, pre-compiled CUDA `torch-harmonics`) — fastest, requires NGC API key.
   - **`container_fcn3_dlc/`** (AWS Training DLC base, PyPI `torch-harmonics`) — no NGC key, slightly slower in float32 fallback.

Both BYOC variants use Flask + gunicorn on port 8080 to implement the SageMaker `/ping` + `/invocations` contract.

---

## Further reading

If this is your first contact with AI-based weather forecasting, the following are worth bookmarking:

- **NVIDIA Earth2Studio**: [github.com/NVIDIA/earth2studio](https://github.com/NVIDIA/earth2studio) — the framework this project sits on top of, with model loaders, data sources (GFS, ERA5, ARCO, IFS), IO backends, and ensemble runners.
- **NVIDIA Modulus / PhysicsNeMo**: [github.com/NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo) — the underlying scientific-ML toolkit that supplies neural operators and data loaders.
- **NOAA GFS**: [www.nco.ncep.noaa.gov/pmb/products/gfs](https://www.nco.ncep.noaa.gov/pmb/products/gfs/) — the freely-available analysis data this project uses for initial conditions, updated every 6 hours.
- **ECMWF ERA5 reanalysis**: [www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) — the 40-year archive most ML weather models are trained on.
- **WeatherBench 2**: [sites.research.google/weatherbench](https://sites.research.google/weatherbench/) — the standard benchmark for ML weather models against IFS HRES.
- **ECMWF AIFS**: [www.ecmwf.int/en/about/media-centre/aifs-blog](https://www.ecmwf.int/en/about/media-centre/aifs-blog) — ECMWF's own ML-based forecast model, now operational.

The original AI-weather papers, in rough chronological order:

- **FourCastNet** (NVIDIA, 2022): [arxiv.org/abs/2202.11214](https://arxiv.org/abs/2202.11214)
- **Pangu-Weather** (Huawei, 2022): [www.nature.com/articles/s41586-023-06185-3](https://www.nature.com/articles/s41586-023-06185-3)
- **GraphCast** (DeepMind, 2023): [www.science.org/doi/10.1126/science.adi2336](https://www.science.org/doi/10.1126/science.adi2336)
- **FuXi** (Fudan, 2023): [www.nature.com/articles/s41612-023-00512-1](https://www.nature.com/articles/s41612-023-00512-1)
