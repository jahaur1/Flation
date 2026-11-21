# ConstrainedE2EFlation: Physically Constrained Runoff Forecasting

Official implementation and reproducibility package for **ConstrainedE2EFlation**, a single-stage end-to-end runoff forecasting model combining frequency-domain cross-channel interaction (`Flation`), time-domain target-history anchoring and regime gating (`TimeGate`), a differentiable straight-through estimator (`STE`) nonnegativity floor, and physical-space dual balance alignment (`MPE² + Pbias²`).

## Repository Structure

```text
.
├── LICENSE                          # MIT License
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
├── reproduce_main_results.py        # Single-script entry point to reproduce main results on all 5 datasets
├── checkpoints/                     # Best-seed pretrained weights for each watershed site
│   ├── site_1015/seed_2023/best_model.pth
│   ├── site_708/seed_2024/best_model.pth
│   ├── site_703/seed_2025/best_model.pth
│   ├── site_693/seed_2025/best_model.pth
│   └── site_626/seed_2023/best_model.pth
├── data/                            # 5 hourly watershed runoff & rainfall datasets
│   ├── WTS1015_1.csv
│   ├── WTS708_1.csv
│   ├── WTS703_1.csv
│   ├── WTS693_1.csv
│   └── WTS626_1.csv
├── reference/
│   └── expected_main_results.csv    # Reference metrics for automated verification
└── src/                             # Core model, data pipeline, and evaluation metrics
    ├── __init__.py
    ├── cross_transformer.py
    ├── data.py
    ├── metrics.py
    ├── model.py
    └── patch_layers.py
```

## Environment & Installation

- **Python**: 3.8+ (verified on Python 3.11 / PyTorch 2.11.0+cu128 and Python 3.8+ / PyTorch 2.4.1+)
- **Hardware**: Supports both CPU and NVIDIA CUDA GPUs (CUDA recommended for training from scratch)

### Setup

```bash
git clone https://github.com/jahaur1/Flation.git
cd Flation

python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for GPU users**: If `pip install torch` installs a CPU-only wheel on your platform, install the CUDA-enabled PyTorch build matching your driver from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

## Reproducing Main Results

All main experimental results across the 5 watershed datasets (`1015`, `708`, `703`, `693`, `626`) can be reproduced directly with a single script using each dataset's best random seed:

### 1. Evaluate Provided Best-Seed Checkpoints (Fast, ~5s on GPU)

```bash
python reproduce_main_results.py
```

This loads the pretrained checkpoints under `checkpoints/`, runs inference on the test split of all 5 datasets, verifies the metrics against `reference/expected_main_results.csv`, and writes:
- `outputs/main_results_summary.csv`
- `outputs/site_<site>/seed_<seed>/metrics.csv`
- `outputs/site_<site>/seed_<seed>/window24.csv` (24-step ground truth and forecasts)

### 2. Train from Scratch and Evaluate

```bash
python reproduce_main_results.py --mode train
```

To run on CPU or on a subset of sites:

```bash
python reproduce_main_results.py --device cpu
python reproduce_main_results.py --sites 1015 708
```

## Main Experimental Results (Best Seed per Dataset)

| Site | Best Seed | MAE | RMSE | NSE | R² | RSR | Pbias (%) | MPE (%) | Negative Forecasts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1015 | 2023 | 0.022072 | 0.071178 | 0.889620 | 0.889742 | 0.332235 | +0.1871% | -3.7859% | 0 |
| 708 | 2024 | 0.110791 | 0.334033 | 0.725340 | 0.725787 | 0.524080 | -3.1518% | +6.5205% | 0 |
| 703 | 2025 | 0.458257 | 1.510269 | 0.477016 | 0.479525 | 0.723176 | -11.5871% | -14.0336% | 0 |
| 693 | 2025 | 0.209677 | 0.751272 | 0.725367 | 0.729513 | 0.524054 | -11.8091% | +32.6313% | 0 |
| 626 | 2023 | 0.081940 | 0.284835 | 0.387506 | 0.394231 | 0.782620 | -20.6618% | +27.7717% | 0 |

## Protocol Summary

- **Task**: 24-step history $\rightarrow$ 24-step horizon hourly runoff forecasting (`c_in = 6`: `Rain`, `Day_of_month`, `Day_of_week`, `Hour`, `Week`, `Qrate`).
- **Data Split & Leak-Free Scaling**: Last 8,750 rows for testing, preceding 8,750 rows for validation, and all earlier rows for training. `log1p` + `StandardScaler` is fit strictly on the training split.
- **Training Configuration**: Two-pass Adam (`lr=1e-3`, `ReduceLROnPlateau` with `factor=0.3`, `patience=3`), `batch_size=128`, `epochs=30`, early stopping `patience=10`, gradient clipping at `5.0`, and physical dual balance loss weights `mpe_weight=0.04`, `pbias_weight=0.25`.

## License

This project is released under the [MIT License](LICENSE).
