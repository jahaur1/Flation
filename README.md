# Flation

## Repository Structure

```text
.
├── LICENSE                          # MIT License
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
├── reproduce_main_results.py        # Single-script entry point to reproduce main results on all 5 datasets
├── checkpoints/                     # Seed-2024 pretrained weights for each watershed site
│   ├── site_1015/seed_2024/best_model.pth
│   ├── site_708/seed_2024/best_model.pth
│   ├── site_703/seed_2024/best_model.pth
│   ├── site_693/seed_2024/best_model.pth
│   └── site_626/seed_2024/best_model.pth
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

All main experimental results across the 5 watershed datasets (`1015`, `708`, `703`, `693`, `626`) can be reproduced directly with a single script using seed 2024:

### 1. Evaluate Provided Seed-2024 Checkpoints (Fast, ~5s on GPU)

```bash
python reproduce_main_results.py
```

This loads the pretrained checkpoints under `checkpoints/` and runs inference on the test split of all 5 datasets. Results are printed to the terminal only and are not saved to files.

### 2. Train from Scratch and Evaluate

```bash
python reproduce_main_results.py --mode train
```

To run on CPU or on a subset of sites:

```bash
python reproduce_main_results.py --device cpu
python reproduce_main_results.py --sites 1015 708
```

## License

This project is released under the [MIT License](LICENSE).
