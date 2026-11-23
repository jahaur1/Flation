"""End-to-end training for ConstrainedE2EFlation.

The model is a single-stage forecaster: frequency-domain cross-channel interaction,
time-domain target-history anchoring and regime gating (TimeGate), a straight-through
nonnegativity floor, and physical-space dual balance loss (MPE^2 + Pbias^2).

All five watershed sites are trained with seed 2024.

Usage:
    python train.py
    python train.py --sites 1015 708 --device cpu
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import prepare_data
from src.model import Model

SEED = 2024
BEST_SEEDS: dict[int, int] = {site: SEED for site in (1015, 708, 703, 693, 626)}
MPE_WEIGHT = 0.04
PBIAS_WEIGHT = 0.25


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_base_model(device: torch.device) -> Model:
    return Model(
        context_window=24,
        target_window=24,
        ablation=0,
        mlp_drop=0.3,
        mlp_hidden=64,
        cf_dim=48,
        cf_depth=4,
        cf_heads=6,
        cf_mlp=128,
        cf_head_dim=32,
        cf_drop=0.2,
        c_in=6,
        patch_len=6,
        stride=6,
        d_model=64,
    ).to(device)


def series_descriptors(seq: torch.Tensor) -> torch.Tensor:
    """Compute 6 summary statistics (mean, std, last, delta, min, max) along time."""
    mean = seq.mean(dim=-1, keepdim=True)
    std = seq.std(dim=-1, unbiased=False, keepdim=True)
    last = seq[:, -1:]
    first = seq[:, :1]
    s_min = seq.min(dim=-1, keepdim=True).values
    s_max = seq.max(dim=-1, keepdim=True).values
    return torch.cat((mean, std, last, last - first, s_min, s_max), dim=-1)


class ConstrainedE2EFlation(nn.Module):
    """Single-stage Flation with time-domain regime gating (TimeGate) and an STE zero-flow floor."""

    def __init__(
        self,
        base_model: nn.Module,
        history: int = 24,
        horizon: int = 24,
        zero_flow_floor: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.register_buffer("zero_flow_floor", torch.tensor(float(zero_flow_floor), dtype=torch.float32))

        self.anchor_linear = nn.Linear(history, horizon)
        self.anchor_mlp = nn.Sequential(
            nn.Linear(history, 64),
            nn.GELU(),
            nn.Linear(64, horizon),
        )
        nn.init.zeros_(self.anchor_mlp[-1].weight)
        nn.init.zeros_(self.anchor_mlp[-1].bias)

        # Gate inputs: target_history (24) + target descriptors (6) + forcing descriptors (6)
        self.gate_net = nn.Sequential(
            nn.Linear(history + 12, 64),
            nn.GELU(),
            nn.Linear(64, horizon),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

    def compute_gate(self, z: torch.Tensor) -> torch.Tensor:
        target_history = z[:, :, self.base.target_index]
        forcing_history = z[:, :, 0]
        gate_in = torch.cat(
            (target_history, series_descriptors(target_history), series_descriptors(forcing_history)),
            dim=-1,
        )
        return torch.sigmoid(self.gate_net(gate_in)).unsqueeze(1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        enhanced, fused, target_history = self.base.encode_ccia(z)
        spectrum = self.base.CIAA(enhanced[:, 1:, :, :], fused)
        y_ccia = self.base.decode_ccia(spectrum)
        y_anchor = (self.anchor_linear(target_history) + self.anchor_mlp(target_history)).unsqueeze(1)
        gate = self.compute_gate(z)
        y_raw = y_anchor + gate * (y_ccia - y_anchor)

        floor = self.zero_flow_floor.to(dtype=y_raw.dtype, device=y_raw.device)
        y_clamped = torch.clamp(y_raw, min=floor)
        return y_raw + (y_clamped - y_raw).detach()


def strict_zero_floor(scaler: StandardScaler) -> float:
    """Compute standardized zero-flow boundary safely above float32 underflow."""
    raw = np.float32(-float(scaler.mean_[0]) / float(scaler.scale_[0]))
    return float(np.nextafter(raw, np.float32(np.inf)))


def compute_loss(
    output_b_h_1: torch.Tensor,
    target_b_h: torch.Tensor,
    scaler: StandardScaler,
    mpe_weight: float = MPE_WEIGHT,
    pbias_weight: float = PBIAS_WEIGHT,
) -> torch.Tensor:
    """Compute standardized log1p MSE plus physical-space MPE^2 and Pbias^2 penalties."""
    target_b_h_1 = target_b_h.unsqueeze(2)
    mse_std = F.mse_loss(output_b_h_1, target_b_h_1)

    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])
    pred_phys = torch.expm1(torch.clamp(output_b_h_1.double() * scale + mean, max=12.0))
    true_phys = torch.expm1(target_b_h_1.double() * scale + mean)

    rel_err = (true_phys - pred_phys) / torch.clamp_min(true_phys, 1e-6)
    pbias_ratio = (pred_phys - true_phys).sum() / torch.clamp_min(true_phys.sum(), 1e-6)
    loss_extra = mpe_weight * rel_err.mean().square() + pbias_weight * pbias_ratio.square()
    return mse_std + loss_extra.to(dtype=mse_std.dtype)


def run_epoch(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    scaler: StandardScaler,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    model.train(optimizer is not None)
    tot_loss = 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            output = model(x.float().to(device)).permute(0, 2, 1)
            loss = compute_loss(output, y.float().to(device), scaler)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            tot_loss += float(loss.item())
    return tot_loss / len(loader)


def train_loop(
    model: ConstrainedE2EFlation,
    loaders: tuple,
    ckpt_path: Path,
    loss_csv: Path,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    lr_patience: int,
) -> int:
    train_loader, val_loader, _, q_scaler = loaders
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=lr_patience)
    best_val, best_epoch, stalled = float("inf"), 0, 0
    history = []

    for ep in range(epochs):
        first_loss = run_epoch(train_loader, model, device, q_scaler, optimizer)
        tr_loss = run_epoch(train_loader, model, device, q_scaler, optimizer)
        val_loss = run_epoch(val_loader, model, device, q_scaler, None)
        if not np.isfinite([tr_loss, val_loss]).all():
            raise ValueError(f"Training produced nonfinite loss at epoch {ep}")

        history.append(
            dict(epoch=ep, first_pass_loss=first_loss, train_loss=tr_loss, val_loss=val_loss)
        )
        pd.DataFrame(history).to_csv(loss_csv, index=False)

        if ep == 0 or val_loss < best_val:
            best_val, best_epoch, stalled = val_loss, ep, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            stalled += 1
        if stalled >= patience:
            break
        scheduler.step(val_loss)

    return best_epoch


def train_site(
    site: int,
    seed: int,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    patience: int,
    lr: float,
    lr_patience: int,
    batch_size: int,
) -> int:
    seed_everything(seed)
    data_path = ROOT / "data" / f"WTS{site}_1.csv"
    loaders = prepare_data(str(data_path), seed, scaling="train", batch_size=batch_size)
    next(iter(loaders[0]))
    model = ConstrainedE2EFlation(
        build_base_model(device),
        zero_flow_floor=strict_zero_floor(loaders[3]),
    ).to(device)
    run_dir = output_dir / f"site_{site}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_epoch = train_loop(
        model=model,
        loaders=loaders,
        ckpt_path=run_dir / "best_model.pth",
        loss_csv=run_dir / "loss.csv",
        device=device,
        epochs=epochs,
        patience=patience,
        lr=lr,
        lr_patience=lr_patience,
    )
    print(f"[Site {site} | Seed {seed}] best_epoch={best_epoch} saved to {run_dir / 'best_model.pth'}", flush=True)
    return best_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConstrainedE2EFlation from scratch with seed 2024.")
    parser.add_argument("--sites", type=int, nargs="+", default=list(BEST_SEEDS), choices=list(BEST_SEEDS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(selected_device if args.device == "auto" else args.device)
    torch.set_num_threads(4)
    print(f"Training seed={SEED} on device={device}", flush=True)
    for site in args.sites:
        train_site(
            site=site,
            seed=BEST_SEEDS[site],
            device=device,
            output_dir=output_dir,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            lr_patience=args.lr_patience,
            batch_size=args.batch_size,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
