"""Evaluate ConstrainedE2EFlation on all 5 watershed datasets.

Provided checkpoints use seed 2024. Pass --mode train to retrain with train.py and then evaluate.

Usage:
    python reproduce_main_results.py                  # Evaluate checkpoints and print metrics only
    python reproduce_main_results.py --mode train     # Train from scratch with seed 2024 and evaluate
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import prepare_data
from src.metrics import METRICS, evaluate
from train import (
    BEST_SEEDS,
    ConstrainedE2EFlation,
    build_base_model,
    seed_everything,
    strict_zero_floor,
    train_loop,
)


def predict_loader(
    loader: DataLoader,
    model: ConstrainedE2EFlation,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.float().to(device)).cpu().numpy().astype(np.float64))
            targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def run_site(
    site: int,
    seed: int,
    mode: str,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    seed_everything(seed)
    data_path = ROOT / "data" / f"WTS{site}_1.csv"
    loaders = prepare_data(str(data_path), seed, scaling="train", batch_size=args.batch_size)
    next(iter(loaders[0]))

    q_scaler = loaders[3]
    model = ConstrainedE2EFlation(
        build_base_model(device),
        zero_flow_floor=strict_zero_floor(q_scaler),
    ).to(device)

    ckpt_path = ROOT / "checkpoints" / f"site_{site}" / f"seed_{seed}" / "best_model.pth"

    t0 = time.time()
    best_epoch = -1
    if mode == "evaluate":
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True), strict=True)
    else:
        run_dir = output_dir / f"site_{site}" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        trained_ckpt = run_dir / "best_model.pth"
        best_epoch = train_loop(
            model=model,
            loaders=loaders,
            ckpt_path=trained_ckpt,
            loss_csv=run_dir / "loss.csv",
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            lr_patience=args.lr_patience,
        )
        model.load_state_dict(torch.load(trained_ckpt, map_location=device, weights_only=True), strict=True)

    pred_std, targ_std = predict_loader(loaders[2], model, device)
    metrics, truth, forecast = evaluate(pred_std, targ_std, q_scaler)
    elapsed = round(time.time() - t0, 1)

    neg_count = int(np.count_nonzero(forecast < 0))
    rec = dict(
        site=site,
        seed=seed,
        best_epoch=best_epoch,
        elapsed_s=elapsed,
        neg_count=neg_count,
        min_forecast=float(np.min(forecast)),
        **{k: metrics[k] for k in METRICS},
    )

    if mode != "evaluate":
        columns = [f"Truth_t{i}" for i in range(1, 25)] + [f"Forecast_t{i}" for i in range(1, 25)]
        pd.DataFrame(np.concatenate([truth, forecast], axis=1), columns=columns).to_csv(
            run_dir / "window24.csv", index=False
        )
        pd.DataFrame([rec]).to_csv(run_dir / "metrics.csv", index=False)

    print(
        f"[Site {site} | Seed {seed}] ({elapsed}s) | "
        f"MAE={rec['MAE']:.6f}  RMSE={rec['RMSE']:.6f}  NSE={rec['NSE']:.6f}  "
        f"R2={rec['R2']:.6f}  RSR={rec['RSR']:.6f}  Pbias={rec['Pbias']:+.4f}%  "
        f"MPE={rec['MPE']:+.4f}%  neg={neg_count}",
        flush=True,
    )
    return rec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ConstrainedE2EFlation on the 5 watershed datasets using seed 2024."
    )
    parser.add_argument(
        "--mode",
        choices=["evaluate", "train"],
        default="evaluate",
        help="'evaluate' loads the provided seed-2024 checkpoints (default); 'train' retrains from scratch.",
    )
    parser.add_argument(
        "--sites",
        type=int,
        nargs="+",
        default=list(BEST_SEEDS),
        choices=list(BEST_SEEDS),
        help="Subset of watershed sites to run (default: all 5 sites).",
    )
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
    if args.mode != "evaluate":
        output_dir.mkdir(parents=True, exist_ok=True)

    selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(selected_device if args.device == "auto" else args.device)
    torch.set_num_threads(4)
    print(f"Running mode={args.mode} on device={device}", flush=True)

    records = []
    for site in args.sites:
        seed = BEST_SEEDS[site]
        rec = run_site(site, seed, args.mode, device, args, output_dir)
        records.append(rec)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(records)[["site", "seed"] + METRICS + ["neg_count", "min_forecast"]]

    print("\n=== Main Results Summary ===")
    print(summary_df.to_string(index=False))
    if args.mode != "evaluate":
        summary_path = output_dir / "main_results_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")

    ref_path = ROOT / "reference" / "expected_main_results.csv"
    if args.mode == "evaluate" and ref_path.is_file():
        ref_df = pd.read_csv(ref_path)
        merged = summary_df.merge(ref_df, on=["site", "seed"], suffixes=("_got", "_ref"))
        max_diff = max(
            float((merged[f"{m}_got"] - merged[f"{m}_ref"]).abs().max()) for m in METRICS
        )
        print(f"Max absolute metric difference vs reference: {max_diff:.3e}")


if __name__ == "__main__":
    main()
