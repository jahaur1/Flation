from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

METRICS = ["MAE", "RMSE", "NSE", "R2", "RSR", "Pbias", "MPE"]


def bias_metrics(forecast: np.ndarray, truth: np.ndarray) -> dict:
    forecast = np.asarray(forecast, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if forecast.shape != truth.shape or truth.size == 0:
        raise ValueError("Predictions and observations must have the same nonempty shape")
    if not np.isfinite(forecast).all() or not np.isfinite(truth).all():
        raise ValueError("Predictions and observations must be finite")
    if (truth < 0).any():
        raise ValueError("Observed runoff must be nonnegative")
    total = float(np.sum(truth, dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Aggregate Pbias requires a positive finite observed sum")
    error = forecast - truth
    mean_error = float(np.mean(error))
    mae = float(np.mean(np.abs(error)))
    if abs(mean_error) > mae + 1e-12 * max(1., mae):
        raise ValueError("Mean error exceeds MAE")
    zero_count = int(np.count_nonzero(truth == 0))
    mpe = float(-100 * np.mean(error / truth)) if zero_count == 0 else float("nan")
    return dict(Pbias=float(100 * np.sum(error, dtype=np.float64) / total),
                MPE=mpe, ME=mean_error, Mean_truth=float(np.mean(truth)),
                zero_truth_count=zero_count)


def evaluate(prediction: np.ndarray, target: np.ndarray, scaler: StandardScaler) -> tuple:
    prediction = prediction.astype(np.float64).reshape(-1, 24)
    target = target.astype(np.float64).reshape(-1, 24)
    forecast = np.expm1(scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(prediction.shape))
    truth = np.expm1(scaler.inverse_transform(target.reshape(-1, 1)).reshape(target.shape))
    diff = truth - forecast
    rmse = np.sqrt(np.mean(np.square(diff)))
    denominator = np.sum(np.square(truth - truth.mean())) * np.sum(np.square(forecast - forecast.mean()))
    metrics = {
        "MAE": np.mean(np.abs(diff)),
        "RMSE": rmse,
        "NSE": 1 - np.sum(np.square(diff)) / np.sum(np.square(truth - truth.mean())),
        "R2": np.square(np.sum((truth - truth.mean()) * (forecast - forecast.mean()))) / denominator if denominator else np.nan,
        "RSR": rmse / np.std(truth),
        **bias_metrics(forecast, truth),
    }
    return ({key: float(value) for key, value in metrics.items()}, truth, forecast)
