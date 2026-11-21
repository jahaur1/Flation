from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ["Rain", "Day_of_month", "Day_of_week", "Hour", "Week", "Qrate"]


def prepare_data(path: str, seed: int, scaling: str = 'legacy', batch_size: int = 32) -> tuple:
    """Keep legacy checkpoints reproducible; use scaling='train' for new experiments."""
    if scaling not in {'legacy', 'train'}:
        raise ValueError("scaling must be 'legacy' or 'train'")
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    frame = pd.read_csv(path, usecols=["Date", "Qrate", "Rain"])
    frame["Date"] = pd.to_datetime(frame["Date"])
    if frame.Date.duplicated().any() or not frame.Date.is_monotonic_increasing:
        raise ValueError("Dates must be unique and ordered")
    if not np.isfinite(frame[["Qrate", "Rain"]].to_numpy()).all():
        raise ValueError("Qrate and Rain must contain finite values")
    if (frame[["Qrate", "Rain"]] < 0).any().any():
        raise ValueError("Qrate and Rain must be nonnegative")
    if len(frame) <= 17547:
        raise ValueError("Insufficient rows for the original split")
    frame["Day_of_month"] = frame.Date.dt.day
    frame["Day_of_week"] = frame.Date.dt.dayofweek
    frame["Hour"] = frame.Date.dt.hour
    frame["Week"] = frame.Date.dt.isocalendar().week.astype(int)
    scalers = {}
    for column in ["Qrate", "Rain"]:
        scaler = StandardScaler()
        logged = np.log1p(frame[[column]])
        fit_part = logged.iloc[:-17500] if scaling == 'train' else logged
        scaler.fit(fit_part)
        frame[column] = scaler.transform(logged)
        scalers[column] = scaler
    frame = frame.set_index("Date")[FEATURES]
    loaders = []
    for index, part in enumerate([frame.iloc[:-17500], frame.iloc[-17500:-8750], frame.iloc[-8750:]]):
        windows = np.lib.stride_tricks.sliding_window_view(part.to_numpy(dtype=np.float64), 48, axis=0)
        x = np.ascontiguousarray(windows[:, :, :24].transpose(0, 2, 1))
        y = np.ascontiguousarray(windows[:, -1, 24:])
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=index == 0, num_workers=0))
    return (*loaders, scalers["Qrate"])
