"""Market CSV validation and chronological window construction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

FEATURE_COLUMNS = ("Open", "Close", "High", "Low", "Volume", "Change")


def _parse_number(value) -> float:
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().replace(",", "")
    multiplier = 1.0
    match = re.fullmatch(r"([+-]?[0-9]*\.?[0-9]+)([KMB])?%?", text, re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot parse numeric market value: {value!r}")
    if match.group(2):
        multiplier = {"K": 1e3, "M": 1e6, "B": 1e9}[match.group(2).upper()]
    return float(match.group(1)) * multiplier


def read_market_csv(path: str | Path, features=FEATURE_COLUMNS) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
        frame = frame.sort_values("Date", kind="stable")
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    frame = frame.copy()
    for column in features:
        frame[column] = frame[column].map(_parse_number)
    if not np.isfinite(frame[list(features)].to_numpy(dtype=float)).all():
        raise ValueError("feature columns contain non-finite values")
    return frame.reset_index(drop=True)


class TimeSeriesWindows(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        target_index: int,
        input_length: int,
        forecast_length: int,
        target_starts: np.ndarray,
    ):
        self.features = np.asarray(features, dtype=np.float32)
        self.target_index = target_index
        self.input_length = input_length
        self.forecast_length = forecast_length
        self.target_starts = np.asarray(target_starts, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.target_starts)

    def __getitem__(self, index: int):
        start = int(self.target_starts[index])
        inputs = self.features[start - self.input_length : start]
        targets = self.features[start : start + self.forecast_length, self.target_index : self.target_index + 1]
        return torch.from_numpy(inputs), torch.from_numpy(targets)


@dataclass(frozen=True)
class PreparedDatasets:
    train: TimeSeriesWindows
    validation: TimeSeriesWindows
    test: TimeSeriesWindows
    scaler: StandardScaler
    target_index: int


def _valid_target_starts(
    partition_start: int,
    partition_end: int,
    input_length: int,
    forecast_length: int,
) -> np.ndarray:
    first = max(input_length, partition_start)
    stop = partition_end - forecast_length + 1
    return np.arange(first, max(first, stop), dtype=np.int64)


def prepare_datasets(
    frame: pd.DataFrame,
    *,
    features=FEATURE_COLUMNS,
    target: str = "Close",
    input_length: int = 60,
    forecast_length: int = 20,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> PreparedDatasets:
    values = frame[list(features)].to_numpy(dtype=np.float64)
    total = len(values)
    train_end = int(total * train_fraction)
    validation_end = train_end + int(total * validation_fraction)
    if train_end < input_length + forecast_length or validation_end >= total:
        raise ValueError("dataset is too short for the requested windows and split")
    scaler = StandardScaler().fit(values[:train_end])
    scaled = scaler.transform(values).astype(np.float32)
    target_index = list(features).index(target)
    common = (scaled, target_index, input_length, forecast_length)
    train = TimeSeriesWindows(
        *common,
        _valid_target_starts(0, train_end, input_length, forecast_length),
    )
    validation = TimeSeriesWindows(
        *common,
        _valid_target_starts(train_end, validation_end, input_length, forecast_length),
    )
    test = TimeSeriesWindows(
        *common,
        _valid_target_starts(validation_end, total, input_length, forecast_length),
    )
    return PreparedDatasets(train, validation, test, scaler, target_index)
