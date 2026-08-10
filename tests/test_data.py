import numpy as np
import pandas as pd

from timekan.data import FEATURE_COLUMNS, prepare_datasets


def test_chronological_targets_remain_in_partitions():
    frame = pd.DataFrame({name: np.arange(200, dtype=float) + i for i, name in enumerate(FEATURE_COLUMNS)})
    datasets = prepare_datasets(frame, input_length=20, forecast_length=5)
    assert datasets.train.target_starts.max() + 5 <= 140
    assert datasets.validation.target_starts.min() >= 140
    assert datasets.validation.target_starts.max() + 5 <= 170
    assert datasets.test.target_starts.min() >= 170
    inputs, targets = datasets.test[0]
    assert inputs.shape == (20, 6)
    assert targets.shape == (5, 1)


def test_scaler_is_fit_on_training_rows_only():
    frame = pd.DataFrame({name: np.arange(100, dtype=float) for name in FEATURE_COLUMNS})
    datasets = prepare_datasets(frame, input_length=10, forecast_length=5)
    assert datasets.scaler.mean_[0] == np.mean(np.arange(70, dtype=float))
