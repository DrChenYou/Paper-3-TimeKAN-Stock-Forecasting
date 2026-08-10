import pytest

from timekan.metrics import regression_metrics


def test_regression_metrics():
    result = regression_metrics([1, 2, 3], [1, 2, 3])
    assert result["rmse"] == 0
    assert result["mae"] == 0
    assert result["mape_percent"] == 0
    assert result["r2"] == 1
    with pytest.raises(ValueError):
        regression_metrics([1, 2], [1])
