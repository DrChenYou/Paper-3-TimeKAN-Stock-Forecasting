import torch

from timekan.model import CascadedFrequencyDecomposition, TimeKAN


def test_decomposition_reconstructs_input():
    values = torch.randn(3, 60, 6)
    decomposition = CascadedFrequencyDecomposition(6, layers=4)
    bands = decomposition(values)
    assert len(bands) == 5
    assert torch.allclose(torch.stack(bands).sum(dim=0), values, atol=1e-5)


def test_timekan_forecast_shape_and_gradients():
    model = TimeKAN(d_model=16, attention_heads=4, maximum_window=9)
    values = torch.randn(2, 60, 6)
    forecast = model(values)
    assert forecast.shape == (2, 20, 1)
    forecast.mean().backward()
    assert model.forecast_head.weight.grad is not None
    gate_gradient = model.decomposition.filters[0].gate.weight.grad
    assert gate_gradient is not None
    assert torch.count_nonzero(gate_gradient) > 0
