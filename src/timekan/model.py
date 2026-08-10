"""PyTorch implementation of the TimeKAN architecture."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AdaptiveMovingAverage(nn.Module):
    """Sample-adaptive, normalized moving-average filter."""

    def __init__(self, features: int, minimum_window: int = 3, maximum_window: int = 15):
        super().__init__()
        if minimum_window <= 0 or maximum_window < minimum_window:
            raise ValueError("invalid adaptive-window bounds")
        self.features = features
        self.minimum_window = minimum_window
        self.maximum_window = maximum_window
        self.gate = nn.Linear(features, features)
        self.filter_logits = nn.Parameter(torch.zeros(maximum_window))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != self.features:
            raise ValueError("expected [batch, time, features]")
        batch, length, features = sequence.shape
        alpha = torch.sigmoid(self.gate(sequence.mean(dim=1))).mean(dim=-1)
        requested = torch.floor(alpha * length).to(torch.int64)
        requested = requested.clamp(self.minimum_window, min(self.maximum_window, length))
        outputs = []

        def apply_filter(values: torch.Tensor, width: int) -> torch.Tensor:
            weights = torch.softmax(self.filter_logits[:width], dim=0)
            kernel = weights.reshape(1, 1, width).repeat(features, 1, 1)
            left = (width - 1) // 2
            right = width - 1 - left
            padded = F.pad(values, (left, right), mode="replicate")
            return F.conv1d(padded, kernel, groups=features).transpose(1, 2)

        for index in range(batch):
            width = int(requested[index].item())
            if width % 2 == 0:
                width = width - 1 if width > self.minimum_window else min(width + 1, length)
            width = max(1, width)
            values = sequence[index : index + 1].transpose(1, 2)
            hard_output = apply_filter(values, width)

            # Preserve the paper's discrete window in the forward pass while using
            # a straight-through proxy so the data-dependent gate remains trainable.
            minimum = min(self.minimum_window, length)
            maximum = min(self.maximum_window, length)
            proxy = (1.0 - alpha[index]) * apply_filter(values, minimum)
            proxy = proxy + alpha[index] * apply_filter(values, maximum)
            outputs.append(hard_output + proxy - proxy.detach())
        return torch.cat(outputs, dim=0)


class CascadedFrequencyDecomposition(nn.Module):
    """Return one detail per layer plus the final residual trend."""

    def __init__(
        self,
        features: int,
        layers: int = 4,
        minimum_window: int = 3,
        maximum_window: int = 15,
    ):
        super().__init__()
        self.filters = nn.ModuleList(
            [
                AdaptiveMovingAverage(features, minimum_window, maximum_window)
                for _ in range(layers)
            ]
        )

    def forward(self, sequence: torch.Tensor) -> list[torch.Tensor]:
        residual = sequence
        bands = []
        for moving_average in self.filters:
            trend = moving_average(residual)
            bands.append(residual - trend)
            residual = trend
        bands.append(residual)
        return bands


class ChebyKANLinear(nn.Module):
    """Linear KAN layer using a learned Chebyshev expansion."""

    def __init__(self, input_dim: int, output_dim: int, order: int = 3):
        super().__init__()
        self.order = order
        self.coefficients = nn.Parameter(torch.empty(input_dim, output_dim, order + 1))
        nn.init.normal_(self.coefficients, mean=0.0, std=1.0 / (input_dim * (order + 1)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = torch.tanh(torch.tanh(values))
        basis = [torch.ones_like(normalized)]
        if self.order >= 1:
            basis.append(normalized)
        for _ in range(2, self.order + 1):
            basis.append(2.0 * normalized * basis[-1] - basis[-2])
        stacked = torch.stack(basis, dim=-1)
        return torch.einsum("...ik,iok->...o", stacked, self.coefficients)


class SharedMKANBlock(nn.Module):
    """Shared ChebyKAN and depthwise-convolution paths for every band."""

    def __init__(self, d_model: int, order: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.chebykan = ChebyKANLinear(d_model, d_model, order)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(sequence)
        kan_path = self.dropout(self.chebykan(normalized))
        conv_path = self.dropout(F.gelu(self.depthwise(normalized.transpose(1, 2))).transpose(1, 2))
        return self.output_norm(sequence + kan_path + conv_path)


class FrequencyMixingBlock(nn.Module):
    """Cross-frequency attention followed by adaptive band weighting."""

    def __init__(
        self,
        d_model: int,
        heads: int = 8,
        dropout: float = 0.1,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.attention = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.band_score = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, bands: torch.Tensor) -> torch.Tensor:
        if bands.ndim != 4:
            raise ValueError("expected [batch, bands, time, d_model]")
        mean = bands.mean(dim=2)
        standard_deviation = bands.std(dim=2, unbiased=False)
        maximum = bands.amax(dim=2)
        tokens, _ = self.attention(mean, mean, mean, need_weights=False)
        weights = torch.sigmoid(
            self.band_score(torch.cat((mean, standard_deviation, maximum), dim=-1))
        )
        fused = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-8)
        return self.norm(fused + self.head(fused))


class TimeKAN(nn.Module):
    """Adaptive frequency-decomposed KAN forecaster."""

    def __init__(
        self,
        input_dim: int = 6,
        forecast_length: int = 20,
        output_dim: int = 1,
        d_model: int = 64,
        decomposition_layers: int = 4,
        chebyshev_order: int = 3,
        attention_heads: int = 8,
        minimum_window: int = 3,
        maximum_window: int = 15,
        dropout: float = 0.1,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.forecast_length = forecast_length
        self.output_dim = output_dim
        self.decomposition = CascadedFrequencyDecomposition(
            input_dim, decomposition_layers, minimum_window, maximum_window
        )
        self.input_projection = nn.Linear(input_dim, d_model)
        self.shared_mkan = SharedMKANBlock(d_model, chebyshev_order, dropout)
        self.frequency_mixer = FrequencyMixingBlock(
            d_model, attention_heads, dropout, head_dropout
        )
        self.forecast_head = nn.Linear(d_model, forecast_length * output_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        bands = self.decomposition(sequence)
        processed = [self.shared_mkan(self.input_projection(band)) for band in bands]
        fused = self.frequency_mixer(torch.stack(processed, dim=1))
        forecast = self.forecast_head(fused)
        return forecast.reshape(sequence.shape[0], self.forecast_length, self.output_dim)
