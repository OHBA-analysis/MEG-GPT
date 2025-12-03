"""Layers for learnable tokenization."""

# Import packages
import torch
import torch.nn as nn
from typing import Callable


class TokenWeightsLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "linear",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.dense_layer = nn.Linear(input_dim, output_dim)
        self.activation_fn = self._get_activation_fn(activation)
        self.norm_layer = nn.LayerNorm(output_dim)
        self.register_buffer("_temperature", torch.tensor(0.0))

    def _get_activation_fn(
        self, activation: str
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        if activation == "linear":
            return nn.Identity()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "softmax":
            return nn.Softmax(dim=-1)
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    @property
    def temperature(self) -> float:
        return float(self._temperature.item())
    
    @temperature.setter
    def temperature(self, value: float) -> None:
        self._temperature.fill_(value)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Project to token logits
        x = self.activation_fn(self.dense_layer(inputs))
        ell = self.activation_fn(x)

        # Normalize and scale
        ell = self.norm_layer(ell) / 0.1
        # ell.shape: (batch_size * n_channels, sequence_length, n_tokens)

        if self.training:
            # Get hard one-hot samples
            theta_sample = torch.argmax(ell, dim=2)
            theta_sample = nn.functional.one_hot(
                theta_sample, num_classes=self.output_dim
            ).float()

            # Get soft softmax weights
            theta_weight = nn.functional.softmax(ell, dim=2)
            # shape: (batch_size * n_channels, sequence_length, n_tokens)

            # Perform annealing
            token_weight = (
                self.temperature * theta_weight + (1 - self.temperature) * theta_sample
            )
            # shape: (batch_size * n_channels, sequence_length, n_tokens)

        else:
            # If not training, use hard argmax one-hot
            token_weight = nn.functional.one_hot(
                torch.argmax(ell, dim=2), num_classes=self.output_dim
            ).float()

        return token_weight
