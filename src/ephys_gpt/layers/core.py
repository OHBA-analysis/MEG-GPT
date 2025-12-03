"""Core layers for EphysGPT models."""

# Import packages
import torch
import torch.nn as nn
from typing import Optional, Union


def rnn_layer(
    rnn_type: str,
    input_size: int,
    hidden_size: int,
) -> nn.Module:
    """
    Creates an RNN layer.

    Parameters
    ----------
    rnn_type : str
        Type of an RNN layer. Options include 'gru' and 'lstm'.
    input_size : int
        The number of expected features in the input.
    hidden_size : int
        The number of features in the hidden state.

    Returns
    -------
    rnn_module : nn.Module
        The RNN layer.
    """
    if rnn_type == "gru":
        rnn_module = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
    elif rnn_type == "lstm":
        rnn_module = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
    else:
        raise ValueError(f"Unknown RNN type: {rnn_type}")
    return rnn_module


class NormalizationLayer(nn.Module):
    """
    Creates a normalization layer that supports LayerNorm, BatchNorm1d, and GroupNorm.

    BatchNorm1d is called the temporal batch normalization.

    Parameters
    ----------
    norm_type : str
        Type of normalization layer. Options include 'layer', 'batch', and 'group'.
    normalized_shape : Union[int, list, torch.Size]
        For LayerNorm, the input shape from an expected input.
        For BatchNorm1d, the number of features or channels of the input.
        For GroupNorm, the number of channels expected in the input.
    n_groups : Optional[int]
        Number of groups to separate the channels into. Required for GroupNorm.
    """
    def __init__(
        self,
        norm_type: str = "layer",
        normalized_shape: Union[int, list, torch.Size] = None,
        n_groups: Optional[int] = None,
    ):
        super().__init__()

        # Validate inputs
        if normalized_shape is None:
            raise ValueError("normalized_shape must be provided.")
        
        if norm_type == "layer":
            self.norm_layer = nn.LayerNorm(normalized_shape)
        elif norm_type == "batch":
            self.norm_layer = nn.BatchNorm1d(normalized_shape)
        elif norm_type == "group":
            if n_groups is None:
                raise ValueError("n_groups must be provided for GroupNorm.")
            self.norm_layer = nn.GroupNorm(n_groups, normalized_shape)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm_layer(inputs)
