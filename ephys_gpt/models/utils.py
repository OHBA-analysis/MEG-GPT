"""Basic layers and modules for the EphysGPT model."""

# Import packages
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


def _get_activation_fn(activation: str) -> nn.Module:
    """
    Returns the corresponding PyTorch activation function.

    Parameters
    ----------
    activation : str
        Name of the activation function.
    """
    activation = activation.lower()  # safeguard

    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "swish" or activation == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")


class ShiftTokenLayer(nn.Module):
    """
    Shifts the input tokens to create (input, target) pairs for
    teacher forcing.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Take all tokens except the last one
        input = x[:, :-1, :]  # (B, L, C)

        # Take all tokens except the first one
        target = x[:, 1:, :]  # (B, L, C)

        return input, target


class NormalizationLayer(nn.Module):
    """
    Creates a normalization layer that supports LayerNorm, BatchNorm1d, and GroupNorm.

    Note:
        - BatchNorm1d is also called the temporal batch normalization.

    Parameters
    ----------
    norm_type : str
        Type of normalization to perform.
    normalized_shape : Union[int, list, torch.Size]
        For LayerNorm, the input shape from an expected input.
        For BatchNorm1d, the number of features or channels in the input.
        For GroupNorm, the number of channels expected in the input.
    n_groups : Optional[int]
        Number of groups for group normalization.
        Required if norm_type is 'group'.
    """
    def __init__(
        self,
        norm_type: str,
        normalized_shape: Union[int, list, torch.Size],
        n_groups: Optional[int] = None,
    ):
        super().__init__()

        if norm_type == "layer":
            self.norm_layer = nn.LayerNorm(normalized_shape)
        elif norm_type == "batch":
            self.norm_layer = nn.BatchNorm1d(normalized_shape)
        elif norm_type == "group":
            if n_groups is None:
                raise ValueError("n_groups must be specified for GroupNorm.")
            self.norm_layer = nn.GroupNorm(n_groups, normalized_shape)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the normalization layer to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to normalize. The last dimension is expected to be
            the feature dimension.
        """
        # LayerNorm natively handles N-dimensional inputs correctly
        if isinstance(self.norm_layer, nn.LayerNorm):
            return self.norm_layer(x)

        # For BatchNorm1d and GroupNorm, we must isolate the feature dimension
        # NOTE: PyTorch BatchNorm/GroupNorm expects the feature dimension at index 1.
        permute_dims = [0, x.dim() - 1] + list(range(1, x.dim() - 1))
        x_permuted = x.permute(*permute_dims)

        if isinstance(self.norm_layer, nn.BatchNorm1d):  # BatchNorm1d expects 2D or 3D inputs
            original_shape = x_permuted.shape
            x_permuted = x_permuted.view(x_permuted.size(0), x_permuted.size(1), -1)
            # temporarily flatten all trailing spatial dimensions
            x_permuted = self.norm_layer(x_permuted)
            x_permuted = x_permuted.view(original_shape)

        else:  # GroupNorm expects at least 3D inputs
            x_permuted = self.norm_layer(x_permuted)

        # Reshape back to the original input shape
        inverse_permute_dims = [0] + list(range(2, x.dim())) + [1]
        return x_permuted.permute(*inverse_permute_dims)


class FeedForwardLayer(nn.Module):
    """
    Standard Transformer feed-forward network block.

    Parameters
    ----------
    model_dim : int
        Dimension of the input features.
    ff_dim : int
        Dimension of the feed-forward layer.
    dropout : float
        Dropout probability.
    activation : str
        Activation function to use.
    """
    def __init__(
        self,
        model_dim: int,
        ff_dim: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            _get_activation_fn(activation),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
