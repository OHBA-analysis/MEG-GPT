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
    elif activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2)
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


class ChannelDropoutLayer(nn.Module):
    """
    Drops entire channel entries from a tensor during training (per-sample).

    Unlike element-wise nn.Dropout, this module zeroes all features for
    randomly selected channels. Each sample in the batch receives an
    independent binary mask. Inverted dropout scaling (1 / (1 - p)) is
    applied so that the expected magnitude is preserved at the evaluation time.

    Parameters
    ----------
    p : float
        Probability of dropping a channel.
    channel_dim : int
        Dimension index of the channel axis in the input tensor.
    """
    def __init__(self, p: float, channel_dim: int) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability p must be in [0, 1), got {p}.")
        self.p = p
        self.channel_dim = channel_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x

        keep_prob = 1.0 - self.p

        # Build a mask that broadcasts over all dims except batch and channel dim
        mask_shape = [1] * x.ndim
        mask_shape[0] = x.size(0)
        mask_shape[self.channel_dim] = x.size(self.channel_dim)
        mask = torch.bernoulli(
            torch.full(mask_shape, keep_prob, device=x.device, dtype=x.dtype)
        )
        return x * mask / keep_prob  # apply inverted dropout scaling

    def extra_repr(self) -> str:
        return f"p={self.p}, channel_dim={self.channel_dim}"
