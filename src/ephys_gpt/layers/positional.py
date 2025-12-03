"""Positional encoding layers for EphysGPT models."""

# Import packages
import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Layer for generating sinusoidal positional encoding for position embeddings.
    Implemented as in "Attention is All You Need" (Vaswani et al., 2017), and partly
    adpated from the `huggingface/transformers` library.

    Parameters
    ----------
    embedding_dim : int
        Dimension of the input embeddings.
    max_sequence_length : int
        Maximum length of the sequence to compute positional encodings for.
    """
    def __init__(
        self,
        embedding_dim: int,
        max_sequence_length: int,
    ):
        super().__init__()
        self.max_sequence_length = max_sequence_length

        if embedding_dim % 2 != 0:
            raise ValueError("embedding dimension must be even for sinusoidal positional encoding.")
        
        # Precompute the positional encoding matrix
        pos_encoding = torch.zeros(max_sequence_length, embedding_dim)
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        # position.shape = (max_sequence_length, 1)
        denominator = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * -(math.log(10000.0) / embedding_dim)
        )
        # denominator.shape = (embedding_dim/2,)

        pos_encoding[:, 0::2] = torch.sin(position * denominator)
        pos_encoding[:, 1::2] = torch.cos(position * denominator)

        # Add a batch dimension
        self.pos_encoding = pos_encoding.unsqueeze(0).unsqueeze(0)
        # pos_encoding.shape = (1, 1, max_sequence_length, embedding_dim)

        # Register as a buffer
        self.register_buffer("pos_encoding", self.pos_encoding)
        # NOTE: This makes it a part of the module's state and moves with .to(device),
        #       but not a trainable parameter

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Gets the positional encoding for the input tensor.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch_size, n_channels, sequence_length, embedding_dim).
        """
        sequence_length = inputs.shape[-2]
        if sequence_length == self.max_sequence_length:
            return self.pos_encoding.to(inputs.dtype)
        else:
            return self.pos_encoding[:, :, :sequence_length, :].to(inputs.dtype)
