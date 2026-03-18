"""
Layers and modules for the input embeddings.

Mathematical Notation:
    - B : batch size
    - L : sequence length
    - C : number of channels
    - d_model : embedding dimension
"""

# Import packages
from __future__ import annotations

import math
import torch
import torch.nn as nn
from einops import rearrange
from typing import List, Optional
from ephys_gpt.typing import Label


class SinusoidalPositionalEncoding(nn.Module):
    """
    Module for generating sinusoidal positional encodings for position embeddings.
    Implemented as in "Attention is All You Need" (Vaswani et al., 2017).

    Parameters
    ----------
    d_model : int
        The embedding dimension. Must be even.
    max_len : int
        The maximum expected sequence length. Default is 1000.
    """
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model

        # Validation
        if self.d_model % 2 != 0:
            raise ValueError(
                f"Embedding dimension (d_model) must be even, but got {d_model}."
            )

        # Precompute positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # shape: (max_len, 1)

        # Compute the scaling factor using log space for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as a buffer (moves to correct device but doesn't receive gradients)
        self.register_buffer("pe", pe, persistent=False)  # exclude buffer from `state_dict`

    def forward(self, x: torch.Tensor, start_index: Optional[int] = 0) -> torch.Tensor:
        """
        Retrieves the positional encoding for the given input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Shape should end with [..., seq_len, d_model].
        start_index : int
            Starting index for the positional encodings.

        Returns
        -------
        position_encodings : torch.Tensor
            Positional encodings broadcasted to the input shape.
        """
        # Get sequence length
        seq_len = x.size(-2)

        # Slice the precomputed positional encoding
        position_encodings = self.pe[start_index : start_index + seq_len, :]
        # shape: (seq_len, d_model)

        # Broadcast to exact input shape, but keep our own embedding dimension
        target_shape = list(x.shape)
        target_shape[-1] = self.d_model
        return position_encodings.expand(target_shape)


class LearnedPositionEmbedding(nn.Module):
    """
    Module for learning position embeddings.

    Parameters
    ----------
    d_model : int
        The embedding dimension.
    max_len : int
        The maximum expected sequence length.
    initializer : Optional[str]
        Initialization strategy. Must match the PyTorch naming convention.
        Defaults to 'xavier_uniform_'.
    """
    def __init__(
        self,
        d_model: int,
        max_len: int,
        initializer: str = "xavier_uniform_",
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Create a parameter tensor (uninitialized)
        self.position_embeddings = nn.Parameter(torch.empty(max_len, d_model))

        # Initialize the weights
        self._apply_initializer(initializer)

    def _apply_initializer(self, initializer_name: str):
        """
        Initializes the embedding weights based on the specified method.
        """
        try:
            init_fn = getattr(nn.init, initializer_name)
            init_fn(self.position_embeddings)
        except AttributeError:
            raise ValueError(f"PyTorch nn.init has no function named '{initializer_name}'.")

    def forward(self, x: torch.Tensor, start_index: Optional[int] = 0) -> torch.Tensor:
        """
        Applies learned position embeddings to the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Shape should end with [..., seq_len, d_model].
        start_index : int
            Starting index for the position embeddings.

        Returns
        -------
        position_embeddings : torch.Tensor
            The learned embeddings broacasted to the input shape.
        """
        # Get sequence length
        seq_len = x.size(-2)

        if start_index + seq_len > self.max_len:
            raise ValueError(
                f"Requested sequence length ({seq_len}) with start index ({start_index}) "
                f"exceeds the maximum length ({self.max_len}) configured for this layer."
            )

        # Slice the embeddings to match the current sequence length
        embeddings = self.position_embeddings[start_index : start_index + seq_len, :]
        # shape: (seq_len, d_model)

        # Broadcast to exact input shape, but keep our own embedding dimension
        target_shape = list(x.shape)
        target_shape[-1] = self.d_model
        return embeddings.expand(target_shape)


class ProjectEmbedding(nn.Module):
    """
    A clean helper module that wraps an embedding layer (Token, Positional, or Channel)
    and conditionally applies a linear projection if the dimensions differ.

    Parameters
    ----------
    base_module : nn.Module
        The base embedding module to wrap.
    in_dim : int
        The input dimension of the embeddings.
    out_dim : int
        The output dimension of the embeddings.
    """
    def __init__(
        self,
        base_module: nn.Module,
        in_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.base_module = base_module
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        # NOTE: If dimensions match, nn.Identity is used; it does nothing and has zero overhead.

    def forward(self, *args, **kwargs):
        return self.proj(self.base_module(*args, **kwargs))


class InputEmbeddingLayer(nn.Module):
    """
    Module for combining token, position, channel, and extra labels into a whole
    input embedding representation.

    Parameters
    ----------
    embedding_dim : int
        Dimension of the embedding space.
    n_tokens : int
        Number of tokens in the vocabulary.
    sequence_length : int
        Length of the input sequences.
    n_channels : int
        Number of channels in the input data.
    token_embedding_dim : Optional[int]
        Dimension of the token embeddings.
    pos_embedding_dim : Optional[int]
        Dimension of the position embeddings.
    pos_embedding_type : str
        Type of the position embeddings.
    channel_embedding_dim : Optional[int]
        Dimension of the channel embeddings.
    extra_label_specs : Optional[List[Label]]
        List of extra label specifications for the input data.
    pretrained_layer : Optional['InputEmbeddingLayer']
        A pretrained layer to initialize the embeddings from.
    """
    def __init__(
        self,
        embedding_dim: int,
        n_tokens: int,
        sequence_length: int,
        n_channels: int,
        token_embedding_dim: Optional[int] = None,
        pos_embedding_dim: Optional[int] = None,
        pos_embedding_type: Optional[str] = "absolute",
        channel_embedding_dim: Optional[int] = None,
        extra_label_specs: Optional[List[Label]] = None,
        pretrained_layer: Optional[InputEmbeddingLayer] = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_tokens = n_tokens
        self.sequence_length = sequence_length
        self.n_channels = n_channels
        self.pos_embedding_type = pos_embedding_type
        self.extra_label_specs = extra_label_specs or []

        # Resolve dimensions (fallback to embedding_dim if None)
        token_dim = token_embedding_dim or embedding_dim
        pos_dim = pos_embedding_dim or embedding_dim
        chan_dim = channel_embedding_dim or embedding_dim

        # Use pretrained layer (if provided)
        if pretrained_layer is not None:
            # In PyTorch, you can simply share the module references.
            # If you want them frozen, you should call .requires_grad_(False) on them externally.
            self.token_embed = pretrained_layer.token_embed
            self.pos_embed = pretrained_layer.pos_embed
            self.channel_embed = pretrained_layer.channel_embed

        # Initialize components for input embeddings
        else:
            # Initialize token embedding
            self.token_embed = ProjectEmbedding(
                base_module=nn.Embedding(n_tokens, token_dim),
                in_dim=token_dim,
                out_dim=embedding_dim,
            )

            # Initialize position embedding
            if pos_embedding_type == "sinusoidal":
                pos_embed = SinusoidalPositionalEncoding(d_model=pos_dim, max_len=sequence_length)
            elif pos_embedding_type == "absolute":
                pos_embed = LearnedPositionEmbedding(d_model=pos_dim, max_len=sequence_length)
            else:
                raise ValueError(f"Unknown pos_embedding_type: {pos_embedding_type}")

            self.pos_embed = ProjectEmbedding(pos_embed, in_dim=pos_dim, out_dim=embedding_dim)

            # Initialize channel embedding
            channel_embed = LearnedPositionEmbedding(d_model=chan_dim, max_len=n_channels)
            self.channel_embed = ProjectEmbedding(channel_embed, in_dim=chan_dim, out_dim=embedding_dim)
            # NOTE: channel embedding uses the exact same class but treats `n_channels` as
            #       the `sequence length`.

        # Initialize extra label embeddings
        self.extra_embeds = nn.ModuleList()
        for label in self.extra_label_specs:
            l_dim = label.label_dim or embedding_dim
            self.extra_embeds.append(
                ProjectEmbedding(
                    base_module=nn.Embedding(label.n_classes, l_dim),
                    in_dim=l_dim,
                    out_dim=embedding_dim,
                )
            )

    def forward(
        self,
        data: torch.Tensor,
        extra_labels: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass summing all embeddings.

        Parameters
        ----------
        data : torch.Tensor
            Input tensor of shape (B, L, C) containing token IDs.
        extra_labels : Optional[List[torch.Tensor]]
            List of tensors of shape (B, L + 1) containing extra label IDs.

        Returns
        -------
        embeddings : torch.Tensor
            Output embeddings of shape (B, L, C, d_model).
        """
        # Get token embeddings
        embeddings = self.token_embed(data)
        # shape: (B, L, C, d_model)

        # Get position embeddings
        if self.pos_embedding_type in ["absolute", "sinusoidal"]:
            # Our PositionalEncoding classes look at dim -2 to determine sequence length.
            # We use einops to swap seq_len (l) and n_channels (c) temporarily.
            embeds_for_pos = rearrange(embeddings, 'b l c d -> b c l d')

            # Retrieve position embeddings
            pos_embeddings = self.pos_embed(embeds_for_pos)  # embeds_for_pos not used in `forward()`
            pos_embeddings = rearrange(pos_embeddings, 'b c l d -> b l c d')
            embeddings = embeddings + pos_embeddings

        # Get channel embeddings
        chan_embeddings = self.channel_embed(embeddings)  # embeddings not used in `forward()`
        embeddings = embeddings + chan_embeddings

        # Get extra embeddings
        if extra_labels is not None:
            for label_tensor, extra_embed in zip(extra_labels, self.extra_embeds):
                # Slice off the last token
                label_tensor = label_tensor[:, :-1]  # shape: (B, L)

                # Get label embeddings
                label_embeddings = extra_embed(label_tensor)  # shape: (B, L, D)
                label_embeddings = label_embeddings.unsqueeze(2)  # shape: (B, L, 1, D)
                embeddings = embeddings + label_embeddings
                # NOTE: PyTorch will automatically broadcast this across the `C`
                #       dimension when added.

        return embeddings
