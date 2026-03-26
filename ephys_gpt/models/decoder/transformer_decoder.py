"""
Layers and modules for the transformer decoder block.

Mathematical Notation:
    - B : batch size
    - L : sequence length
    - C : number of channels
    - D : model dimension
    - E : embedding dimension
"""

# Import packages
import logging
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from typing import List, Optional

from ephys_gpt.models.decoder.attention import MultiHeadGASPAttention
from ephys_gpt.models.utils import NormalizationLayer, FeedForwardLayer


_logger = logging.getLogger(__name__)


class DecoderLayer(nn.Module):
    """
    A single layer of the EphysGPT Transformer Decoder, combining
    Channel Attention, Time Attention, and Cross Attention.
    """
    def __init__(
        self,
        n_heads: int,
        model_dim: int,
        n_channels: int,
        n_patches_out: int,
        patch_len_out: int,
        n_patches_in: int,
        patch_len_in: int,
        unpatched_len_in: int,
        l_unpatched_b: int,
        l_patched_b: int,
        do_chan_attention: bool,
        do_cross_attention: bool,
        chan_attention_mask: np.ndarray,
        chan_attn_chandim: int,
        feed_forward_dim: int,
        feed_forward_activation: str,
        dropout: float,
        norm_type: str,
        n_groups: int = None,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.do_chan_attention = do_chan_attention
        self.do_cross_attention = do_cross_attention
        self.seq_len_out = n_patches_out * patch_len_out
        self.seq_len_in = n_patches_in * patch_len_in

        # ------------------------
        # Channel Attention Branch
        # ------------------------
        if do_chan_attention:
            self.chan_attention_dropout = nn.Dropout(dropout)
            self.norm_chan1 = NormalizationLayer(norm_type, model_dim, n_groups)

            # Project channel dimension if specified
            if chan_attn_chandim is not None:
                self.chan_att_dim_proj = nn.Linear(n_channels, chan_attn_chandim)
            else:
                self.chan_att_dim_proj = nn.Identity()
            # NOTE: This layer is shared across all heads and layers.

            self.gasp_chan_attention = MultiHeadGASPAttention(
                n_heads=n_heads,
                model_dim=model_dim,
                l_other=self.seq_len_in,
                n_patches_out=n_channels,
                patch_len_out=1,
                n_patches_in=chan_attn_chandim if chan_attn_chandim is not None else n_channels,
                patch_len_in=1,
                unpatched_len_in=0,
                causal=False,
                l_unpatched_b=None,
                l_patched_b=None,
                attention_mask=chan_attention_mask,
            )

            self.norm_chan2 = NormalizationLayer(norm_type, model_dim, n_groups)
            self.ff_chan = FeedForwardLayer(model_dim, feed_forward_dim, dropout, feed_forward_activation)
        else:
            self.register_module("gasp_chan_attention", None)

        # ---------------------
        # Time Attention Branch
        # ---------------------
        self.time_attention_dropout = nn.Dropout(dropout)
        self.norm_time1 = NormalizationLayer(norm_type, model_dim, n_groups)

        n_patches_out_time = n_patches_in if do_cross_attention else n_patches_out
        patch_len_out_time = patch_len_in if do_cross_attention else patch_len_out

        self.gasp_time_attention = MultiHeadGASPAttention(
            n_heads=n_heads,
            model_dim=model_dim,
            l_other=n_channels,
            n_patches_out=n_patches_out_time,
            patch_len_out=patch_len_out_time,
            n_patches_in=n_patches_in,
            patch_len_in=patch_len_in,
            unpatched_len_in=unpatched_len_in,
            causal=True,
            l_unpatched_b=l_unpatched_b,
            l_patched_b=l_patched_b,
            attention_mask=None,
        )

        # ------------------------------------------------------------
        # Cross Attention Branch (Time queries -> Channel keys/values)
        # ------------------------------------------------------------
        if do_chan_attention and do_cross_attention:
            self.gasp_cross_attention = MultiHeadGASPAttention(
                n_heads=n_heads,
                model_dim=model_dim,
                l_other=n_channels,
                n_patches_out=n_patches_out,
                patch_len_out=patch_len_out,
                n_patches_in=n_patches_in,
                patch_len_in=patch_len_in,
                unpatched_len_in=0,  # fixed not to use unpatching
                causal=True,
                l_unpatched_b=l_unpatched_b,
                l_patched_b=l_patched_b,
                attention_mask=None,
            )
        else:
            self.register_module("gasp_cross_attention", None)

        # ------------------
        # Final Feed-Forward
        # ------------------
        self.norm_time2 = NormalizationLayer(norm_type, model_dim, n_groups)
        self.ff_time = FeedForwardLayer(model_dim, feed_forward_dim, dropout, feed_forward_activation)

    def forward(self, x: torch.Tensor, chan_attention_weight: float = 1.0) -> torch.Tensor:
        """
        Forward pass for the decoder layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, C, D).
        chan_attention_weight : float
            Multiplier for stochastic channel attention dropout (0.0 or 1.0).
            Multiplier for stochastic channel attention dropout (0.0 or 1 / (1 - p) during training, 1.0 during eval).
        """
        # Apply channel attention
        if self.do_chan_attention:
            # Transpose dimensions for channel attention
            xt = rearrange(x, "b l c d -> b c l d")
            # shape: (B, C, L, D)

            xt_residual = xt  # for residual connection

            xt = self.norm_chan1(xt)  # normalization

            # Apply dense projection over the channel dimension if required
            xt_dim = rearrange(xt, "b c l d -> b l d c")
            xt_dim = self.chan_att_dim_proj(xt_dim)
            xt_dim = rearrange(xt_dim, "b l d c_d -> b c_d l d")
            # shape: (B, {C, chan_attn_chandim}, L, D)

            # Channel attention (query=xt, key=xt_dim, value=xt_dim)
            xt = self.gasp_chan_attention(xt, xt_dim, xt_dim)
            xt = self.chan_attention_dropout(xt)
            # shape: (B, C, L, D)

            # Add residual (weighted for stochastic depth; see TransformerDecoder class)
            xt = (chan_attention_weight * xt) + xt_residual

            # Pass through the feed-forward layer
            x_chan = rearrange(xt, "b c l d -> b l c d")  # swap back to standard format
            x_residual = x_chan  # for residual connection

            x_chan = self.norm_chan2(x_chan)  # normalization
            x_chan = self.ff_chan(x_chan)
            x_chan = (chan_attention_weight * x_chan) + x_residual

            if not self.do_cross_attention:
                x = x_chan

        # Apply time attention
        x_residual = x
        x_time = self.norm_time1(x)

        x_time = self.gasp_time_attention(x_time, x_time, x_time)
        x_time = self.time_attention_dropout(x_time)

        out_len = x_time.size(1)
        x_time = x_time + x_residual[:, -out_len:, :, :]
        # NOTE: because sequence length might have shrunk, slice from the end.

        # Apply cross attention
        if self.do_cross_attention and self.do_chan_attention:
            x_residual = x_time

            x_time = self.gasp_cross_attention(query=x_time, key=x_chan, value=x_chan)

            out_len = x_time.size(1)
            x_time = x_time + x_residual[:, -out_len:, :, :]

        x = x_time

        # Apply final feed-forward layer
        x_residual = x
        x = self.norm_time2(x)
        x = self.ff_time(x)
        x = x + x_residual

        return x


class TransformerDecoder(nn.Module):
    """
    Transformer decoder used in the EphysGPT model.
    Stacks multiple DecoderLayers.
    """
    def __init__(
        self,
        n_heads: int,
        model_dim: int,
        embedding_dim: int,
        n_channels: int,
        n_patches_out: List[int],
        patch_len_out: List[int],
        n_patches_in: List[int],
        patch_len_in: List[int],
        unpatched_len_in: List[int],
        l_unpatched_b: List[Optional[int]],
        l_patched_b: List[Optional[int]],
        do_chan_attention: List[bool],
        do_cross_attention: List[bool],
        chan_attention_mask: List[Optional[np.ndarray]],
        chan_attn_chandim: Optional[int],
        full_channel_attention_dropout: float,
        feed_forward_dim: int,
        feed_forward_activation: str,
        dropout: float,
        norm_type: str,
        n_groups: Optional[int] = None,
    ):
        super().__init__()
        self.n_layers = len(n_patches_out)
        self.model_dim = model_dim
        self.full_channel_attention_dropout = full_channel_attention_dropout

        # Initialize input dropout and projection layers
        if model_dim != embedding_dim:
            self.input_projection = nn.Linear(embedding_dim, model_dim)
        else:
            self.input_projection = nn.Identity()
        self.input_dropout = nn.Dropout(dropout)

        # Build decoder layers
        self.layers = nn.ModuleList()
        for n in range(self.n_layers):
            curr_in_len = n_patches_in[n] * patch_len_in[n]
            curr_out_len = n_patches_out[n] * patch_len_out[n]

            _logger.info(f"Initializing Decoder Layer {n}. Sequence length: {curr_in_len} -> {curr_out_len}")

            # Validation check
            if n > 0:
                prev_out_len = n_patches_out[n - 1] * patch_len_out[n - 1]
                if prev_out_len != curr_in_len:
                    raise ValueError(
                        f"Layer {n} input length mismatch: Previous output length "
                        f"({prev_out_len}) != current input length ({curr_in_len})."
                    )

            layer = DecoderLayer(
                n_heads=n_heads,
                model_dim=model_dim,
                n_channels=n_channels,
                n_patches_out=n_patches_out[n],
                patch_len_out=patch_len_out[n],
                n_patches_in=n_patches_in[n],
                patch_len_in=patch_len_in[n],
                unpatched_len_in=unpatched_len_in[n],
                l_unpatched_b=l_unpatched_b[n],
                l_patched_b=l_patched_b[n],
                do_chan_attention=do_chan_attention[n],
                do_cross_attention=do_cross_attention[n],
                chan_attention_mask=chan_attention_mask[n],
                chan_attn_chandim=chan_attn_chandim,
                feed_forward_dim=feed_forward_dim,
                feed_forward_activation=feed_forward_activation,
                dropout=dropout,
                norm_type=norm_type,
                n_groups=n_groups,
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Transformer decoder with random channel attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, C, E).

        Returns
        -------
        x : torch.Tensor
            Output tensor of shape (B, L_out, C, D).
        """
        # Apply input projection and dropout layers
        x = self.input_projection(x)
        x = self.input_dropout(x)
        # shape: (B, L, C, D)

        # Employ stochastic depth for channel attention (per batch)
        chan_weight = 1.0
        if self.training and self.full_channel_attention_dropout > 0.0:
            # If a random float is less than the dropout threshold, drop the channel branch
            if torch.rand(1).item() < self.full_channel_attention_dropout:
                chan_weight = 0.0
            else:
                # Scale up by 1 / (1 - p) to maintain expected value
                chan_weight = 1.0 / (1.0 - self.full_channel_attention_dropout)
                # NOTE: Inverted dropout applied to compensate for the fact that more signals
                #       will be active during evaluation. This keeps the training and evaluation
                #       distributions aligned.
                #       E[logit_train] = (1 - q) * 0 + q * (x / q) = x = E[logit_eval]
                #       where q is 1 - dropout_rate.
                # NOTE: This is done automatically by the Dropout layer in PyTorch. It might be
                #       worth using the Dropout layer instead of this manual implementation.

        for layer in self.layers:
            x = layer(x, chan_attention_weight=chan_weight)

        return x
