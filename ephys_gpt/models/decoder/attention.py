"""
Attentional building blocks of the transformer decoder.

Mathematical Notation:
    - B : batch size
    - H : number of heads
"""

# Import packages
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Union


_logger = logging.getLogger(__name__)


class Attention(nn.Module):
    """
    Standard scaled dot-product attention layer.

    The mathematical operation performed is:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Parameters
    ----------
    key_dim : int
        Dimension of the key/query vectors.
    """
    def __init__(self, key_dim: int):
        super().__init__()
        self.key_dim = key_dim

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the attention mechanism.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor of shape (B, H, l_out, l_other, key_dim).
        k : torch.Tensor
            Key tensor of shape (B, H, l_in, l_other, key_dim).
        v : torch.Tensor
            Value tensor of shape (B, H, l_in, l_other, key_dim).
        mask : Optional[torch.Tensor]
            Boolean attention mask of shape (l_out, l_in). True indicates elements to be masked.

        Returns
        -------
        output : torch.Tensor
            Output tensor of shape (B, H, l_out, l_other, key_dim).
        """
        # Rearrange for attention over the time/sequence dimension
        q = rearrange(q, "b h l_out l_other k_d -> b h l_other l_out k_d")

        # Transpose k for matrix multiplication
        k = rearrange(k, "b h l_in l_other k_d -> b h l_other k_d l_in")

        # Rearrange for attention
        v = rearrange(v, "b h l_in l_other k_d -> b h l_other l_in k_d")

        # Compute scaled dot-product attention
        attn = (q @ k) / math.sqrt(self.key_dim)
        # shape: (B, H, l_other, l_out, l_in)

        if mask is not None:
            # PyTorch broadcasts the mask (l_out, l_in) to match attn shape
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)  # normalize with softmax

        # Replace NaNs with 0 if there are fully-masked rows
        attn = torch.nan_to_num(attn, nan=0.0)
        # NOTE: This can happen when l_out > l_in and the mask fully masks out some query positions.
        #       In that case, softmax returns NaN for those positions, which we replace with 0 to 
        #       avoid propagating NaNs through the output.

        # Apply attention weights to values
        output = attn @ v
        # shape: (B, H, l_other, l_out, key_dim)

        # Rearrange back to original dimensions
        output = rearrange(output, "b h l_other l_out k_d -> b h l_out l_other k_d")

        return output


class GASPAttention(nn.Module):
    """
    GASP attention layer that computes the attention mask with patching and sparse
    banding incorporated.

    With sparse banding, attention is focused on a limited context window,
    allowing for more efficient computation. The sparse banding window sizes
    for patch and unpatch are:
        - Patched window: l_patched_b * 2 + 1
        - Unpatched window: l_unpatched_b * 2 + 1

    Parameters
    ----------
    n_heads : int
        Number of attention heads.
    key_dim : int
        Key dimension per head.
    l_other : int
        Dimensionality of the 'other' sequence element.
    n_patches_out : int
        Number of patches to output.
    patch_len_out : int
        Length of output patches.
    n_patches_in : int
        Number of patches in the input.
    patch_len_in : int
        Length of input patches.
    unpatched_len_in : int
        Number of unpatched elements to attend to.
    causal : bool
        Whether to enforce causality (autoregressive masking).
    l_unpatched_b : Optional[int]
        Unpatched bandwidth limit of sparse banding.
        If None, no banding is applied.
    l_patched_b : Optional[int]
        Patched bandwidth limit of sparse banding.
        If None, no banding is applied.
    attention_mask : Optional[Union[np.ndarray, torch.Tensor]]
        Initial attention mask to use or build upon.
    """
    def __init__(
        self,
        n_heads: int,
        key_dim: int,
        l_other: int,
        n_patches_out: int,
        patch_len_out: int,
        n_patches_in: int,
        patch_len_in: int,
        unpatched_len_in: int,
        causal: bool,
        l_unpatched_b: Optional[int] = None,
        l_patched_b: Optional[int] = None,
        attention_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.key_dim = key_dim
        self.l_other = l_other
        self.n_patches_out = n_patches_out
        self.patch_len_out = patch_len_out
        self.n_patches_in = n_patches_in
        self.patch_len_in = patch_len_in
        self.unpatched_len_in = unpatched_len_in
        self.causal = causal
        self.l_unpatched_b = l_unpatched_b
        self.l_patched_b = l_patched_b

        self.l_in = self.n_patches_in * self.patch_len_in
        self.l_out = self.n_patches_out * self.patch_len_out

        # Register mask as a buffer so it automatically moves to the correct device (CPU/GPU)
        mask = self._compute_attention_mask(attention_mask)
        self.register_buffer("attention_mask", mask)

        self.attention_layer = Attention(key_dim)

    def _compute_attention_mask(
        self,
        base_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Computes the boolean mask for attention (row: query; column: key).
        Note that `True` means the position WILL be masked (ignored).
        """
        # Initialize mask as a boolean tensor
        if base_mask is None:
            mask = torch.zeros(
                (self.l_out, self.n_patches_in + self.unpatched_len_in), dtype=torch.bool
            )
        else:
            if isinstance(base_mask, np.ndarray):
                mask = torch.from_numpy(base_mask).bool()
            else:
                mask = base_mask.bool()

        # Incorporate sparse banding on unpatched tokens
        if self.l_unpatched_b is not None:
            for i in range(self.n_patches_in, self.n_patches_in + self.unpatched_len_in):
                for j in range(self.l_out):
                    lower = j - self.l_unpatched_b
                    upper = j + self.l_unpatched_b
                    if i < lower or i > upper:
                        mask[j, i] = True

        # Incorporate sparse banding on patched tokens
        if self.l_patched_b is not None:
            for i in range(self.n_patches_in):
                for j in range(self.l_out):
                    lower = j // self.patch_len_in - self.l_patched_b
                    upper = j // self.patch_len_in + self.l_patched_b
                    if i < lower or i > upper:
                        mask[j, i] = True

        if self.causal:
            # Patch masking
            for i in range(self.n_patches_in):
                m_idx = (i + 1 - self.n_patches_in) * self.patch_len_in + self.l_out - 1
                # (i + 1 - self.n_patches_in): distance of the current patch i from the end of the patch sequence
                # self.patch_len_in: converts patch-level distance to actual sequence time steps
                # self.l_out - 1: last index of the output sequence
                m_idx = max(0, m_idx)
                mask[:m_idx, i] = True

            # Unpatched masking
            for i in range(self.n_patches_in, self.n_patches_in + self.unpatched_len_in):
                end_idx = self.l_out - (self.n_patches_in + self.unpatched_len_in) + i
                mask[:end_idx, i] = True

        return mask

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the GASP attention layer.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor of shape (B, H, l_out, l_other, key_dim).
        k : torch.Tensor
            Key tensor of shape (B, H, l_in, l_other, key_dim).
        v : torch.Tensor
            Value tensor of shape (B, H, l_in, l_other, key_dim).

        Returns
        -------
        output : torch.Tensor
            Output tensor of shape (B, H, l_out, l_other, key_dim).
        """
        return self.attention_layer(q, k, v, mask=self.attention_mask)


class MultiHeadGASPAttention(nn.Module):
    """
    Multi-head GASP attention layer.

    Parameters
    ----------
    model_dim : int
        Model dimension per head.
    *args : Any
        See GASPAttention for descriptions of other parameters.
    """
    def __init__(
        self,
        n_heads: int,
        model_dim: int,
        l_other: int,
        n_patches_out: int,
        patch_len_out: int,
        n_patches_in: int,
        patch_len_in: int,
        unpatched_len_in: int,
        causal: bool,
        l_unpatched_b: Optional[int] = None,
        l_patched_b: Optional[int] = None,
        attention_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.model_dim = model_dim
        self.key_dim = model_dim // n_heads
        self.l_other = l_other
        self.n_patches_out = n_patches_out
        self.patch_len_out = patch_len_out
        self.n_patches_in = n_patches_in
        self.patch_len_in = patch_len_in
        self.unpatched_len_in = unpatched_len_in

        self.l_in = self.n_patches_in * self.patch_len_in
        self.l_out = self.n_patches_out * self.patch_len_out

        _logger.info(f"Initialized MultiHeadGASPAttention layer.")

        # Patch projection
        if patch_len_in > 1:
            in_features = self.n_heads * self.patch_len_in
            self.patch_projection_k = nn.Linear(in_features, self.n_heads)
            self.patch_projection_v = nn.Linear(in_features, self.n_heads)
        else:
            self.patch_projection_k = nn.Identity()
            self.patch_projection_v = nn.Identity()

        # Input projections
        self.patched_projection_k = nn.Linear(self.model_dim, self.model_dim)
        self.patched_projection_v = nn.Linear(self.model_dim, self.model_dim)

        if unpatched_len_in > 0:
            self.unpatched_projection_k = nn.Linear(self.model_dim, self.model_dim)
            self.unpatched_projection_v = nn.Linear(self.model_dim, self.model_dim)
        else:
            self.unpatched_projection_k = nn.Identity()
            self.unpatched_projection_v = nn.Identity()

        self.query_projection = nn.Linear(self.model_dim, self.model_dim)
        self.output_projection = nn.Linear(self.model_dim, self.model_dim)

        # Initialize GASP layer for time and channel attention
        self.gasp_layer = GASPAttention(
            n_heads=n_heads,
            key_dim=self.key_dim,
            l_other=l_other,
            n_patches_out=n_patches_out,
            patch_len_out=patch_len_out,
            n_patches_in=n_patches_in,
            patch_len_in=patch_len_in,
            unpatched_len_in=unpatched_len_in,
            causal=causal,
            l_unpatched_b=l_unpatched_b,
            l_patched_b=l_patched_b,
            attention_mask=attention_mask,
        )

    def _patch_inputs(
        self,
        x: torch.Tensor,
        projection_layer: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits and projects input into patched and unpatched segments.
        """
        # --- Process Patched Part ---
        # Get l_in elements from the start and reshape into patches
        x_patched = x[:, :self.l_in, :, :]

        # Rearrange patch dimensions
        patched_x = rearrange(
            x_patched,
            "b (p_in pl_in) l_other d -> b p_in l_other d pl_in",
            p_in=self.n_patches_in,
            pl_in=self.patch_len_in,
        )

        # Reshape to isolate dimensions for the projection layer
        patched_x = rearrange(
            patched_x,
            "b p_in l_other (h k_d) pl_in -> b p_in l_other k_d (h pl_in)",
            h=self.n_heads,
        )

        # Apply projections
        if self.patch_len_in > 1:
            patched_x = projection_layer(patched_x)

        # Merge key dimension and heads back to model dimension
        patched_x = rearrange(
            patched_x, 
            "b p_in l_other k_d h -> b p_in l_other (h k_d)",
        )

        # --- Process Unpatched Part ---
        if self.unpatched_len_in > 0:
            unpatched_x = x[:, (self.l_in - self.unpatched_len_in):, :, :]
        else:
            unpatched_x = torch.empty(0, device=x.device)
        
        return patched_x, unpatched_x

    def _perceiver_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gets the last l_out elements of the sequence.
        Always assumes that l_in >= l_out in the configuration.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """
        if x.size(1) > self.l_out:  # ignores beginning tokens
            return x[:, -self.l_out:, :, :]
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the multi-head attention mechanism.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape (B, l_out, l_other, model_dim).
        key : torch.Tensor
            Key tensor of shape (B, l_in, l_other, model_dim).
        value : torch.Tensor
            Value tensor of shape (B, l_in, l_other, model_dim).

        Returns
        -------
        output : torch.Tensor
            Output tensor of shape (B, l_out, l_other, model_dim).
        """
        # Apply patching to keys
        patched_k, unpatched_k = self._patch_inputs(key, self.patch_projection_k)

        k_patched = self.patched_projection_k(patched_k)
        # shape: (B, n_patches_in, l_other, model_dim)

        if self.unpatched_len_in > 0:
            k_unpatched = self.unpatched_projection_k(unpatched_k)
            # shape: (B, unpatched_len_in, l_other, model_dim)
            k = torch.cat([k_patched, k_unpatched], dim=1)
            # shape: (B, n_patches_in + unpatched_len_in, l_other, model_dim)
        else:
            k = k_patched

        # Apply patching to values
        patched_v, unpatched_v = self._patch_inputs(value, self.patch_projection_v)

        v_patched = self.patched_projection_v(patched_v)
        # shape: (B, n_patches_in, l_other, model_dim)

        if self.unpatched_len_in > 0:
            v_unpatched = self.unpatched_projection_v(unpatched_v)
            # shape: (B, unpatched_len_in, l_other, model_dim)
            v = torch.cat([v_patched, v_unpatched], dim=1)
            # shape: (B, n_patches_in + unpatched_len_in, l_other, model_dim)
        else:
            v = v_patched

        # Apply projections to Q
        perceiver_q = self._perceiver_x(query)
        q = self.query_projection(perceiver_q)
        # shape: (B, l_out, l_other, model_dim)

        # Split heads
        q = rearrange(q, "b l_out l_other (h k_d) -> b h l_out l_other k_d", h=self.n_heads)
        k = rearrange(k, "b l_in l_other (h k_d) -> b h l_in l_other k_d", h=self.n_heads)
        v = rearrange(v, "b l_in l_other (h k_d) -> b h l_in l_other k_d", h=self.n_heads)

        # Pass through attention layer
        output = self.gasp_layer(q, k, v)
        # shape: (B, H, l_out, l_other, key_dim)

        # Combine heads and apply output projection
        output = rearrange(output, "b h l_out l_other k_d -> b l_out l_other (h k_d)")
        output = self.output_projection(output)
        # shape: (B, l_out, l_other, model_dim)

        return output


if __name__ == "__main__":

    def _plot_attention_mask(gasp_layer: GASPAttention):
        """
        Visualizes the attention mask.
        """
        # Get attention mask (move to cpu)
        mask = (~gasp_layer.attention_mask).cpu().numpy().transpose()
        # mask = mask.numpy().transpose()  # invert boolean mapping for visualization
        print(f"Attention mask shape: {mask.shape}")
        print(type(mask))

        # Plot the attention mask
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
        ax.set_xlabel("Query (Target)")
        ax.set_ylabel("Key (Attends to)")
        ax.set_xticks(np.arange(-.5, mask.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, mask.shape[0], 1), minor=True)
        ax.grid(which="minor", color="tab:grey", linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        fig.savefig("test_attn_mask.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    print("Running MultiHeadGASPAttention Test ...")

    # Set up configuration mimicking Layer 2 from original TF test script
    B = 1
    H = 2
    n_chans = 10
    key_dim = 16
    model_dim = H * key_dim

    # Layer parameters
    n_patches_out = 2
    patch_len_out = 3
    n_patches_in = 4
    patch_len_in = 4
    unpatched_len_in = 3
    l_unpatched_b=None
    l_patched_b=None

    # # Layer parameters (for time attention)
    # n_patches_out = 10
    # patch_len_out = 8
    # n_patches_in = 25
    # patch_len_in = 4
    # unpatched_len_in = 3
    # l_unpatched_b=None
    # l_patched_b=5

    # # Layer parameters (for channel attention)
    # n_patches_out = 52
    # patch_len_out = 1
    # n_patches_in = 52
    # patch_len_in = 1
    # unpatched_len_in = 0
    # l_unpatched_b=None
    # l_patched_b=None

    # Initialize layer
    mha_layer = MultiHeadGASPAttention(
        n_heads=H,
        model_dim=model_dim,
        l_other=n_chans,
        n_patches_out=n_patches_out,
        patch_len_out=patch_len_out,
        n_patches_in=n_patches_in,
        patch_len_in=patch_len_in,
        unpatched_len_in=unpatched_len_in,
        causal=True,
        l_unpatched_b=l_unpatched_b,
        l_patched_b=l_patched_b,
    )

    # Calculate required sequence lengths for the inputs
    l_out_total = n_patches_out * patch_len_out
    l_in_total = n_patches_in * patch_len_in

    # Create dummy tensors (B, sequence_length, n_channels, model_dim)
    query = torch.randn(B, l_out_total, n_chans, model_dim)
    key = torch.randn(B, l_in_total, n_chans, model_dim)
    value = torch.randn(B, l_in_total, n_chans, model_dim)

    # Forward pass
    output = mha_layer(query, key, value)

    print(f"Input Query shape:  {query.shape}")
    print(f"Input Key shape:    {key.shape}")
    print(f"Final Output shape: {output.shape}")

    assert output.shape == query.shape, "Output shape should match Query shape!"
    print("Shape assertion passed! Generating plot...") 

    # Plot the causal mask matrix
    _plot_attention_mask(mha_layer.gasp_layer)
