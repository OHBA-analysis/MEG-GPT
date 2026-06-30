"""Utility helper functions for analyzing spatial attention."""

# Import packages
import logging
import math
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from meg_gpt.models.meg_gpt import MEGGPTModule


_logger = logging.getLogger(__name__)


def get_channel_embedding_affinity(
    embeddings: torch.Tensor,
    metric_type: str,
    eps: float = 1e-8,
    remove_diagonal: bool = True,
) -> np.ndarray:
    """
    Computes the affinity (e.g., similarity, distance) of channel embeddings.

    Parameters
    ----------
    embeddings : torch.Tensor
        The input channel embeddings.
        Shape should be (n_channels, channel_embedding_dim).
    metric_type : str
        The type of affinity metric to compute.
    eps : float
        Epsilon value to avoid division by zero.

    Returns
    -------
    output : np.ndarray
        The affinity matrix of shape (n_channels, n_channels).
    """
    W = embeddings.float()

    # Compute affinity matrix
    if metric_type == "cosine_similarity":
        dots = W @ W.T  # shape: (n_channels, n_channels)
        norms = torch.linalg.norm(W, dim=1, keepdim=True)  # shape: (n_channels, 1)
        output = dots / (norms * norms.T + eps)

    elif metric_type == "dot_product":
        output = W @ W.T

    elif metric_type == "euclidean_distance":
        diff = W[:, None, :] - W[None, :, :]
        # shape: (n_channels, n_channels, channel_embedding_dim)
        output = torch.linalg.norm(diff, dim=-1)

    elif metric_type == "correlation":
        W_centered = W - W.mean(dim=1, keepdim=True)
        dots = W_centered @ W_centered.T
        norms = torch.linalg.norm(W_centered, dim=1, keepdim=True)
        output = dots / (norms * norms.T + eps)

    else:
        raise ValueError(f"Unsupported metric: {metric_type}")

    # Move output to CPU and convert to numpy
    output = output.cpu().numpy()  # shape: (n_channels, n_channels)

    # Remove diagonal elements if specified
    if remove_diagonal:
        np.fill_diagonal(output, 0.0)

    return output


def extract_channel_attention_matrix(
    pl_module: "MEGGPTModule",
    dataloader: DataLoader,
    sample_size: Optional[Union[float, int]] = None,
    device: str = "cpu",
    aggregate: bool = False,
) -> Dict[int, torch.Tensor]:
    """
    Extracts channel attention matrices from a trained MEG-GPT model.

    Uses `register_forward_hook` to capture the softmax attention matrices
    computed inside each channel-attention layer, iterating over all batches
    in `dataloader` and concatenating the results.

    The hook is attached to the `GASPAttention` instance (accessible via
    `layer.gasp_chan_attention.gasp_layer`) of every `DecoderLayer` that
    has `do_chan_attention=True`. Inside the hook the attention matrices are
    recomputed from the projected, head-split query and key tensors that are
    passed into that module, together with the attention mask that is already
    registered as a buffer on the same module.

    Each batch dict is expected to have the same structure as the batches
    produced by `MEGGPTDataModule`: a `"data"` key containing token labels
    sequences, plus one key per entry in `extra_label_specs` (e.g.
    "subject_labels").

    Parameters
    ----------
    pl_module : MEGGPTModule
        Trained MEG-GPT Lightning module.
    dataloader : DataLoader
        DataLoader whose batches are dicts with at least a "data" key
        containing token labels sequences of shape (B, L+1, C), plus any
        extra-label keys required by the model config.
        Typically the validation DataLoader from `MEGGPTDataModule`.
    sample_size : Optional[Union[float, int]]
        If provided, randomly subsample the dataloader's dataset before
        extracting attention matrices.

        - float in (0, 1]: fraction of samples to draw, e.g. 0.1 for 10%.
        - int > 0: absolute number of samples to draw.

        If None (default), all samples in the dataloader are used.
        Sampling is without replacement and uses the current global random
        state; call `torch.manual_seed` beforehand for reproducibility.
    device : str
        Device on which to run the forward pass (e.g. "cpu" or "cuda").
    aggregate : bool
        If True, accumulate an online mean over samples and sequence length
        instead of storing all tensors. Returns tensors of shape (H, C_q, C_k)
        rather than (N, H, L, C_q, C_k). Use this when the full validation set
        is too large to hold in memory.

    Returns
    -------
    attn_matrices : Dict[int, torch.Tensor]
        Mapping from decoder layer index to a channel attention matrix.
        Only layers with `do_chan_attention=True` are included.

        When `aggregate=False` (default), each tensor has shape
        (N, H, L, C_q, C_k) where:

        - N  : total number of samples across all batches
        - H  : number of attention heads
        - L  : sequence length (the time dimension, which is the "other"
               dimension in the channel attention formulation)
        - C_q: number of query channels (always n_channels)
        - C_k: number of key channels (equals `chan_attn_chandim` when that
               config option is set, otherwise equals C_q)

        When `aggregate=True`, each tensor has shape (H, C_q, C_k),
        representing the mean attention matrix averaged over all samples and
        the sequence length dimension.

        The matrix elements are non-negative and sum to 1 along the C_k axis
        (i.e. each query channel's attention over all key channels is a
        probability distribution). Positions masked by `chan_attention_mask`
        will be zero.

    Raises
    ------
    ValueError
        If none of the model's decoder layers have channel attention enabled.

    Examples
    --------
    >>> attn = extract_channel_attention_matrices(pl_module, val_loader, device="cpu")
    >>> layer_idx = next(iter(attn))
    >>> matrices = attn[layer_idx]  # (N, H, L, C, C)
    >>> mean_matrices = matrices.mean(dim=(0, 1, 2)).numpy()  # (C, C)
    """
    # Identify decoder layers that have channel attention enabled
    decoder_layers = pl_module.model.transformer_decoder.layers
    chan_attn_modules: Dict[int, torch.nn.Module] = {
        i: layer.gasp_chan_attention.gasp_layer
        for i, layer in enumerate(decoder_layers)
        if layer.do_chan_attention
    }

    if not chan_attn_modules:
        raise ValueError(
            "No decoder layers with channel attention found. "
            "Check that at least one entry in the 'do_chan_attention' config list is True."
        )

    # Optionally subsample the dataset
    n_total = len(dataloader.dataset)
    if sample_size is not None:
        if isinstance(sample_size, float):
            if not 0.0 < sample_size <= 1.0:
                raise ValueError(
                    f"sample_size as a float must be in (0, 1], got {sample_size}."
                )
            n_samples = max(1, int(sample_size * n_total))
        else:
            if sample_size < 1 or sample_size > n_total:
                raise ValueError(
                    f"sample_size as an int must be in [1, {n_total}], got {sample_size}."
                )
            n_samples = sample_size

        indices = torch.randperm(n_total)[:n_samples].tolist()
        subset = Subset(dataloader.dataset, indices)
        dataloader = DataLoader(
            subset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            collate_fn=dataloader.collate_fn,
            worker_init_fn=dataloader.worker_init_fn,
        )
        _logger.info("Subsampled %d / %d validation sequences.", n_samples, n_total)

    _logger.info(
        "Extracting channel attention matrices from %d layer(s): %s",
        len(chan_attn_modules),
        list(chan_attn_modules.keys()),
    )

    # Extra label keys to pull from each batch dict, mirroring training_step
    extra_label_specs = pl_module.config.input_embedding.extra_label_specs

    # Per-layer storage: running (sum, count) when aggregate=True, else list of tensors
    captured: Dict[int, Any] = (
        {i: (None, 0) for i in chan_attn_modules}
        if aggregate
        else {i: [] for i in chan_attn_modules}
    )

    def _make_hook(layer_idx: int):
        """
        Returns a forward hook for the GASPAttention module at `layer_idx`.

        GASPAttention.forward() receives head-split (q, k, v) tensors from
        MultiHeadGASPAttention and passes them directly to Attention.forward()
        with the registered attention_mask buffer.  Hooking at this level gives
        us q, k with shapes (B, H, l_out, l_other, key_dim) and access to
        module.attention_mask and module.key_dim — everything needed to
        reconstruct the attention matrices without modifying the model.
        """
        def hook(
            module: torch.nn.Module,
            input: tuple,
            output: torch.Tensor,
        ) -> None:
            q, k = input[0], input[1]
            # q.shape: (B, H, C_q, L, key_dim)
            # k.shape: (B, H, C_k, L, key_dim)

            # Mirror the rearrangements performed inside Attention.forward()
            q_r = rearrange(q, "b h l_out l_other k_d -> b h l_other l_out k_d")
            k_r = rearrange(k, "b h l_in l_other k_d -> b h l_other k_d l_in")

            # Scaled dot-product; shape: (B, H, l_other, C_q, C_k)
            attn = (q_r @ k_r) / math.sqrt(module.key_dim)

            # Apply the attention mask (True → masked out)
            mask = module.attention_mask  # (C_q, C_k) bool buffer
            if mask is not None and mask.any():
                attn = attn.masked_fill(mask, float("-inf"))

            attn = F.softmax(attn, dim=-1)

            # Replace NaNs that arise from fully-masked rows
            attn = torch.nan_to_num(attn, nan=0.0)

            if aggregate:
                # Average over sequence length and batch samples
                attn_mean = attn.detach().cpu().mean(dim=2)
                running_sum, count = captured[layer_idx]
                batch_sum = attn_mean.sum(dim=0)  # (H, C_q, C_k)
                captured[layer_idx] = (
                    batch_sum if running_sum is None else running_sum + batch_sum,
                    count + attn_mean.shape[0],
                )
            else:
                captured[layer_idx].append(attn.detach().cpu())

        return hook

    # Register hooks
    hook_handles = [
        module.register_forward_hook(_make_hook(i))
        for i, module in chan_attn_modules.items()
    ]

    # Iterate over all batches with hooks active
    was_training = pl_module.training
    pl_module.eval()
    pl_module.to(device)
    try:
        with torch.no_grad():
            for batch in tqdm(
                dataloader, desc="Extracting attention matrices", total=len(dataloader)
            ):
                x = batch["data"].to(device)
                extra_labels = (
                    [batch[spec.name].to(device) for spec in extra_label_specs]
                    if extra_label_specs else []
                )
                pl_module.model(x, extra_labels)
    finally:
        for handle in hook_handles:
            handle.remove()
        if was_training:
            pl_module.train()

    if aggregate:
        return {i: s / n for i, (s, n) in captured.items()}

    # Concatenate per-batch results along the sample dimension
    return {i: torch.cat(captured[i], dim=0) for i in captured}


def channel_attention_rollout(
    attention_matrices: Dict[int, torch.Tensor],
    add_residual: bool = True,
) -> torch.Tensor:
    """
    Computes attention rollout (Abnar & Zuidema, 2020) over channel attention.

    Averages per-layer matrices over heads, optionally mixes in the identity
    residual (0.5*A + 0.5*I) and re-normalises rows, then sequentially
    matrix-multiplies across layers to produce a global channel-to-channel
    influence matrix.

    Parameters
    ----------
    attention_matrices : Dict[int, torch.Tensor]
        Output of `extract_channel_attention_matrix(aggregate=True)`.
        Each value has shape (H, C_q, C_k).
    add_residual : bool
        If True, apply `A_hat = 0.5*A + 0.5*I` and re-normalise rows
        before multiplying. Requires square matrices (C_q == C_k).

    Returns
    -------
    rollout : torch.Tensor
        Rollout matrix of shape (C, C). Row i is a probability distribution over
        key channels representing cumulative influence on query channel i across
        all layers.

    Raises
    ------
    ValueError
        If attention_matrices is empty, or if any layer has C_q != C_k
        (caused by chan_attn_chandim reducing the key channel dimension).

    References
    ----------
    Abnar, S. & Zuidema, W. (2020). Quantifying Attention Flow in
    Transformers. ACL 2020. https://arxiv.org/abs/2005.00928
    """
    if not attention_matrices:
        raise ValueError(
            "attention_matrices is empty. "
            "Ensure extract_channel_attention_matrix returned at least one layer."
        )

    for (layer_idx, mat) in attention_matrices.items():
        C_q, C_k = mat.shape[1], mat.shape[2]
        if C_q != C_k:
            raise ValueError(
                f"Layer {layer_idx} has a non-square attention matrix "
                f"(C_q={C_q}, C_k={C_k}). Rollout requires square matrices. "
                "This is caused by chan_attn_chandim reducing the key channel "
                "dimension. Remove chan_attn_chandim or set it equal to n_channels."
            )

    rollout: Optional[torch.Tensor] = None

    for layer_idx in sorted(attention_matrices.keys()):
        A = attention_matrices[layer_idx].float().mean(dim=0)
        # average over heads; shape: (C, C)

        if add_residual:
            C = A.shape[0]
            I = torch.eye(C, dtype=A.dtype, device=A.device)
            A = 0.5 * A + 0.5 * I
            A = A / A.sum(dim=-1, keepdim=True)

        rollout = A if rollout is None else A @ rollout

    return rollout
