"""Utility helper functions for generative sampling."""

# Import packages
import torch
from torch.distributions import Categorical
from typing import Optional


def sample_from_logits(
    logits: torch.Tensor,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    typical_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Samples from the logits using either the top-p, top-k, or locally typical sampling.

    If none are specified, argmax sampling is used, where the token with the
    highest logit is selected.

    Parameters
    ----------
    logits : torch.Tensor
        Logits to sample from. Shape is (*batch_dims, n_tokens).
    top_p : float, optional
        The cumulative probability threshold for top-p (nucleus) sampling.
    top_k : int, optional
        The number of top logits to sample from.
    typical_p : float, optional
        The cumulative probability threshold for locally typical sampling
        (Meister et al., 2023).

    Returns
    -------
    sampled_tokens : torch.Tensor
        The sampled tokens. Shape is (*batch_dims,).
    """
    n_specified = sum(arg is not None for arg in [top_p, top_k, typical_p])
    if n_specified > 1:
        raise ValueError(
            "Only one of 'top_p', 'top_k', or 'typical_p' can be specified."
        )
    elif top_p is not None:
        sampled_tokens = top_p_sampling(logits, top_p)
    elif top_k is not None:
        sampled_tokens = top_k_sampling(logits, top_k)
    elif typical_p is not None:
        sampled_tokens = typical_sampling(logits, typical_p)
    else:
        sampled_tokens = torch.argmax(logits, dim=-1)  # argmax sampling

    return sampled_tokens.to(torch.long)


def top_k_sampling(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Top-k sampling; only samples from the largest k logits.

    Parameters
    ----------
    logits : torch.Tensor
        Logits to sample from. Shape is (*batch_dims, n_tokens).
    k : int
        The number of top logits to sample from.

    Returns
    -------
    sampled_tokens : torch.Tensor
        The sampled tokens. Shape is (*batch_dims,).
    """
    # Get k largest logits and their indices
    top_k_logits, top_k_indices = torch.topk(logits, k=k, dim=-1)

    # Sample from the top k logits
    sampled_indices = Categorical(logits=top_k_logits).sample()

    # Get the corresponding tokens
    sampled_tokens = torch.gather(
        top_k_indices, dim=-1, index=sampled_indices.unsqueeze(-1)
    ).squeeze(-1)

    return sampled_tokens


def top_p_sampling(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Top-p (nucleus) sampling; samples from the smallest set of tokens whose cumulative
    probability exceeds p.

    Parameters
    ----------
    logits : torch.Tensor
        Logits to sample from. Shape is (*batch_dims, n_tokens).
    p : float
        The cumulative probability threshold for top-p sampling.

    Returns
    -------
    sampled_tokens : torch.Tensor
        The sampled tokens. Shape is (*batch_dims,).
    """
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # Compute cumulative probabilities
    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Mask logits beyond the cutoff index
    sorted_indices_to_remove = cum_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    # NOTE: We shift right by 1 to include the first token that crosses the threshold.
    #       This ensures that the cumulative probability of the selected set of tokens
    #       is >= ("exceeds") p.

    # Apply the mask to the logits
    sorted_logits[sorted_indices_to_remove] = -float("inf")

    # Sample from the masked logits
    sampled_indices = Categorical(logits=sorted_logits).sample()

    # Get the corresponding tokens
    sampled_tokens = torch.gather(
        sorted_indices, dim=-1, index=sampled_indices.unsqueeze(-1)
    ).squeeze(-1)

    return sampled_tokens


def typical_sampling(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Locally typical sampling (Meister et al., 2022).

    This method samples from tokens whose surprisal is closest to the
    distribution's entropy (the Asymptotic Equipartition Property typical set),
    potentially suppressing attractor tokens without distorting the learned
    distribution.

    Parameters
    ----------
    logits : torch.Tensor
        Logits to sample from. Shape is (*batch_dims, n_tokens).
    p : float
        Cumulative probability mass of the typical set to retain.

    Returns
    -------
    sampled_tokens : torch.Tensor
        The sampled tokens. Shape is (*batch_dims,).
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    # Shannon entropy H = -sum(p * log_p)
    entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)

    # Typicality score: |surprisal - entropy| (0 = perfectly typical)
    typicality = torch.abs(-log_probs - entropy)

    # Sort by typicality ascending (most typical first)
    sorted_typicality, sorted_indices = torch.sort(typicality, dim=-1)
    sorted_probs = probs.gather(-1, sorted_indices)

    # Keep the smallest typical set whose cumulative mass >= p.
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    sorted_to_remove = cumulative_probs - sorted_probs > p
    # NOTE: cumulative_probs - sorted_probs is the exclusive prefix sum (mass strictly before i),
    #       so the token that first pushes the sum over p already evaluates to False (kept).
    #       No shift is needed here, unlike top_p_sampling which uses the inclusive cumsum.

    sorted_logits = logits.gather(-1, sorted_indices)
    sorted_logits[sorted_to_remove] = -float("inf")

    sampled_indices = Categorical(logits=sorted_logits).sample()

    return torch.gather(
        sorted_indices, dim=-1, index=sampled_indices.unsqueeze(-1)
    ).squeeze(-1)
