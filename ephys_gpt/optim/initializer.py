"""Utility functions for model initialization."""

import torch
import torch.nn as nn
import torch.nn.init as init


def init_model_weights(module: nn.Module) -> None:
    """
    Walks through `module` and applies inits:
      - nn.Linear -> Xavier uniform (bias as zeros)
      - nn.Conv1d -> Xavier uniform (bias as zeros)
      - nn.LayerNorm -> weight=1, bias=0
      - nn.Embedding -> Uniform [-0.05, 0.05]

    Example usage:
      model = EphysTokenizer(config)
      init_model_weights(model)

    Parameters
    ----------
    module : nn.Module
        The model or layer to initialize.
    """
    # If user passes the whole model, apply recursively
    for m in module.modules():
        # Linear
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        # Conv1d
        elif isinstance(m, nn.Conv1d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        # LayerNorm
        elif isinstance(m, nn.LayerNorm):
            if getattr(m, "weight", None) is not None:
                init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                init.zeros_(m.bias)
        # Embedding
        elif isinstance(m, nn.Embedding):
            init.uniform_(m.weight, a=-0.05, b=0.05)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()
