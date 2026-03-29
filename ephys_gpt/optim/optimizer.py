"""Functions to handle optimizers for model training."""

# Import packages
import torch
import torch.optim as optim
from omegaconf import DictConfig
from typing import Any, Iterable


def resolve_optimizer(
        params: Iterable, optimizer_descriptor: Any
    ) -> torch.optim.Optimizer:
    """
    Resolves a PyTorch optimizer from an optimizer descriptor.

    Parameters
    ----------
    params : Iterable
        The parameters to optimize.
    optimizer_descriptor : Any
        The optimizer descriptor. This can be:
            - a callable: optimizer_descriptor(params) -> optimizer instance
            - a torch optimizer instance -> returned directly
            - a tuple/list: (torch.optim.OptimizerClass, {"lr":..., ...})
            - a dict/DictConfig -> attempt to map to torch optimizer
    """
    if callable(optimizer_descriptor):
        return optimizer_descriptor(params)

    if isinstance(optimizer_descriptor, torch.optim.Optimizer):
        return optimizer_descriptor

    if isinstance(optimizer_descriptor, (list, tuple)) and len(optimizer_descriptor) >= 1:
        optim_cls = optimizer_descriptor[0]
        optim_kwargs = optimizer_descriptor[1] if len(optimizer_descriptor) > 1 else {}
        return optim_cls(params, **optim_kwargs)

    if isinstance(optimizer_descriptor, (dict, DictConfig)):
        name = optimizer_descriptor.get("name", "adam").lower()
        lr = optimizer_descriptor.get("learning_rate", 1e-3)
        eps = optimizer_descriptor.get("eps", 1e-7)

        if name == "adam":
            return optim.Adam(params, lr=lr, eps=eps)

        if name == "adamw":
            betas = optimizer_descriptor.get("betas", (0.9, 0.999))
            if isinstance(betas, list):
                betas = tuple(betas)
            weight_decay = optimizer_descriptor.get("weight_decay", 0.01)
            return optim.AdamW(
                params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
            )

        if name == "sgd":
            return optim.SGD(params, lr=lr)

        return optim.Adam(params, lr=lr, eps=eps)

    raise ValueError("Unsupported optimizer descriptor.")
