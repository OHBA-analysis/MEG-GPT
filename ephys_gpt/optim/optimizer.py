"""Functions to handle optimizers and learning rate schedulers for model training."""

# Import packages
import math
import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch.optim.lr_scheduler import _LRScheduler
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


def resolve_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_descriptor: Any,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Resolves a PyTorch LR scheduler from a scheduler descriptor.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to schedule.
    scheduler_descriptor : Any
        A dict/DictConfig with at minimum a 'name' key. Supported names:
            - 'cosine_annealing': CosineAnnealingLR. Requires 'T_max'.
              Optional: 'eta_min' (default 0.0).
            - 'cosine_annealing_warm_restarts': CosineAnnealingWarmRestarts.
              Requires 'T_0'. Optional: 'T_mult' (default 1), 'eta_min' (default 0.0).
            - 'cosine_annealing_with_warmup': LambdaLR with linear warmup then cosine annealing.
              Requires 'T_max' and 'warmup_epochs'. Optional: 'warmup_start_factor' (default 1e-8),
              'eta_min' (default 0.0).
    """
    if scheduler_descriptor is None:
        return None

    name = scheduler_descriptor.get("name", "").lower()
    eta_min = scheduler_descriptor.get("eta_min", 0.0)

    if name == "cosine_annealing":
        T_max = scheduler_descriptor["T_max"]
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )

    if name == "cosine_annealing_warm_restarts":
        T_0 = scheduler_descriptor["T_0"]
        T_mult = scheduler_descriptor.get("T_mult", 1)
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )

    if name == "cosine_annealing_with_warmup":
        total_steps = scheduler_descriptor["total_steps"]
        warmup_steps = scheduler_descriptor["warmup_steps"]
        warmup_start_lr = scheduler_descriptor.get("warmup_start_lr", 1.0e-08)
        return CosineAnnealingWithWarmupLR(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            eta_min=eta_min,
        )

    raise ValueError(
        f"Unsupported lr_scheduler name: '{name}'. Choose 'cosine_annealing', "
        "'cosine_annealing_warm_restarts', or 'cosine_annealing_with_warmup'."
    )


class CosineAnnealingWithWarmupLR(_LRScheduler):
    """
    Custom learning rate scheduler combining a cosine annealing scheduler with
    a linear warmup.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer to schedule.
    total_steps : int
        Total number of training steps/epochs.
    warmup_steps : int
        Number of warmup steps/epochs.
    warmup_start_lr : float
        Initial learning rate for warmup.
    eta_min : float
        Minimum learning rate after cosine annealing.
    last_epoch : int
        The index of last epoch.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        # Validation
        if self.warmup_steps > self.total_steps:
            raise ValueError("warmup_steps must be <= total_steps")

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        # Linear warmup phase
        if step < self.warmup_steps and self.warmup_steps > 0:
            progress = step / (self.warmup_steps - 1)
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * progress
                for base_lr in self.base_lrs
            ]

        # Cosine annealing phase
        progress = (step - self.warmup_steps + 1) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
            for base_lr in self.base_lrs
        ]
