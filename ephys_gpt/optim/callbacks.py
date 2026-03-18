"""Callback functions for the EphysGPT training."""

# Import packages
import logging
import math
import os
import pytorch_lightning as pl
from pathlib import Path


_logger = logging.getLogger(__name__)


class LyapunovBetaSchedulerCallback(pl.Callback):
    """
    Lightning Callback to adapt the Lyapunov beta parameter to target a 
    specific Lyapunov to Cross-Entropy loss ratio.

    Note:
        - ratio = lyapunov_loss / cross_entropy_loss
        - If ratio > target, beta will decrease to reduce lyapunov_loss weight.
        - If ratio < target, beta will increase to boost lyapunov_loss weight.
        - Updates are multiplicative in log-space and clipped to [min_beta, max_beta].

    Parameters
    ----------
    lyapunov_module_attr: str
        The attribute name of the Lyapunov module.
    ce_loss_metric: str
        The metric name for the cross-entropy loss.
    lyap_loss_metric: str
        The metric name for the Lyapunov loss.
    target_ratio: float
        The target ratio for the Lyapunov loss to cross-entropy loss.
    adaptation_rate: float
        The rate at which the beta parameter adapts to changes in the loss ratio.
    min_beta: float
        The minimum value for the beta parameter.
    max_beta: float
        The maximum value for the beta parameter.
    warmup_epochs: int
        The number of warmup epochs for the beta parameter. Beta will be applied
        after the warmup period.
    ema_decay: float
        The exponential moving average decay rate.
    eps: float
        A small value to prevent division by zero.
    """
    def __init__(
        self,
        ce_loss_metric: str = "train/cross_entropy_loss",
        lyap_loss_metric: str = "train/lyapunov_loss",
        target_ratio: float = 0.1,
        adaptation_rate: float = 0.1,
        min_beta: float = 1e-6,
        max_beta: float = 10.0,
        warmup_epochs: int = 0,
        ema_decay: float = 0.9,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.ce_loss_metric = ce_loss_metric
        self.lyap_loss_metric = lyap_loss_metric

        self.target_ratio = target_ratio
        self.adaptation_rate = adaptation_rate
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.warmup_epochs = warmup_epochs
        self.ema_decay = ema_decay
        self.eps = eps

        self._ema_ce = None
        self._ema_lyap = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        lyap_module = pl_module.model.lyapunov_loss
        _logger.info(
            "Lyapunov Beta Scheduler enabled: beta=%.6g target_ratio=%.4g",
            lyap_module.beta.item(),
            self.target_ratio,
        )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Safely fetch metrics
        metrics = trainer.callback_metrics
        ce_tensor = metrics.get(self.ce_loss_metric)
        lyap_tensor = metrics.get(self.lyap_loss_metric)

        if ce_tensor is None or lyap_tensor is None:
            _logger.debug("LyapunovBetaScheduler: Required metrics missing this epoch. Skipping step.")
            return

        ce = ce_tensor.item()
        lyap = lyap_tensor.item()

        if self._ema_ce is None:
            # Initialize EMA with first observed epoch values
            self._ema_ce = ce
            self._ema_lyap = lyap
        else:
            # Smooth out noisy epoch-to-epoch losses before computing the control ratio
            self._ema_ce = self.ema_decay * self._ema_ce + (1 - self.ema_decay) * ce
            self._ema_lyap = self.ema_decay * self._ema_lyap + (1 - self.ema_decay) * lyap

        # Compute controlled ratio (how strong Lyapunov regularization is relative to CE loss)
        ratio = self._ema_lyap / max(self.eps, self._ema_ce)

        lyap_module = pl_module.model.lyapunov_loss
        beta = lyap_module.beta.item()

        if trainer.current_epoch + 1 > self.warmup_epochs:
            # Log-space controller (enables stable multiplicative updates around target ratio)
            log_error = math.log((ratio + self.eps) / (self.target_ratio + self.eps))
            # If ratio > target -> log_error > 0 -> beta decreases
            # If ratio < target -> log_error < 0 -> beta increases
            beta = beta * math.exp(-self.adaptation_rate * log_error)
            # Clip to enforce safety bounds and prevent runaway growth/collapse
            beta = max(self.min_beta, min(self.max_beta, beta))
            lyap_module.beta.fill_(beta)  # in-place update to preserve device placement

        # Log the scheduler metrics back to the PL Module
        _logger.info(f"Epoch {trainer.current_epoch} - beta: {beta:<.6g} | ratio: {ratio:<.6g}")
        pl_module.log("lyapunov_scheduler/beta", beta, on_epoch=True, sync_dist=True)
        pl_module.log("lyapunov_scheduler/loss_ratio", ratio, on_epoch=True, sync_dist=True)

    def state_dict(self) -> dict:
        """Saves the callback state to the checkpoint."""
        return {
            "ema_ce": self._ema_ce,
            "ema_lyap": self._ema_lyap,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restores the callback state from the checkpoint."""
        self._ema_ce = state_dict.get("ema_ce", None)
        self._ema_lyap = state_dict.get("ema_lyap", None)


class LyapunovMuSchedulerCallback(pl.Callback):
    """
    Lightning Callback to adapt the Lyapunov mu parameter to target a 
    specific V0 to stability loss ratio.

    Note:
        - ratio = (mu * v0_loss) / stability_loss
        - If ratio > target, mu will decrease to reduce v0_loss weight.
        - If ratio < target, mu will increase to boost v0_loss weight.
        - Updates are multiplicative in log-space and clipped to [min_mu, max_mu].
    """
    def __init__(
        self,
        stability_loss_metric: str = "train/lyapunov_stability_loss",
        v0_loss_metric: str = "train/lyapunov_v0_loss",
        target_ratio: float = 0.1,
        adaptation_rate: float = 0.1,
        min_mu: float = 1e-6,
        max_mu: float = 100.0,
        warmup_epochs: int = 0,
        ema_decay: float = 0.9,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.stability_loss_metric = stability_loss_metric
        self.v0_loss_metric = v0_loss_metric

        self.target_ratio = target_ratio
        self.adaptation_rate = adaptation_rate
        self.min_mu = min_mu
        self.max_mu = max_mu
        self.warmup_epochs = warmup_epochs
        self.ema_decay = ema_decay
        self.eps = eps

        self._ema_stability = None
        self._ema_v0 = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        lyap_module = pl_module.model.lyapunov_loss
        _logger.info(
            "Lyapunov Mu Scheduler enabled: mu=%.6g target_ratio=%.4g",
            lyap_module.mu.item(),
            self.target_ratio,
        )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Safely fetch metrics
        metrics = trainer.callback_metrics
        stability_tensor = metrics.get(self.stability_loss_metric)
        v0_tensor = metrics.get(self.v0_loss_metric)

        if stability_tensor is None or v0_tensor is None:
            _logger.debug("LyapunovMuScheduler: Required metrics missing this epoch. Skipping step.")
            return

        stability = stability_tensor.item()
        v0 = v0_tensor.item()

        if self._ema_stability is None:
            # Initialize EMA with first observed epoch values
            self._ema_stability = stability
            self._ema_v0 = v0
        else:
            # Smooth out noisy epoch-to-epoch losses before computing the control ratio
            self._ema_stability = self.ema_decay * self._ema_stability + (1 - self.ema_decay) * stability
            self._ema_v0 = self.ema_decay * self._ema_v0 + (1 - self.ema_decay) * v0

        lyap_module = pl_module.model.lyapunov_loss
        mu = lyap_module.mu.item()

        # Compute controlled ratio (relative weight of the V(0) anchor term)
        ratio = (mu * self._ema_v0) / max(self.eps, self._ema_stability)

        if trainer.current_epoch + 1 > self.warmup_epochs:
            # Log-space controller (enables stable multiplicative updates around target ratio)
            log_error = math.log((ratio + self.eps) / (self.target_ratio + self.eps))
            # If ratio > target -> log_error > 0 -> mu decreases
            # If ratio < target -> log_error < 0 -> mu increases
            mu = mu * math.exp(-self.adaptation_rate * log_error)
            # Clip to enforce safety bounds and prevent runaway growth/collapse
            mu = max(self.min_mu, min(self.max_mu, mu))
            lyap_module.mu.fill_(mu)  # in-place update to preserve device placement

        # Log the scheduler metrics back to the PL Module
        _logger.info(f"Epoch {trainer.current_epoch} - mu: {mu:<.6g} | ratio: {ratio:<.6g}")
        pl_module.log("lyapunov_scheduler/mu", mu, on_epoch=True, sync_dist=True)
        pl_module.log("lyapunov_scheduler/v0_ratio", ratio, on_epoch=True, sync_dist=True)

    def state_dict(self) -> dict:
        """Saves the callback state to the checkpoint."""
        return {
            "ema_stability": self._ema_stability,
            "ema_v0": self._ema_v0,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restores the callback state from the checkpoint."""
        self._ema_stability = state_dict.get("ema_stability", None)
        self._ema_v0 = state_dict.get("ema_v0", None)


class CheckpointCallback(pl.Callback):
    """
    Callback to save model checkpoints during training.

    Parameters
    ----------
    save_freq : int
        Frequency (in epochs) to save checkpoints.
    checkpoint_dir : str
        Directory to save checkpoints.
    """
    def __init__(self, save_freq: int, checkpoint_dir: str):
        super().__init__()
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.history_path = str(Path(self.checkpoint_dir).parent / "history.pkl")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        if epoch % self.save_freq == 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"ckpt-epoch{epoch}.ckpt")
            _logger.info(f"\nSaving checkpoint to {checkpoint_path}.")
            trainer.save_checkpoint(checkpoint_path, weights_only=False)
