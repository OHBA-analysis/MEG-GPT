"""
Implementation of the EphysGPT model.

Mathematical Notation:
    - B : batch size
    - L : sequence length
    - C : channel dimension
    - E : embedding dimension
    - D : model dimension
    - N_t : number of tokens (vocabulary size)
"""

# Import packages
import logging
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
from glob import glob
from omegaconf import OmegaConf
from typing import Dict, List, Optional
from ephys_gpt.configs import Config, get_config
from ephys_gpt.models import InputEmbeddingLayer, TransformerDecoder
from ephys_gpt.models.embeddings import LearnedPositionEmbedding
from ephys_gpt.models.utils import ShiftTokenLayer
from ephys_gpt.optim.losses import CrossEntropyLoss
from ephys_gpt.optim.optimizer import resolve_optimizer, resolve_lr_scheduler
from ephys_gpt.optim.initializer import init_model_weights


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


_NO_DECAY_TYPES = (
    nn.Embedding,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.BatchNorm1d,
    LearnedPositionEmbedding,
)


def _get_param_groups(model: nn.Module, weight_decay: float) -> List[Dict]:
    """
    Splits model parameters into two groups: those that receive weight decay
    and those that do not (embeddings, normalization layers, biases).
    """
    decay, no_decay = set(), set()
    for module_name, module in model.named_modules():
        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if isinstance(module, _NO_DECAY_TYPES) or "bias" in param_name:
                no_decay.add(full_name)
            else:
                decay.add(full_name)
    param_dict = {n: p for n, p in model.named_parameters()}
    return [
        {"params": [param_dict[n] for n in sorted(decay)],    "weight_decay": weight_decay},
        {"params": [param_dict[n] for n in sorted(no_decay)], "weight_decay": 0.0},
    ]


class EphysGPT(nn.Module):
    """
    EphysGPT class.

    Parameters
    ----------
    config : Config
        Configuration object.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config.config_class

        _logger.info("Initializing EphysGPT model.")

        # Get configs for each model components
        emb_cfg = self.config.input_embedding
        decoder_cfg = self.config.transformer_decoder
        loss_cfg = self.config.loss

        # Initialize input embedding layer
        self.shift_token_layer = ShiftTokenLayer()
        self.input_embedding_layer = InputEmbeddingLayer(
            emb_cfg.embedding_dim,
            emb_cfg.n_tokens,
            self.config.sequence_length,
            self.config.n_channels,
            emb_cfg.token_embedding_dim,
            emb_cfg.pos_embedding_dim,
            emb_cfg.pos_embedding_type,
            emb_cfg.channel_embedding_dim,
            emb_cfg.extra_label_specs,
        )

        # Initialize transformer decoder layer
        self.transformer_decoder = TransformerDecoder(
            decoder_cfg.n_heads,
            decoder_cfg.model_dim,
            emb_cfg.embedding_dim,
            self.config.n_channels,
            decoder_cfg.n_patches_out,
            decoder_cfg.patch_len_out,
            decoder_cfg.n_patches_in,
            decoder_cfg.patch_len_in,
            decoder_cfg.unpatched_len_in,
            decoder_cfg.l_unpatched_b,
            decoder_cfg.l_patched_b,
            decoder_cfg.do_chan_attention,
            decoder_cfg.do_cross_attention,
            decoder_cfg.chan_attention_mask,
            decoder_cfg.chan_attn_chandim,
            decoder_cfg.full_channel_attention_dropout,
            decoder_cfg.channel_attention_channel_dropout,
            decoder_cfg.time_attention_channel_dropout,
            decoder_cfg.feed_forward_dim,
            decoder_cfg.feed_forward_activation,
            decoder_cfg.dropout,
            decoder_cfg.norm_type,
            decoder_cfg.n_groups,
        )

        # Initialize prediction head layer
        self.prediction_head = nn.Linear(decoder_cfg.model_dim, emb_cfg.n_tokens)

        # Initialize loss layers
        self.cross_entropy_loss = CrossEntropyLoss(
            loss_cfg.loss_sequence_length,
            loss_cfg.top_k,
            loss_cfg.label_smoothing,
        )

        # Initialize model weights
        init_model_weights(self)

    def forward(
        self,
        x: torch.Tensor,
        extra_labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, l_in + 1, C).
        extra_labels : List[torch.Tensor]
            List of extra label tensors, each of shape (B, l_in + 1).
        """
        # Ensure that input tensors are integers for embedding lookups
        x = x.to(torch.long)
        extra_labels = [label.to(torch.long) for label in extra_labels]

        # Shift the tokens
        input, target = self.shift_token_layer(x)
        # *.shape: (B, l_in, C)

        # Get the input embeddings
        input_embeddings = self.input_embedding_layer(input, extra_labels)
        # shape: (B, l_in, C, E)

        # Forward pass through the decoder
        decoder_output = self.transformer_decoder(input_embeddings)
        # shape: (B, l_out, C, D)

        # Get next token logits
        y_pred_logits = self.prediction_head(decoder_output)
        # shape: (B, l_out, C, N_t)

        # Compute losses
        ce_loss, _, ce_metrics = self.cross_entropy_loss(y_pred_logits, target)

        return {
            "logits": y_pred_logits,
            "total_loss": ce_loss,
            "cross_entropy_loss": ce_loss,
            "cross_entropy_metrics": ce_metrics,
        }

    def get_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Gets embeddings weights from the model (as detached CPU tensors).

        Returns
        -------
        embeddings : Dict[str, torch.Tensor]
            Dictionary of model embedding weights.
        """
        # Get input embedding layer configs
        emb_cfg = self.config.input_embedding

        # Collect embeddings
        layer = self.input_embedding_layer
        embeddings = {
            "token": layer.token_embed.base_module.weight.detach().cpu(),
        }
        if emb_cfg.pos_embedding_type == "absolute":
            embeddings["position"] = layer.pos_embed.base_module.position_embeddings.detach().cpu()
        embeddings["channel"] = layer.channel_embed.base_module.position_embeddings.detach().cpu()
        for i, label in enumerate(emb_cfg.extra_label_specs):
            embeddings[label.name] = layer.extra_embeds[i].base_module.weight.detach().cpu()

        return embeddings

    def plot_attention_masks(self, save_path: Optional[str] = None) -> None:
        """
        Plots the attention mask for every GASPAttention instance in the decoder.
        See TransformerDecoder.plot_attention_masks() for details.

        Parameters
        ----------
        save_path : Optional[str]
            File path to save the figure. If None, the figure is shown interactively.
        """
        self.transformer_decoder.plot_attention_masks(save_path=save_path)


class EphysGPTModule(pl.LightningModule):
    """
    EphysGPT Lightning Module.

    Parameters
    ----------
    config : Config
        Configuration object.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.base_config = config
        self.config = config.config_class
        self.model = EphysGPT(config)

    def forward(
        self,
        x: torch.Tensor,
        extra_labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, l_in + 1, C, E).
        extra_labels : List[torch.Tensor]
            List of extra label tensors, each of shape (B, l_in + 1).
        """
        return self.model(x, extra_labels)

    def training_step(self, batch, batch_idx):
        """
        Training step.
        """
        x = batch["data"]
        extra_label_specs = self.config.input_embedding.extra_label_specs
        extra_labels = [
            batch[label.name] for label in extra_label_specs
        ] if extra_label_specs else []

        outputs = self.forward(x, extra_labels)

        log_kwargs = {
            "on_step": False, "on_epoch": True, "prog_bar": True,
            "batch_size": self.config.training.batch_size,
            "sync_dist": self.config.training.multi_gpu,
        }

        self.log("train/loss", outputs["total_loss"], **log_kwargs)
        self.log("train/cross_entropy_loss", outputs["cross_entropy_loss"], **log_kwargs)
        # NOTE: on_epoch logs the mean across all steps (batches) in the epoch.

        # Automatically log all the sub-metrics from each loss layer
        for metric_name, metric_val in outputs["cross_entropy_metrics"].items():
            self.log(f"train/{metric_name}", metric_val, **log_kwargs)
        
        return outputs["total_loss"]

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        """
        x = batch["data"]
        extra_label_specs = self.config.input_embedding.extra_label_specs
        extra_labels = [
            batch[label.name] for label in extra_label_specs
        ] if extra_label_specs else []

        outputs = self.forward(x, extra_labels)

        log_kwargs = {
            "on_step": False, "on_epoch": True, "prog_bar": True,
            "batch_size": self.config.training.batch_size,
            "sync_dist": self.config.training.multi_gpu,
        }

        self.log("val/loss", outputs["total_loss"], **log_kwargs)
        self.log("val/cross_entropy_loss", outputs["cross_entropy_loss"], **log_kwargs)
        # NOTE: on_epoch logs the mean across all steps (batches) in the epoch.

        # Automatically log all the sub-metrics from each loss layer
        for metric_name, metric_val in outputs["cross_entropy_metrics"].items():
            self.log(f"val/{metric_name}", metric_val, **log_kwargs)

        return outputs["total_loss"]

    def on_train_epoch_start(self):
        """
        Logs and prints the current learning rate at the start of each epoch.
        """
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        _logger.info(f"Epoch {self.current_epoch} - learning_rate: {lr:.6g}")
        self.log(
            "train/learning_rate", lr,
            on_epoch=True, prog_bar=False,
            sync_dist=self.config.training.multi_gpu,
        )

    def configure_optimizers(self):
        """
        Configures optimizers for training.
        """
        # Validation
        if (
            self.config is None 
            or not hasattr(self.config.training, "optimizer") 
            or not self.config.training.optimizer
        ):
            raise ValueError("Optimizer is not defined in the training configuration.")

        # Get optimizer
        optim_description = self.config.training.optimizer
        weight_decay = optim_description.get("weight_decay", 0.01)
        optimizer_name = optim_description.get("name", "adam").lower()
        if optimizer_name == "adamw" and weight_decay > 0:
            params = _get_param_groups(self, weight_decay)
        else:
            params = self.parameters()
        optimizer = resolve_optimizer(params, optim_description)

        # Get learning rate scheduler
        sched_description = getattr(self.config.training, "lr_scheduler", None)
        if sched_description:
            scheduler = resolve_lr_scheduler(optimizer, sched_description)
            interval = sched_description.get("interval", "epoch")
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": interval, "frequency": 1},
            }

        return optimizer

    def fit(
        self,
        trainer: pl.Trainer,
        datamodule: pl.LightningDataModule,
        ckpt_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Fits the model using the specified trainer and datamodule.
        """
        # Run training
        trainer.fit(self, datamodule=datamodule, ckpt_path=ckpt_path, **kwargs)

    # ----------------
    # Saving & Loading
    # ----------------

    def save(self, dirname: str) -> None:
        """
        Saves the model state to the specified directory.

        Parameters
        ----------
        dirname : str
            Directory to save the model files.
        """
        # Save model state
        os.makedirs(dirname, exist_ok=True)
        model_path = os.path.join(dirname, "model_state.pt")
        torch.save(self.model.state_dict(), model_path)
        _logger.info(f"Saved model state to {model_path}.")

    @classmethod
    def load_model(
        cls,
        dirname: str,
        config: Optional[Config] = None,
        checkpoint: Optional[str] = None,
        map_location: str = "cpu",
        strict: bool = True,
    ):
        """
        Loads the model from the specified directory.
        (Note that this function is mainly for the inference.)

        Parameters
        ----------
        dirname : str
            Directory to load the model files from.
        config : Config, optional
            Configuration object. If None, a config will be loaded from
            the specified directory.
        checkpoint : str, optional
            Checkpoint file path, file name, or "latest" to load the
            latest checkpoint. If None, the model will be loaded using a
            `model_state.pt` file.
        map_location : str, optional
            Map location for loading the model. Defaults to "cpu".
        strict : bool, optional
            Whether to enforce strict loading of model weights. Defaults to True.
        """
        # Load configuration if not provided
        if config is None:
            cfg = OmegaConf.load(f"{dirname}/config.yaml")
            config = get_config(cfg.model_config)

        # Instantiate module
        model_module = cls(config)

        # Helper function to find the latest checkpoint
        def _find_latest_ckpt(checkpoint_dir: str):
            files = sorted(
                glob(os.path.join(checkpoint_dir, "*.ckpt")), key=os.path.getmtime
            )
            return files[-1] if files else None

        if checkpoint:
            if checkpoint == "latest":
                ckpt_dir = os.path.join(dirname, "checkpoints")
                ckpt_path = _find_latest_ckpt(ckpt_dir)
                if ckpt_path is None:
                    raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}.")
            elif os.path.isabs(checkpoint) or os.path.exists(checkpoint):
                ckpt_path = checkpoint
            else:
                ckpt_candidate = os.path.join(dirname, checkpoint)
                if os.path.exists(ckpt_candidate):
                    ckpt_path = ckpt_candidate
                else:
                    raise FileNotFoundError(
                        f"Checkpoint {checkpoint} not found (tried as absolute path and under {dirname})."
                    )
            _logger.info(f"Loading model from checkpoint: {ckpt_path}")

            # Load Lightning checkpoint (safe on CPU)
            ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
            # NOTE: Includes model weights, optimizer / scheduler / AMP states, and metadata.

            # Load model weights (inference-friendly)
            state_dict = ckpt["state_dict"]
            model_module.load_state_dict(state_dict, strict=strict)

        else:
            # Weights-only path (inference-friendly)
            state_path = os.path.join(dirname, "model_state.pt")
            if not os.path.exists(state_path):
                raise FileNotFoundError(f"Model state file not found at {state_path}.")
            _logger.info(f"Loading model from file: {state_path}")

            model_state = torch.load(state_path, map_location=map_location, weights_only=True)
            model_module.model.load_state_dict(model_state, strict=strict)

        return model_module
