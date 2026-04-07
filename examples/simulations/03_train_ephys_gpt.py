"""Train EphysGPT on the simulated data."""

# Import packages
import hydra
import logging
import pickle
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from ephys_gpt.configs import get_config
from ephys_gpt.data.datasets import SimulationDataset
from ephys_gpt.data.dataloader import EphysGPTDataModule
from ephys_gpt.models.ephys_gpt import EphysGPTModule
from ephys_gpt.optim import callbacks
from ephys_gpt.utils.post_hoc import get_history


_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="models/generator", config_name="config")
def main(cfg: DictConfig):

    # ---------- Setting Up ---------- #

    _logger.info("\n===== Configuration =====:\n" + OmegaConf.to_yaml(cfg))

    # Set main config
    run_dir = cfg.main.run_dir
    gpus = cfg.main.gpus
    precision = cfg.main.precision
    deterministic = cfg.main.deterministic
    seed = cfg.main.seed
    checkpoint = cfg.main.checkpoint
    wandb_cfg = cfg.main.get("wandb", {})

    # Set data config
    data_dir = cfg.data_config.data_dir
    Fs = cfg.data_config.sampling_frequency

    if data_dir is None:
        data_dir = Path("./data_burst")
    else:
        data_dir = Path(data_dir)
    data_dir = data_dir / "tokenized_data"

    # Validate number of tokens
    n_tokens = cfg.model_config.input_embedding.n_tokens
    with open("models/tokenizer/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    if len(vocab["total_token_counts"]) != n_tokens or n_tokens is None:
        n_tokens = len(vocab["total_token_counts"]) + 1
    _logger.info(f"Using {n_tokens} tokens.")

    # Set model training config
    batch_size = cfg.model_config.training.batch_size
    val_split = cfg.model_config.training.val_split
    n_epochs = cfg.model_config.training.n_epochs
    multi_gpu = cfg.model_config.training.multi_gpu

    # Load EphysGPT model config
    model_config = get_config(cfg.model_config)  # Config object
    model_cfg = model_config.config_class  # EphysGPT-specific Config object

    # Get directories
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "checkpoints").mkdir(exist_ok=True)
    (Path(run_dir) / "figures").mkdir(exist_ok=True)

    # Set seed (for reproducibility)
    pl.seed_everything(seed, workers=True)

    # ---------- Dataset ---------- #

    # Prepare dataset and data module
    sim_data = SimulationDataset(
        data_path=data_dir,
        window_len=int(model_cfg.sequence_length + 1),
        sampling_frequency=Fs,
        info=["subject", "dataset"],
        standardize=False,
    )
    sim_datamodule = EphysGPTDataModule(
        dataset=sim_data,
        batch_size=batch_size,
        val_split=val_split,
        split_method="subject_window",
        is_distributed=multi_gpu,
        seed=seed,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    # ---------- Model Training ---------- #

    if checkpoint is None:
        _logger.info("Training EphysGPT model ...")

        # Build network via Lightning module
        pl_module = EphysGPTModule(model_config)

        # Set loggers
        csv_logger = CSVLogger(save_dir=run_dir, name="csv_logs")
        log_dir = Path(run_dir) / "csv_logs/version_0"
        loggers = [csv_logger]

        if wandb_cfg.get("enabled", False):
            wandb_logger = WandbLogger(
                project=wandb_cfg.get("project", "EphysGPT"),
                name=wandb_cfg.get("name", None),
                save_dir=run_dir,
            )
            watch_log = wandb_cfg.get("watch_log", None)
            if watch_log:
                wandb_logger.watch(
                    pl_module,
                    log=watch_log,
                    log_freq=int(wandb_cfg.get("watch_log_freq", 100)),
                )
            loggers.append(wandb_logger)

        # Set callbacks
        checkpoint_callback = callbacks.CheckpointCallback(
            save_freq=1, checkpoint_dir=f"{run_dir}/checkpoints"
        )
        cbs = [
            checkpoint_callback,
            ModelSummary(max_depth=2),
            TQDMProgressBar(),
        ]

        # Set trainer
        trainer_kwargs = dict(
            max_epochs=int(n_epochs),
            logger=loggers,
            callbacks=cbs,
            deterministic=deterministic,
            precision=precision,
        )
        if gpus and gpus > 0:
            trainer_kwargs["accelerator"] = "gpu"
            trainer_kwargs["devices"] = gpus

        trainer = pl.Trainer(**trainer_kwargs)

        # Run training via the module wrapper
        pl_module.fit(trainer=trainer, datamodule=sim_datamodule)
        pl_module.save(run_dir)  # save model weights and token vocab
        get_history(log_dir, save_dir=run_dir)  # save training history
        _logger.info(f"Training finished. Model saved to: {run_dir}")

    _logger.info("Training complete.")


if __name__ == "__main__":
    main()
