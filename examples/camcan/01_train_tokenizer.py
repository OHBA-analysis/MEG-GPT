"""Train EphysTokenizer on the CamCAN data."""

# Import packages
import hydra
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from pnpl.datasets import CamcanGlasser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from ephys_tokenizer.configs import get_config
from ephys_tokenizer.models import callbacks
from ephys_tokenizer.models.ephys_tokenizer import EphysTokenizerModule
from ephys_tokenizer.utils import plotting
from ephys_tokenizer.utils.train import get_history

from ephys_gpt.data.dataloader import EphysGPTDataModule


_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="models/tokenizer", config_name="config")
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

    use_gpu = (gpus is not None and gpus > 0)

    # Set data config
    data_dir = Path(cfg.data_config.data_dir)
    Fs = cfg.data_config.sampling_frequency

    # Load tokenizer model config
    model_config = get_config(cfg.model_config)  # Config object
    model_cfg = model_config.config_class  # tokenizer-specific Config object

    # Set model training config
    sequence_length = model_cfg.sequence_length
    batch_size = model_cfg.training.batch_size
    n_epochs = model_cfg.training.n_epochs
    multi_gpu = model_cfg.training.multi_gpu

    if gpus > 1 and not multi_gpu:
        _logger.warning("Multi-GPU training is not enabled in the model config. Enabling now ...")
        multi_gpu = True

    # Get directories
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "checkpoints").mkdir(exist_ok=True)
    (Path(run_dir) / "figures").mkdir(exist_ok=True)

    # Set seed (for reproducibility)
    pl.seed_everything(seed, workers=True)

    # ---------- Dataset ---------- #

    # Select subset of the data
    train_idx = np.array([
        38, 57, 421, 534, 413, 146, 245, 152, 410, 139, 79, 583, 489,
        67, 218, 260, 342, 118, 372, 51, 592, 289, 598, 504, 538, 171,
       320, 137, 41, 157, 341, 596, 375, 502, 32, 590, 560, 37, 155,
       495, 142, 183, 332, 339, 353, 518, 194, 475, 93, 64,
    ])  # selected using the numpy random generator with seed=813
    _logger.info(f"Number of training subjects: {len(train_idx)}")

    # Get data files
    data_files = sorted(data_dir.glob("*/sflip_parc-raw.fif"))
    data_files = [data_files[i] for i in train_idx]
    subject_ids = sorted([Path(f).parent.name for f in data_files])
    # NOTE: It is important to sort the subject IDs, as they get sorted automatically
    #       inside the CamcanGlasser dataset.
    #       If not, this creates mismatch between subjects in `plot_fitted_signal()`.

    # Prepare dataset and data module
    camcan_data = CamcanGlasser(
        data_path=data_dir,
        window_len=sequence_length,
        info=["subject", "dataset"],
        picks="misc",
        reject_by_annotation="omit",
        sampling_frequency=Fs,
        standardize=True,
        include_subjects=subject_ids,
        verbose=False,
    )
    camcan_datamodule = EphysGPTDataModule(
        dataset=camcan_data,
        batch_size=batch_size,
        val_split=0,
        split_method="subject_window",
        is_distributed=multi_gpu,
        seed=seed,
        num_workers=6,
        pin_memory=use_gpu,
        persistent_workers=True,
        drop_last=True,
    )

    # ---------- Model Training ---------- #

    trainer = None

    if checkpoint is None:
        # Build network via Lightning module
        pl_module = EphysTokenizerModule(model_config)

        # Set logger
        logger = CSVLogger(save_dir=run_dir, name="csv_logs")

        # Set callbacks
        checkpoint_callback = callbacks.CheckpointCallback(
            save_freq=1, checkpoint_dir=f"{run_dir}/checkpoints"
        )
        temperature_callback = callbacks.TemperatureAnnealingCallback(
            n_stages=model_cfg.callback.temperature_annealing["n_stages"],
            n_epochs=model_cfg.callback.temperature_annealing["n_annealing_epochs"],
            multi_gpu=multi_gpu,
        )
        cbs = [
            checkpoint_callback,
            temperature_callback,
            ModelSummary(),
            TQDMProgressBar()
        ]

        # Set trainer
        trainer_kwargs = dict(
            max_epochs=int(n_epochs),
            logger=logger,
            callbacks=cbs,
            deterministic=deterministic,
            precision=int(precision),
        )
        if use_gpu:
            trainer_kwargs["accelerator"] = "gpu"
            trainer_kwargs["devices"] = gpus

        trainer = pl.Trainer(**trainer_kwargs)

        # Run training via the module wrapper (refactors vocab after training)
        pl_module.fit(trainer=trainer, datamodule=camcan_datamodule)

        # Save trained model
        if trainer.is_global_zero:
            # Save model weights and token vocab
            pl_module.save(run_dir)

            # Save training history
            log_dir = Path(logger.log_dir)
            get_history(log_dir, save_dir=run_dir)

            _logger.info(f"Training finished. Model saved to: {run_dir}")

    else:
        # Load model
        pl_module = EphysTokenizerModule.load_model(run_dir, checkpoint=checkpoint)

        # Set up data module for testing
        camcan_datamodule.setup(stage="test")

    # ---------- Data Visualization ---------- #

    if trainer is None or trainer.is_global_zero:

        # Get full dataloader
        full_dataloader = camcan_datamodule.full_dataloader()
        # NOTE: Using `full_dataloader()` indicates we are using the full data subset defined
        #       above, not the entire Cam-CAN dataset.

        # Compute PVE
        pve = pl_module.get_pve(dataloader=full_dataloader)
        _logger.info(f"Percentage of Variance Explained (PVE) - Average: {pve.mean()}")
        plotting.plot_pve(pve, plot_dir=f"{run_dir}/figures")

        # Plot token kernel response
        token_response, input = pl_module.get_token_kernel_response(
            dataloader=full_dataloader, input="impulse"
        )
        plotting.plot_token_response(token_response, input, plot_dir=f"{run_dir}/figures")

        # Plot token counts histogram
        plotting.plot_token_counts(
            vocab=f"{run_dir}/vocab.pkl", plot_dir=f"{run_dir}/figures"
        )

        # Set up a dataset for the first subject only (needed for plot_fitted_signal)
        ex_data = CamcanGlasser(
            data_path=data_dir,
            window_len=sequence_length,
            info=["subject", "dataset"],
            picks="misc",
            reject_by_annotation="omit",
            sampling_frequency=Fs,
            standardize=True,
            include_subjects=[subject_ids[0]],
            verbose=False,
        )
        ex_datamodule = EphysGPTDataModule(
            dataset=ex_data,
            batch_size=batch_size,
            val_split=0,
            split_method="subject_window",
            is_distributed=multi_gpu,
            seed=seed,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
        )
        ex_datamodule.setup(stage="test")

        # Tokenize and reconstruct data for single subject
        tokens, token_weights = pl_module.tokenize_data(
            ex_datamodule.full_dataloader(),
            batch_size=32,
            remap=True,
            return_weights=True,
            num_workers=6,
        )
        reconstructed_data = pl_module.reconstruct_data(tokens)

        # Plot signals reconstructed from tokenized data (for one subject)
        plotting.plot_fitted_signal(
            original_data_path=(data_dir / subject_ids[0] / "sflip_parc-raw.fif"),
            reconstructed_data=reconstructed_data,
            token_weights=token_weights,
            subject_idx=0,
            plot_dir=f"{run_dir}/figures",
        )

    _logger.info("Training complete.")


if __name__ == "__main__":
    main()
