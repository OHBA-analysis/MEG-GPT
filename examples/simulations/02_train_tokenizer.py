"""Train EphysTokenizer on the simulated data."""

# Import packages
import hydra
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm.auto import tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from ephys_tokenizer.configs import get_config
from ephys_tokenizer.models import callbacks
from ephys_tokenizer.models.ephys_tokenizer import EphysTokenizerModule
from ephys_tokenizer.utils import plotting
from ephys_tokenizer.utils.train import get_history

from meg_gpt.data.datasets import SimulationDataset
from meg_gpt.data.dataloader import MEGGPTDataModule
from meg_gpt.utils.processing import standardize, temporal_filter


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
    data_dir = cfg.data_config.data_dir
    Fs = cfg.data_config.sampling_frequency

    if data_dir is None:
        data_dir = Path("./data_burst")
    else:
        data_dir = Path(data_dir)

    # Load tokenizer model config
    model_config = get_config(cfg.model_config)  # Config object
    model_cfg = model_config.config_class  # tokenizer-specific Config object

    # Set model training config
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

    # Get data files
    data_files = sorted(data_dir.glob("x_*.npy"))

    # Bandpass filter data
    if data_dir.stem == "data_burst":
        save_dir = data_dir / "filtered_data"

        if not save_dir.exists():
            save_dir.mkdir()
            for n, file in enumerate(data_files):
                data = np.load(file)
                data = temporal_filter(data, Fs, low_freq=5, high_freq=40)
                data = standardize(data)

                np.save(save_dir / f"x_{n:0{len(str(len(data_files)))}}.npy", data)
    else:
        save_dir = data_dir

    # NOTE: For bursting simulation data, we apply bandpass filtering. This is not necessary,
    #       but makes the training more stable.

    # Prepare dataset and data module
    sim_data = SimulationDataset(
        data_path=save_dir,
        window_len=model_cfg.sequence_length,
        sampling_frequency=Fs,
        info=["subject", "dataset"],
        standardize=False,
    )
    sim_datamodule = MEGGPTDataModule(
        dataset=sim_data,
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
        pl_module.fit(trainer=trainer, datamodule=sim_datamodule)

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
        sim_datamodule.setup(stage="test")

    # ---------- Data Saving ---------- #

    if trainer is None or trainer.is_global_zero:
        # Tokenize data used for training the model
        tokens, token_weights = pl_module.tokenize_data(
            sim_datamodule.full_dataloader(),
            batch_size=16,
            remap=True,
            return_weights=True,
            num_workers=6,
        )

        # Reconstruct data from tokens
        reconstructed_data = pl_module.reconstruct_data(tokens)

        # Save tokenized and reconstructed data
        tkn_data_dir = data_dir / "tokenized_data"
        tkn_data_dir.mkdir(exist_ok=True)

        recon_data_dir = data_dir / "reconstructed_data"
        recon_data_dir.mkdir(exist_ok=True)

        for n, file in enumerate(tqdm(data_files, desc="Saving tokenized data")):
            subject_id = file.stem.split("_")[1]
            np.save(tkn_data_dir / f"token_{subject_id}.npy", tokens[n])
            np.save(recon_data_dir / f"recon_{subject_id}.npy", reconstructed_data[n])

    # ---------- Visualization ---------- #

    if trainer is None or trainer.is_global_zero:
        # Compute PVE
        pve = pl_module.get_pve(dataloader=sim_datamodule.full_dataloader())
        print(f"Percentage of Variance Explained (PVE) - Average: {pve.mean()}")
        plotting.plot_pve(pve, plot_dir=f"{run_dir}/figures")

        # Plot token kernel response
        token_response, input = pl_module.get_token_kernel_response(
            dataloader=sim_datamodule.full_dataloader(),
            input="impulse",
        )
        plotting.plot_token_response(token_response, input, plot_dir=f"{run_dir}/figures")

        # Plot token counts histogram
        plotting.plot_token_counts(
            vocab=f"{run_dir}/vocab.pkl", plot_dir=f"{run_dir}/figures"
        )

        # Plot signals reconstructed from tokenized data (for one session)
        plotting.plot_fitted_signal(
            original_data_path=data_files[0],
            reconstructed_data=reconstructed_data,
            token_weights=token_weights,
            subject_idx=0,
            plot_dir=f"{run_dir}/figures",
        )

    _logger.info("Training complete.")


if __name__ == "__main__":
    main()
