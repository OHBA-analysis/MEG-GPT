"""Tokenize the CamCAN data using trained EphysTokenizer."""

# Import packages
import hydra
import logging
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from pnpl.datasets import CamcanGlasser
from tqdm.auto import tqdm

from ephys_tokenizer.configs import get_config
from ephys_tokenizer.models.ephys_tokenizer import EphysTokenizerModule
from meg_gpt.data.dataloader import MEGGPTDataModule


_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="models/tokenizer", config_name="config")
def main(cfg: DictConfig):

    # ---------- Setting Up ---------- #

    _logger.info("\n===== Configuration =====:\n" + OmegaConf.to_yaml(cfg))

    # Set main config
    run_dir = cfg.main.run_dir
    gpus = cfg.main.gpus
    seed = cfg.main.seed
    checkpoint = cfg.main.checkpoint

    use_gpu = (gpus is not None and gpus > 0)

    if checkpoint is None:
        raise FileNotFoundError("No checkpoint found for loading model.")

    # Set data config
    data_dir = Path(cfg.data_config.data_dir)
    Fs = cfg.data_config.sampling_frequency

    # Set up directories to save data
    save_dir = Path("./data")
    tkn_data_dir = save_dir / "tokenized_data"
    recon_data_dir = save_dir / "reconstructed_data"

    tkn_data_dir.mkdir(parents=True, exist_ok=True)
    recon_data_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer model config
    model_config = get_config(cfg.model_config)  # Config object
    model_cfg = model_config.config_class  # tokenizer-specific Config object

    # Set model training config
    sequence_length = model_cfg.sequence_length
    batch_size = model_cfg.training.batch_size
    multi_gpu = model_cfg.training.multi_gpu

    if gpus > 1 and not multi_gpu:
        _logger.warning("Multi-GPU training is not enabled in the model config. Enabling now ...")
        multi_gpu = True

    # Set seed (for reproducibility)
    pl.seed_everything(seed, workers=True)

    # ---------- Data Tokenization ---------- #

    # Load model
    pl_module = EphysTokenizerModule.load_model(run_dir, checkpoint=checkpoint)

    # Get data files
    data_files = sorted(data_dir.glob("*/sflip_parc-raw.fif"))
    subject_ids = sorted([Path(f).parent.name for f in data_files])

    # Process subjects in partitions to avoid OOM
    partition_size = 50
    partitions = [
        subject_ids[i:i + partition_size]
        for i in range(0, len(subject_ids), partition_size)
    ]

    for p, part_subjects in enumerate(tqdm(partitions, desc="Processing partitions")):
        _logger.info(f"Partition {p + 1}/{len(partitions)}: {len(part_subjects)} subjects")

        # Prepare dataset and data module
        camcan_data = CamcanGlasser(
            data_path=data_dir,
            window_len=sequence_length,
            info=["subject", "dataset"],
            picks="misc",
            reject_by_annotation="omit",
            sampling_frequency=Fs,
            standardize=True,
            include_subjects=part_subjects,
            verbose=False,
        )
        camcan_datamodule = MEGGPTDataModule(
            dataset=camcan_data,
            batch_size=batch_size,
            val_split=0,
            split_method="subject_window",
            is_distributed=multi_gpu,
            seed=seed,
            num_workers=6,
            pin_memory=use_gpu,
            persistent_workers=True,
            drop_last=False,
        )
        camcan_datamodule.setup(stage="test")  # set up data module for testing    

        # Tokenize data used for training the model
        tokens = pl_module.tokenize_data(
            camcan_datamodule.full_dataloader(),
            batch_size=32,
            remap=True,
            return_weights=False,
            num_workers=6,
        )

        # Reconstruct data from tokens
        reconstructed_data = pl_module.reconstruct_data(tokens)

        # Save tokenized and reconstructed data
        for n, sid in enumerate(tqdm(part_subjects, desc="Saving tokenized data")):
            np.save(tkn_data_dir / f"token_{sid}.npy", tokens[n])
            np.save(recon_data_dir / f"recon_{sid}.npy", reconstructed_data[n])

    _logger.info("Tokenization complete.")


if __name__ == "__main__":
    main()
