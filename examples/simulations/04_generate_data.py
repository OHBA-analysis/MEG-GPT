"""Generate simulated data using trained EphysGPT model."""

# Import packages
import gc
import hydra
import logging
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from ephys_gpt.inference.generator import EphysGPTGenerator
from ephys_gpt.models.ephys_gpt import EphysGPTModule
from ephys_tokenizer.models.ephys_tokenizer import EphysTokenizerModule


_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="models/generator", config_name="config")
def main(cfg: DictConfig):

    # ---------- Setting Up ---------- #

    _logger.info("\n===== Configuration =====:\n" + OmegaConf.to_yaml(cfg))

    # Set main config
    run_dir = cfg.main.run_dir
    seed = cfg.main.seed
    checkpoint = cfg.main.checkpoint

    # Set data config
    data_dir = cfg.data_config.data_dir
    Fs = cfg.data_config.sampling_frequency

    if data_dir is None:
        data_dir = Path("./data_burst")
    else:
        data_dir = Path(data_dir)
    data_dir = data_dir / "generated_data"
    data_dir.mkdir(exist_ok=True)

    # Set model training config
    sequence_length = cfg.model_config.sequence_length

    # Set seed (for reproducibility)
    pl.seed_everything(seed, workers=True)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _logger.info(f"Using device: {device}")

    # ---------- Model Training ---------- #

    if checkpoint is None:
        raise ValueError(
            "No checkpoint provided. Please provide a checkpoint to load a model."
        )
    else:
        # Load model
        _logger.info("Loading EphysGPT model ...")
        pl_module = EphysGPTModule.load_model(run_dir, checkpoint=checkpoint)
        pl_module.to(device)

        # Load tokenizer
        _logger.info("Loading tokenizer model ...")
        tkn_module = EphysTokenizerModule.load_model("models/tokenizer", checkpoint="latest")
        tkn_module.to(device)

    # ---------- Data Generation ---------- #

    # Set data generation hyperparameters
    n_subjects = 6
    subject_labels = np.arange(n_subjects)
    gen_batch_size = 3

    # Initialize generator
    generator = EphysGPTGenerator(
        model=pl_module.model,
        tokenizer=tkn_module,
    )

    # Generate data in batch
    for start in range(0, n_subjects, gen_batch_size):
        _logger.info(f"Generating data for subjects {start} to {start + gen_batch_size} ...")

        # Prepare extra labels
        subj_lbls = subject_labels[start : start + gen_batch_size]
        current_batch_size = len(subj_lbls)
        subject_labels_chunk = np.broadcast_to(
            subj_lbls[:, None],
            shape=(current_batch_size, sequence_length + 1),
        )

        # Generate data per subject
        generated_data = generator.generate_data(
            n_samples = (2 * 60 * Fs),  # 2 minutes
            top_p=0.99,
            temperature=1.0,
            batch_size=current_batch_size,
            extra_labels=[subject_labels_chunk.astype(np.int32)],
        )

        # Save generated data
        for i, subj_lbl in enumerate(subj_lbls):
            suffix = f"{subj_lbl:0{len(str(n_subjects))}d}"
            np.save(data_dir / f"gen_{suffix}.npy", generated_data[i])

        # Drop chunk outputs promptly to keep peak memory bounded
        del generated_data
        gc.collect()

    _logger.info("Generation complete.")


if __name__ == "__main__":
    main()
