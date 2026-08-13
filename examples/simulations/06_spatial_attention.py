"""Analyze spatial information learned by a trained MEG-GPT model."""

# Import packages
import hydra
import logging
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from pathlib import Path

import pytorch_lightning as pl

from meg_gpt.data.dataloader import MEGGPTDataModule
from meg_gpt.data.datasets import SimulationDataset
from meg_gpt.models.meg_gpt import MEGGPTModule
from meg_gpt.utils.spatial import (
    get_channel_embedding_affinity,
    extract_channel_attention_matrix,
)


_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="models/generator", config_name="config")
def main(cfg: DictConfig):

    # ---------- Setting Up ---------- #

    # Set main config
    run_dir = cfg.main.run_dir
    seed = cfg.main.seed
    checkpoint = cfg.main.checkpoint

    plot_dir = Path(run_dir) / "figures" / "spatial_analysis"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Set data config
    data_dir = cfg.data_config.data_dir
    Fs = cfg.data_config.sampling_frequency
    
    if data_dir is None:
        data_dir = Path("./data_burst")
    else:
        data_dir = Path(data_dir)
    data_dir = Path(data_dir) / "tokenized_data"

    # Set model training config
    sequence_length = cfg.model_config.sequence_length
    n_heads = cfg.model_config.transformer_decoder.n_heads
    batch_size = cfg.model_config.training.batch_size
    val_split = cfg.model_config.training.val_split
    multi_gpu = cfg.model_config.training.multi_gpu

    # Set seed (for reproducibility)
    pl.seed_everything(seed, workers=True)

    # ---------- Load Model ---------- #

    if checkpoint is None:
        raise ValueError(
            "No checkpoint provided. Please provide a checkpoint to load a model."
        )
    else:
        # Load model
        _logger.info("Loading MEG-GPT model ...")
        pl_module = MEGGPTModule.load_model(run_dir, checkpoint=checkpoint)

    # ---------- Analyze Learned Spatial Information ---------- #

    _logger.info("Analyzing learned channel embeddings ...")

    # Extract learned channel embeddings
    embeddings = pl_module.model.get_embeddings()
    channel_embeddings = embeddings["channel"]
    # shape: (n_channels, channel_embedding_dim)

    # Compute cosine similarities between embedding vectors
    cos_sim = get_channel_embedding_affinity(
        channel_embeddings,
        metric_type="cosine_similarity",
    )  # shape: (n_channels, n_channels)

    # Plot cosine similarity
    fig, ax = plt.subplots()
    im = ax.imshow(cos_sim, cmap="viridis")
    fig.colorbar(im)
    n_channels = cos_sim.shape[0]
    ticks = range(n_channels)
    ax.set(
        xticks=ticks,
        yticks=ticks,
        xticklabels=[i + 1 for i in ticks],
        yticklabels=[i + 1 for i in ticks],
        xlabel="Channels",
        ylabel="Channels",
        title="Cosine Similarity",
    )
    fig.savefig(plot_dir / "cosine_similarity.png")
    plt.close(fig)

    # ---------- Analyze Channel Attention Weights ---------- #

    # Build validation dataloader using the same split/collate logic as training
    sim_data = SimulationDataset(
        data_path=data_dir,
        window_len=int(sequence_length + 1),
        sampling_frequency=Fs,
        info=["subject", "dataset"],
        standardize=False,
    )
    sim_datamodule = MEGGPTDataModule(
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
    sim_datamodule.setup("validate")
    val_dataloader = sim_datamodule.val_dataloader()

    _logger.info("Extracting channel attention matrices ...")

    # Extract channel attention matrices from all channel-attention-enabled layers
    # Here, we iterate over the full validation set.
    attn_weights = extract_channel_attention_matrix(
        pl_module,
        dataloader=val_dataloader,
        device="cpu",
    )
    # attn_weights: dict mapping layer_idx -> (N, H, L, C_q, C_k)
    # NOTE: Sequence length L may differ per layer.
    
    n_layers = len(attn_weights)

    # Plot channel attention matrices
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4), squeeze=False)
    for ax, (layer_idx, weights) in zip(axes[0], attn_weights.items()):
        # Average over samples, heads, and sequence length
        mean_weights = weights.mean(dim=(0, 1, 2)).numpy()  # (C_q, C_k)
        
        im = ax.imshow(mean_weights, cmap="viridis", vmin=0.0)
        fig.colorbar(im, ax=ax)
        ax.set(
            xticks=range(mean_weights.shape[1]),
            yticks=range(mean_weights.shape[0]),
            xticklabels=[i + 1 for i in range(mean_weights.shape[1])],
            yticklabels=[i + 1 for i in range(mean_weights.shape[0])],
            xlabel="Key Channel",
            ylabel="Query Channel",
            title=f"Channel Attention\n(Layer {layer_idx})"
        )

    fig.tight_layout()
    fig.savefig(plot_dir / "channel_attention_weights.png")
    plt.close(fig)

    # Plot per-head attention matrices for each layer.
    # Each figure has one row per head, one column per layer.
    fig, axes = plt.subplots(
        n_heads, n_layers,
        figsize=(5 * n_layers, 4 * n_heads),
        squeeze=False,
    )
    for col, (layer_idx, weights) in enumerate(attn_weights.items()):
        # Average over samples and sequence length, keep heads separate
        head_weights = weights.mean(dim=(0, 2)).numpy()  # (H, C_q, C_k)
        for head_idx in range(n_heads):
            ax = axes[head_idx][col]
            w = head_weights[head_idx]  # (C_q, C_k)
            im = ax.imshow(w, cmap="viridis", vmin=0.0)
            fig.colorbar(im, ax=ax)
            ax.set(
                xticks=range(w.shape[1]),
                yticks=range(w.shape[0]),
                xticklabels=[i + 1 for i in range(w.shape[1])],
                yticklabels=[i + 1 for i in range(w.shape[0])],
                xlabel="Key Channel",
                ylabel="Query Channel",
                title=f"Layer {layer_idx} | Head {head_idx + 1}",
            )

    fig.tight_layout()
    fig.savefig(plot_dir / "channel_attention_weights_per_head.png")
    plt.close(fig)

    _logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
