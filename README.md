# EphysGPT

**EphysGPT** is a transformer-based foundation model pretrained on non-invasive human electrophysiological signals (MEG/EEG). It employs a sequential spatial and temporal attention mechanism to learn whole brain dynamics from the human brain activity.

🙋‍♂️ Please email SungJun Cho at sungjun.cho@ndcn.ox.ac.uk or simply open a GitHub issue if you have any questions or concerns.

## Table of Contents

- [Requirements](#-requirements)
- [Installation](#-installation)
- [Quick Start](#️-quick-start)
- [Project Structure](#-project-structure)
- [License](#-license)

## 🎯 Requirements

This project has the following dependencies:

* python=3.10
* pytorch=2.5.1
* pytorch-cuda=12.1
* pytorch-lightning=2.6.1

For a full list of required packages, please refer to `envs/egpt.yml`.

## 📌 Installation

To install `EphysGPT`, you can follow the steps below:

1. Clone the repository.
   ```bash
   git clone git@github.com:OHBA-analysis/EphysGPT.git
   cd EphysGPT
   ```
2. Create and activate a virtual environment.
   ```bash
   mamba env create -f envs/egpt.yml
   conda activate egpt
   ```
3. Install required packages.
   ```bash
   pip install -e .
   ```

In addition, if you do not already have a tokenizer or pre-tokenized data, you can install and use our external package:

4. Install `EphysTokenizer`.
   ```bash
   git clone git@github.com:OHBA-analysis/EphysTokenizer.git
   cd EphysTokenizer
   pip install -e .
   ```

> [!WARNING]
> Loading the Cam-CAN dataset as a PyTorch `Dataset` currently requires the `pnpl` and `pnpl-internal` packages.
> We are in the process of restructuring these packages, and `pnpl-internal` is not yet publicly available.
> An updated version of `pnpl` will be released soon. Meanwhile, users may integrate their own datasets and data loaders.

## ⚡️ Quick Start

The fastest way to get started is to review the example scripts in the `examples/simulations` directory.

These scripts demonstrate how to configure, train, and evaluate the models. Each training run generates a `figures` subdirectory containing basic post hoc analysis outputs.

## 📚 Project Structure

<details> <summary><strong>Directory Tree</strong></summary>

```
EphysGPT/
├── envs/
│   └── egpt.yml                        # Conda environment specification (dependencies for training and experiments)
│
├── ephys_gpt/
│   ├── configs/
│   │   ├── __init__.py                 # Exports Config wrapper and get_config() factory
│   │   └── config.py                   # Configuration dataclasses (EphysGPTConfig, InputEmbeddingConfig,
│   │                                   # TransformerDecoderConfig, TrainingConfig, LossConfig)
│   │
│   ├── data/
│   │   ├── datasets.py                 # SimulationDataset: data sequencing and preparation
│   │   ├── dataloader.py               # EphysGPTDataModule (LightningDataModule): train/val/test splits,
│   │   │                               # batching, distributed samplers
│   │   └── simulation.py               # TDEBurstSimulation: HMM-driven synthetic MEG/EEG data generation
│   │
│   ├── models/
│   │   ├── decoder/
│   │   │   ├── attention.py            # Attention, MultiHeadGASPAttention (temporal/spatial attention)
│   │   │   └── transformer_decoder.py  # TransformerDecoder, DecoderLayer (time + channel branches,
│   │   │                               # optional cross-attention, per-layer patchification)
│   │   │
│   │   ├── embeddings.py               # InputEmbeddingLayer (token + position + channel + extra-label)
│   │   ├── ephys_gpt.py                # EphysGPT (nn.Module): ShiftTokenLayer → Embedding → Decoder → Prediction Head;
│   │   │                               # EphysGPTModule (LightningModule): training loop, save/load
│   │   └── utils.py                    # Supporting layers (layer norm, feedforward blocks, etc.)
│   │
│   ├── optim/
│   │   ├── callbacks.py                # PyTorch Lightning training callbacks (logging, checkpointing, etc.)
│   │   ├── initializer.py              # Model weight initializations
│   │   ├── losses.py                   # Loss objective functions
│   │   └── optimizer.py                # Optimizer and LR schedulers
│   │
│   ├── inference/
│   │   └── generator.py                # EphysGPTGenerator: loads checkpoint, autoregressive sampling
│   │
│   ├── utils/
│   │   ├── array_ops.py                # Array manipulation utilities (sliding windows, etc.)
│   │   ├── plotting.py                 # Visualisation utilities
│   │   ├── post_hoc.py                 # Post hoc analysis utilities
│   │   ├── processing.py               # Data handling and processing utilities
│   │   └── sampling.py                 # Sampling and data generation utilities
│   │
│   └── typing.py                       # Custom dataclasses and type aliases
│
└── examples/
    └── simulations/
        ├── models/
        │   ├── generator/
        │   │   └── config.yaml         # Hydra config for EphysGPT training
        │   └── tokenizer/
        │       └── config.yaml         # Hydra config for tokenizer training
        │
        ├── 01_simulate_data.py         # Generate synthetic TDE burst data
        ├── 02_train_tokenizer.py       # Train EphysTokenizer on simulated data
        ├── 03_train_ephys_gpt.py       # Train EphysGPT on tokenised data (Hydra + Lightning)
        ├── 04_generate_data.py         # Autoregressive generation from trained checkpoint
        ├── 05_plot_results.py          # Visualize outputs and post-hoc analysis
```

</details>

## 🪪 License

Copyright (c) 2026 [OHBA Analysis Group](https://github.com/OHBA-analysis). `EphysGPT` is a free and open-source software licensed under the [MIT License](https://github.com/OHBA-analysis/EphysGPT/blob/main/LICENSE).
