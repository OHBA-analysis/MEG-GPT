"""Utility helper functions for visualization."""

# Import packages
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Dict


def plot_history(
    history: Dict[str, np.ndarray],
    save_dir: Optional[str] = None,
) -> None:
    """
    Plots the training and validation history.

    Parameters
    ----------
    history : Dict[str, np.ndarray]
        Dictionary containing the model training history.
    save_dir : str, optional
        Directory to save the training history plot.
        If None, the history will not be saved.
    """
    # Plot loss and accuracy
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    ax.plot(history["loss"], color="tab:blue", label="Loss")
    ax.plot(history["val_loss"], color="tab:blue", linestyle="--", label="Val Loss")

    ax_twinx = ax.twinx()
    ax_twinx.plot(history["top_1_acc"], color="tab:orange", label="Top-1 Acc.")
    ax_twinx.plot(
        history["val_top_1_acc"], linestyle="--", color="tab:orange", label="Val Top-1 Acc."
    )

    # Axis settings
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax_twinx.set_ylabel("Accuracy")
    ax.set_title("Training History")
    ax.legend(loc="upper left")
    ax_twinx.legend(loc="upper right")

    # Save plot
    if save_dir is not None:
        save_path = Path(save_dir) / "history.png"
        fig.savefig(save_path)
        print(f"Saved training history to {save_path}")
    else:
        plt.show()

    return None
