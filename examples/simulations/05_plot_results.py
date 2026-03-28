"""Post-hoc analysis on Ephys-GPT generated data."""

# Import packages
import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
from pathlib import Path
from scipy import signal
from ephys_gpt.utils.processing import standardize, time_delay_embed
from ephys_gpt.utils.post_hoc import functional_connectivity, compute_aec
from ephys_gpt.utils.plotting import plot_history


_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    # ---------- Setting Up ---------- #

    # Set directories
    DATA_DIR = Path("./data_burst")  # alternatively, use: "./data_tde"
    MODEL_DIR = Path("./models/generator")

    PLOT_DIR = MODEL_DIR / "figures"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Set hyperparameters
    Fs = 100  # sampling frequency (Hz)

    # Get data file paths
    generated_data_paths = sorted((DATA_DIR / "generated_data").glob("gen_*.npy"))
    n_subjects = len(generated_data_paths)
    _logger.info(f"Found {n_subjects} generated data files.")
    original_data_paths = sorted(DATA_DIR.glob("*.npy"))[:n_subjects]  # match subjects

    # Load original and generated data
    original_data = [np.load(path) for path in original_data_paths]
    generated_data = [np.load(path) for path in generated_data_paths]

    # Standardize original data
    original_data = [standardize(data, axis=0) for data in original_data]

    # Truncate data to the same length
    gen_data_len = generated_data[0].shape[0]
    original_data = [data[:gen_data_len] for data in original_data]

    # ---------- Visualization ---------- #

    # 1. Plot training history
    _logger.info("Computing and plotting training history ...")

    with open(MODEL_DIR / "history.pkl", "rb") as f:
        hist = pickle.load(f)
    plot_history(hist, PLOT_DIR)

    # 2. Plot amplitude envelope correlation (AEC)
    _logger.info("Computing and plotting AECs ...")

    # Compute AEC
    original_aec = [compute_aec(data, Fs, n_window=101) for data in original_data]
    generated_aec = [compute_aec(data, Fs, n_window=101) for data in generated_data]

    # Average AEC across subjects
    original_aec = np.mean(original_aec, axis=0)
    generated_aec = np.mean(generated_aec, axis=0)

    # Remove diagonals
    original_aec = original_aec - np.eye(original_aec.shape[0])
    generated_aec = generated_aec - np.eye(generated_aec.shape[0])

    # Plot AEC for original and generated data
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    vmin = np.min([original_aec, generated_aec])
    vmax = np.max([original_aec, generated_aec])
    sns.heatmap(original_aec, ax=axes[0], cmap="viridis", vmin=vmin, vmax=vmax)
    sns.heatmap(generated_aec, ax=axes[1], cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Original AEC")
    axes[1].set_title("Generated AEC")
    fig.savefig(PLOT_DIR / "aec_comparison.png")
    plt.close(fig)

    # 3. Plot Static PSD Comparison
    _logger.info("Computing and plotting static PSDs ...")

    # Compute static PSDs
    def compute_psd(data, sampling_frequency):
        """Calculates PSD for each channel and average them."""
        nperseg = min(int(2 * sampling_frequency), data.shape[0])
        f, Pxx = signal.welch(
            data, fs=sampling_frequency, nperseg=nperseg, axis=0
        )
        Pxx = np.mean(Pxx, axis=1)  # average across channels
        return f, Pxx

    original_psds = [compute_psd(data, Fs) for data in original_data]
    generated_psds = [compute_psd(data, Fs) for data in generated_data]

    # Average PSDs across subjects
    psd_orig_mean = np.mean([p[1] for p in original_psds], axis=0)
    psd_orig_std = np.std([p[1] for p in original_psds], axis=0)

    psd_gen_mean = np.mean([p[1] for p in generated_psds], axis=0)
    psd_gen_std = np.std([p[1] for p in generated_psds], axis=0)

    if all(original_psds[0][0] == generated_psds[0][0]):
        f = original_psds[0][0]
    else:
        raise ValueError(
            "Frequency bins do not match between original and generated PSDs."
        )

    # Plot PSDs for original and generated data
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    ax.plot(f, psd_orig_mean, label="Original", color="black")
    ax.fill_between(
        f, psd_orig_mean - psd_orig_std, psd_orig_mean + psd_orig_std,
        color="black", alpha=0.2,
    )
    ax.plot(f, psd_gen_mean, label="Generated", color="red")
    ax.fill_between(
        f, psd_gen_mean - psd_gen_std, psd_gen_mean + psd_gen_std,
        color="red", alpha=0.2,
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title("Static PSD Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "psd_comparison.png")
    plt.close(fig)

    # 4. Plot TDE covariance matrices
    _logger.info("Computing and plotting TDE covariance matrices ...")

    # Concatenate original and generated data over subjects
    original_data_concat = np.concatenate(original_data, axis=0)
    generated_data_concat = np.concatenate(generated_data, axis=0)
    # shape: (n_subjects * n_samples, n_channels)

    # Plot TDE covariance matrices for original and generated data
    def plot_tde_cov(
        inputs,
        n_embeddings,
        titles,
    ):
        """Plots the TDE covariance of the inputs."""
        # Define a helper function
        def _get_tde_cov(data):
            data = standardize(data, axis=0)
            X = time_delay_embed(data, n_embeddings)
            X = standardize(X, axis=0)
            tde_cov = functional_connectivity(X, conn_type="cov")
            return tde_cov

        # Set data dimensions
        n_data = len(inputs)
        n_channels = inputs[0].shape[1]

        # Compute TDE covariance for each input
        tde_covs = []
        for n in range(n_data):
            tcov = _get_tde_cov(inputs[n])
            tcov_hollow = tcov.copy()

            # Zero out (n_embeddings x n_embeddings) diagonal blocks
            for c in range(n_channels):
                tcov_hollow[
                    c * n_embeddings:(c + 1) * n_embeddings,
                    c * n_embeddings:(c + 1) * n_embeddings
                ] = 0
            tde_covs.append(tcov_hollow)

        # Plot TDE covariance matrices
        fig, axes = plt.subplots(1, n_data, figsize=(6 * n_data, 5))
        if n_data == 1: axes = [axes]

        vmin = np.min([np.min(m) for m in tde_covs])
        vmax = np.max([np.max(m) for m in tde_covs])

        ticks = np.linspace(
            0, (n_channels - 1) * n_embeddings, n_channels
        ) + (n_embeddings // 2)
        tick_labels = np.arange(n_channels) + 1

        for i, mat in enumerate(tde_covs):
            sns.heatmap(
                mat, vmin=vmin, vmax=vmax, cmap="viridis", cbar=True, ax=axes[i]
            )
            axes[i].set_title(titles[i])

            # Only have a label for the start of each channel block
            axes[i].set_xticks(ticks)
            axes[i].set_xticklabels(tick_labels)
            axes[i].set_yticks(ticks)
            axes[i].set_yticklabels(tick_labels)

        fig.tight_layout()
        fig.savefig(PLOT_DIR / "tde_cov_comparison.png")
        plt.close(fig)

    plot_tde_cov(
        inputs=[original_data_concat, generated_data_concat],
        titles=["Original TDE Covariance", "Generated TDE Covariance"],
        n_embeddings=15,
    )

    # 5. Plot time series and spectrogram summary
    _logger.info("Computing and plotting time series and spectrogram ...")

    # Use the first subject and last channel for comparison
    ts_orig = original_data[0][:, 3]
    ts_gen = generated_data[0][:, 3]

    # Set a specific segment to plot (e.g., last 2000 samples)
    n_samples = min(2000, len(ts_orig))
    times = np.arange(n_samples) / Fs

    # Compute spectrograms first to get common vmin and vmax
    nperseg_spec = min(int(Fs / 2), n_samples)
    noverlap_spec = int(nperseg_spec * 0.9)
    f_spec_o, t_spec_o, Sxx_o = signal.spectrogram(
        ts_orig, fs=Fs, window="hann", nperseg=nperseg_spec, noverlap=noverlap_spec
    )
    f_spec_g, t_spec_g, Sxx_g = signal.spectrogram(
        ts_gen, fs=Fs, window="hann", nperseg=nperseg_spec, noverlap=noverlap_spec
    )

    vmin_spec = min(np.min(Sxx_o), np.min(Sxx_g))
    vmax_spec = max(np.max(Sxx_o), np.max(Sxx_g))

    # Compute PSDs
    nperseg_psd = min(int(2 * Fs), n_samples)
    f_psd_o, Pxx_o = signal.welch(ts_orig, fs=Fs, nperseg=nperseg_psd)
    f_psd_g, Pxx_g = signal.welch(ts_gen, fs=Fs, nperseg=nperseg_psd)

    # Initialize plotting objects
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex="row", sharey="row")
    fig.suptitle("Time Series and Spectrogram Comparison (Subject 1, Channel 4)")

    x_labels = ["Time (s)", "Time (s)", "Frequency (Hz)"]
    y_labels = ["Amplitude", "Amplitude", "Power"]
    titles_orig = ["Original Time Series", "Original Spectrogram", "Original PSD (Segment)"]
    titles_gen = ["Generated Time Series", "Generated Spectrogram", "Generated PSD (Segment)"]

    # Plot summary for original data
    axes[0, 0].plot(times, ts_orig[-n_samples:], color="black")  # time-series
    axes[1, 0].pcolormesh(
        t_spec_o, f_spec_o, Sxx_o,
        shading="gouraud", cmap="viridis",
        vmin=vmin_spec, vmax=vmax_spec,
    )  # spectrogram
    axes[2, 0].plot(f_psd_o, Pxx_o, color="black")  # PSD

    for i in range(3):
        axes[i, 0].set_xlabel(x_labels[i])
        axes[i, 0].set_ylabel(y_labels[i])
        axes[i, 0].set_title(titles_orig[i])

    # Plot summary for generated data
    axes[0, 1].plot(times, ts_gen[-n_samples:], color="red")  # time-series
    axes[1, 1].pcolormesh(
        t_spec_g, f_spec_g, Sxx_g,
        shading="gouraud", cmap="viridis",
        vmin=vmin_spec, vmax=vmax_spec,
    )  # spectrogram
    axes[2, 1].plot(f_psd_g, Pxx_g, color="red")  # PSD

    for i in range(3):
        axes[i, 1].set_xlabel(x_labels[i])
        axes[i, 1].set_ylabel(y_labels[i])
        axes[i, 1].set_title(titles_gen[i])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(PLOT_DIR / "time_series_spectrogram.png")
    plt.close(fig)
