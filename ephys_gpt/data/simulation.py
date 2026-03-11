"""
Data simulation using a bursting TDE-based model.

Mathematical Notations:
    - C: number of channels
    - E: number of embeddings
    - T: number of time points
    - N: number of subjects
    - M: number of modes
"""

# Import packages
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import warnings
from tqdm import tqdm
from typing import List, Optional, Tuple, Union
from ephys_gpt.utils.array_ops import get_one_hot
from ephys_gpt.utils.processing import time_delay_embed, standardize
from ephys_gpt.utils.post_hoc import functional_connectivity


_logger = logging.getLogger(__name__)


class HMM:
    """
    Hidden Markov Model (HMM) for time series data.

    Parameters
    ----------
    trans_prob : np.ndarray or str
        Transition probability matrix as a numpy array or a string
        ('sequence' or 'uniform') to generate a transition probability matrix.
    stay_prob : float, optional
        Used to generate the transition probability matrix when `trans_prob`
        is a string. Must be between 0 and 1.
    n_states : int, optional
        Number of states. Needed when `trans_prob` is a string.
    """
    def __init__(
        self,
        trans_prob: Union[np.ndarray, str],
        stay_prob: Optional[float] = None,
        n_states: Optional[int] = None,
    ):
        if isinstance(trans_prob, np.ndarray):
            if trans_prob.ndim != 2:
                raise ValueError("trans_prob must be a 2D array.")

            if trans_prob.shape[0] != trans_prob.shape[1]:
                raise ValueError("trans_prob must be a square matrix.")

            # Check if the rows of the transition probability matrix sum to one
            # We allow a small error (1e-12) as rounding errors
            row_sums = trans_prob.sum(axis=1)
            col_sums = trans_prob.sum(axis=0)
            if not all(np.isclose(row_sums, 1)):
                if all(np.isclose(col_sums, 1)):
                    trans_prob = trans_prob.T
                    warnings.warn(
                        "Rows of trans_prob matrix must sum to 1. Transpose taken.",
                        RuntimeWarning,
                    )
                else:
                    raise ValueError("Rows of trans_prob must sum to 1.")

            self.trans_prob = trans_prob

        elif isinstance(trans_prob, str):
            # Validation
            if trans_prob not in ["sequence", "uniform"]:
                raise ValueError(
                    "trans_prob must be a np.ndarray, 'sequence', or 'uniform'."
                )

            # Special case of there being only one state
            if n_states == 1:
                self.trans_prob = np.ones([1, 1])

            # Sequential transition probability matrix
            elif trans_prob == "sequence":
                if stay_prob is None or n_states is None:
                    raise ValueError(
                        "If trans_prob is 'sequence', stay_prob and n_states "
                        "must be passed."
                    )
                self.trans_prob = self.construct_sequence_trans_prob(
                    stay_prob, n_states
                )

            # Uniform transition probability matrix
            elif trans_prob == "uniform":
                if n_states is None:
                    raise ValueError(
                        "If trans_prob is 'uniform', n_states must be passed."
                    )
                if stay_prob is None:
                    stay_prob = 1.0 / n_states
                self.trans_prob = self.construct_uniform_trans_prob(
                    stay_prob,
                    n_states,
                )

        elif trans_prob is None and n_states == 1:
            self.trans_prob = np.ones([1, 1])

        # Infer the number of states
        self.n_states = self.trans_prob.shape[0]

    @staticmethod
    def construct_sequence_trans_prob(stay_prob, n_states):
        trans_prob = np.zeros([n_states, n_states])
        np.fill_diagonal(trans_prob, stay_prob)
        np.fill_diagonal(trans_prob[:, 1:], 1 - stay_prob)
        trans_prob[-1, 0] = 1 - stay_prob
        return trans_prob

    @staticmethod
    def construct_uniform_trans_prob(stay_prob, n_states):
        single_trans_prob = (1 - stay_prob) / (n_states - 1)
        trans_prob = np.ones((n_states, n_states)) * single_trans_prob
        trans_prob[np.diag_indices(n_states)] = stay_prob
        return trans_prob

    def generate_states(self, n_samples: int) -> np.ndarray:
        """
        Generates a sequence of states based on the transition probability matrix.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.

        Returns
        -------
        state_time_courses : np.ndarray
            A 2D array containing the one-hot encoded states.
            Shape is (n_samples, n_states).
        """
        rands = [
            iter(np.random.choice(self.n_states, size=n_samples, p=self.trans_prob[i]))
            for i in range(self.n_states)
        ]
        states = np.zeros(n_samples, int) # time course always starts from state 0
        for sample in range(1, n_samples):
            states[sample] = next(rands[states[sample - 1]])
        return get_one_hot(states, n_states=self.n_states)


class TDEBurstSimulation:
    """
    Simulates data using a bursting TDE-based generative model.

    Parameters
    ----------
    true_tde_covs : List[np.ndarray]
        List of n_modes items, where each item is a (CE, CE)
        covariance matrix.
    n_subjects : int, optional
        Number of subjects.
    n_samples : int, optional
        Number of samples per subject per channel.
    n_embeddings: int, optional
        Number of embeddings
    sampling_frequency : int, optional
        Sampling frequency in Hz.
    stay_prob : float, optional
        Probability of staying in the same state in the HMM.
    data_dir : str, optional
        Directory to save simulated data.
    rho : float, optional
        Ridge (Tikhonov) regularization parameter.
        It is added to the diagonal of `Sig11` before computing its 
        pseudo-inverse to improve numerical stability and guarantee
        positive definiteness.
    """
    def __init__(
        self,
        true_tde_covs: List[np.ndarray],
        n_subjects: int = 10,
        n_samples: int = 1000,
        n_embeddings: int = 1,
        sampling_frequency: int = 100,
        stay_prob: float = 0.98,
        data_dir: Optional[str] = None,
        rho: float = 0.1,
    ):
        # Set parameters
        self.true_tde_covs = true_tde_covs
        self.n_subjects = n_subjects
        self.n_samples = n_samples
        self.n_embeddings = n_embeddings
        self.sampling_frequency = sampling_frequency
        self.stay_prob = stay_prob
        self.data_dir = data_dir or "sim_data"
        self.rho = rho

        # Create data directory if it does not exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

        # Get data dimensions
        self.n_channels = self.true_tde_covs[0].shape[0] // self.n_embeddings
        self.n_modes = len(self.true_tde_covs)

    def _generate_data(self, tde_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates data from a TDE covariance matrix.

        Parameters
        ----------
        tde_cov : np.ndarray
            TDE covariance matrix of shape (CE, CE).
            Order of rows/columns is assumed to correspond to
            tde_cov being made up of blocks of E x E matrices.

        Returns
        -------
        gen_data : np.ndarray
            Generated data of shape (T, C).
        tde_cov : np.ndarray
            Reshaped TDE covariance matrix used for generation.
            Shape is (EC, EC).
        """
        # Validation
        if tde_cov.ndim != 2 or tde_cov.shape[0] != tde_cov.shape[1]:
            raise ValueError("TDE covariance matrix must be a 2D square matrix.")

        if tde_cov.shape[0] / self.n_embeddings != self.n_channels:
            raise ValueError("Inconsistent TDE covariance matrix dimensions.")

        # Set parameters
        C = self.n_channels
        E = self.n_embeddings
        T = self.n_samples
        rho = self.rho

        # Reshape TDE covariance matrix
        tde_cov = tde_cov.reshape(C, E, C, E)
        tde_cov = tde_cov[:, ::-1, :, ::-1]  # reverse embedding dimensions
        tde_cov = np.transpose(tde_cov, [1, 0, 3, 2])
        tde_cov = tde_cov.reshape(E * C, E * C)
        # NOTE: Reversing embeddings converts lag-order to chronological to align with the mathematical partitioning.
        #       `time_delay_embed` outputs in lag-order [Present, Past-1, Past-2]. However, our Schur complement expects
        #       chronological order [Past-2, Past-1, Present] so that `Sig22 = tde_cov[-C:, -C:]` correctly isolates the "Present" timepoint.
        #       Without this flip, the AR model generates the time series backward.

        # Partition covariance matrix
        Sig11 = tde_cov[:-C, :-C]  # ((E-1)*C, (E-1)*C)
        Sig12 = tde_cov[:-C, -C:]  # ((E-1)*C, C)
        Sig21 = tde_cov[-C:, :-C]  # (C, (E-1)*C)
        Sig22 = tde_cov[-C:, -C:]  # (C, C)
        # See "Conditional distributions" in:
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution

        # Compute inverse of Sig11
        invSig11 = np.linalg.pinv(Sig11 + np.eye(Sig11.shape[0]) * rho)
        # adds ridge to guarantee stable matrix inversion

        # Compute Schur complement of Sig11
        Sig = (Sig22 - Sig21 @ invSig11 @ Sig12) + np.eye(Sig22.shape[0]) * 0.001

        # Compute projection matrix
        proj = Sig21@invSig11

        # Set initial conditions
        x_past = np.random.multivariate_normal(
            mean=np.zeros(tde_cov.shape[0]),
            cov=tde_cov,
            size=1,
        )  # shape: (1, EC)
        x_past = x_past[:, :-C].T  # shape: ((E-1)*C, 1)

        # Generate data
        gen_data = np.zeros((T, C))
        for t in range(T):
            # Sample next values
            mu = proj @ x_past
            x_t = np.expand_dims(
                np.random.multivariate_normal(mu.squeeze(), Sig), axis=1
            )  # shape: (C, 1)
            gen_data[t, :] = x_t.squeeze()

            # Shift past values and append
            x_past = np.concatenate([x_past[C:, :], x_t])  # shape: ((E-1)*C, 1)

        # Standardize
        gen_data = standardize(gen_data, axis=0)

        return gen_data, tde_cov

    def simulate(
        self,
        obs_noise_std: float = 0.0,
        save: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates data with bursts (and noise) using a Hidden Markov Model (HMM).
        For each subject, activity patterns are generated per mode and combined.

        Parameters
        ----------
        obs_noise_std : float, optional
            Standard deviation of the Gaussian observation noise to add.
            Defaults to 0.
        save : bool, optional
            Whether to save the simulated data and ground truth mode time courses.

        Returns
        -------
        data : np.ndarray
            Simulated data with a shape (N, T, C).
        mtc : np.ndarray
            Mode time courses with a shape (N, T, M).
        """
        # Initialize HMM
        hmm = HMM(
            trans_prob="uniform",
            stay_prob=self.stay_prob,
            n_states=self.n_modes,
        )

        # Simulate synthetic signals
        data = np.zeros((self.n_subjects, self.n_samples, self.n_channels))
        mtc = np.zeros((self.n_subjects, self.n_samples, self.n_modes))

        for n in tqdm(range(self.n_subjects), desc="Simulating bursts", total=self.n_subjects):
            mtc[n, :, :] = hmm.generate_states(self.n_samples)

            for m in range(self.n_modes):
                activity, _ = self._generate_data(self.true_tde_covs[m])
                data[n, :, :] += np.expand_dims(mtc[n, :, m], axis=1) * activity

        _logger.info("Data simulation complete.")

        # Add observation white noise
        if obs_noise_std > 0:
            noise = np.random.normal(0, obs_noise_std, size=data.shape)
            data += noise

        # Save simulated data and ground truth mode time courses
        if save:
            os.makedirs(f"{self.data_dir}/ground_truth", exist_ok=True)
            for n in range(self.n_subjects):
                np.save(f"{self.data_dir}/x_{n:0{len(str(self.n_subjects))}d}.npy", data[n])
                np.save(f"{self.data_dir}/ground_truth/mode_tcs_subj{n:0{len(str(self.n_subjects))}d}.npy", mtc[n])

            _logger.info("Data saving complete.")

        return data, mtc

    def plot_summary(
        self,
        idx_start: int = 0,
        idx_end: int = None,
        channels_to_plot: Union[list, np.ndarray] = None,
        conn_type: str = "cov",
        plot_dir: str = None,
    ) -> None:
        """
        Plots a summary of the simulated data.
        
        Figures include:
            - mode time courses
            - channel-wise simulated data
            - power spectral density
            - TDE connectivity matrices

        Parameters
        ----------
        idx_start : int, optional
            Start index for plotting.
        idx_end : int, optional
            End index for plotting.
        channels_to_plot : list, optional
            List of channels to plot. If None, all channels will be plotted.
        conn_type : str, optional
            Type of connectivity to plot (e.g., "cov" or "corr").
        plot_dir : str, optional
            Directory to save plots.
        """
        # Get file paths
        data_files = [
            f"{self.data_dir}/x_{n:0{len(str(self.n_subjects))}d}.npy"
            for n in range(self.n_subjects)
        ]
        mtc_files = [
            f"{self.data_dir}/ground_truth/mode_tcs_subj{n:0{len(str(self.n_subjects))}d}.npy"
            for n in range(self.n_subjects)
        ]

        # Specify subjects to use as an example
        sub1 = 0
        sub2 = self.n_subjects - 1

        # Load data
        mtc_sub1 = np.load(mtc_files[sub1])
        data_sub1 = np.load(data_files[sub1])
        data_sub2 = np.load(data_files[sub2])

        # Plot mode time courses
        _plot_mtc(
            mtc=mtc_sub1,
            idx_start=idx_start,
            idx_end=idx_end,
            sampling_frequency=self.sampling_frequency,
            plot_dir=plot_dir,
        )

        # Plot data
        _plot_channel_data(
            data1=data_sub1,
            data2=data_sub2,
            sel_ch=channels_to_plot,
            idx_start=idx_start,
            idx_end=idx_end,
            sampling_frequency=self.sampling_frequency,
            plot_dir=plot_dir,
        )

        # Plot PSDs
        _plot_psds(
            data1=data_sub1,
            data2=data_sub2,
            sel_ch=channels_to_plot,
            sampling_frequency=self.sampling_frequency,
            plot_dir=plot_dir,
        )

        # Plot ground truth and generated TDE connectivity matrices
        _plot_tde_conn(
            data=data_sub1,
            mtc=mtc_sub1,
            true_tde_conns=self.true_tde_covs,
            n_embeddings=self.n_embeddings,
            conn_type=conn_type,
            plot_dir=plot_dir,
        )

        _logger.info(f"Visualizations complete.")


# -----------------------------
# Visualization for Simulations
# -----------------------------

def _plot_mtc(
    mtc: np.ndarray,
    idx_start: int = 0,
    idx_end: int = None,
    sampling_frequency: int = 1,
    plot_dir: str = None,
) -> None:
    """
    Plots mode time courses.

    Parameters
    ----------
    mtc : np.ndarray
        Mode time courses to plot.
    idx_start : int
        Start index for slicing.
    idx_end : int
        End index for slicing.
    sampling_frequency : int
        Sampling frequency of the data.
    plot_dir : str
        Directory to save plots.
    """
    # Validate inputs
    if mtc.ndim != 2:
        raise ValueError("mtc must be a 2D array.")

    # Slice mode time courses
    T, M = mtc.shape
    mtc = mtc[idx_start:idx_end, :]

    # Initialize figures
    fig, axes = plt.subplots(M, 1, figsize=(10, 2 * M), sharex=True)
    
    if not hasattr(axes, "__iter__"):  # if a single Axes is returned
        axes = [axes]

    x_times = np.arange(T) / sampling_frequency  # time vector
    x_times = x_times[idx_start:idx_end]

    # Plot mode time courses
    for i, ax in enumerate(axes):  # iterate over modes
        ax.plot(x_times, mtc[:, i])
        ax.set_ylim([-0.1, 1.1])
        ax.set_yticks([0, 1])
        ax.set_ylabel(f"Mode {i}")
    axes[-1].set_xlabel("Time (s)")

    fig.suptitle("Mode Time Courses")
    fig.tight_layout()

    # Save figure
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(
            f"{plot_dir}/mode_time_courses.png", bbox_inches="tight", dpi=300
        )
        plt.close(fig)


def _plot_channel_data(
    data1: np.ndarray,
    data2: np.ndarray,
    sel_ch: List[int] = None,
    idx_start: int = 0,
    idx_end: int = None,
    sampling_frequency: int = 1,
    plot_dir: str = None,
) -> None:
    """
    Plots channel-wise data for two subjects.

    Parameters
    ----------
    data1 : np.ndarray
        Data for subject 1.
    data2 : np.ndarray
        Data for subject 2.
    sel_ch : List[int], optional
        Channel indices to plot.
    idx_start : int, optional
        Start index for slicing.
    idx_end : int, optional
        End index for slicing.
    sampling_frequency : int, optional
        Sampling frequency of the data.
    plot_dir : str, optional
        Directory to save plots.
    """
    # Validation
    if data1.shape[1] != data2.shape[1]:
        raise ValueError("Input data must have the same number of channels.")

    # Set channels to plot
    selected_channels = sel_ch or np.arange(data1.shape[1])
    n_channels = len(selected_channels)

    # Initialize figures
    fig, axes = plt.subplots(n_channels, 1, figsize=(25, 5 * n_channels), sharex=True)

    x_times = np.arange(min([data1.shape[0], data2.shape[0]])) / sampling_frequency
    x_times = x_times[idx_start:idx_end]

    # Plot channel-wise data time series
    for i, ch in enumerate(selected_channels):
        axes[i].plot(
            x_times,
            data1[idx_start:idx_end, ch],
            color="tab:red",
            label="Subject 1" if i == 0 else None,
        )
        axes[i].plot(
            x_times,
            data2[idx_start:idx_end, ch],
            color="tab:blue",
            label="Subject 2" if i == 0 else None,
        )
        axes[i].set_ylabel(f"Channel {ch}")
        if i == 0: axes[i].legend()
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Simulated data")
    
    # Save figure
    fig.tight_layout()
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(
            f"{plot_dir}/data_chan.png", bbox_inches="tight", dpi=300
        )
        plt.close(fig)


def _plot_psds(
    data1: np.ndarray,
    data2: np.ndarray,
    sel_ch: List[int] = None,
    sampling_frequency: int = 1,
    plot_dir: str = None,
) -> None:
    """
    Plots the power spectral density (PSD) for two subjects.

    Parameters
    ----------
    data1 : np.ndarray
        Data for subject 1.
    data2 : np.ndarray
        Data for subject 2.
    sel_ch : List[int], optional
        Channel indices to plot.
    sampling_frequency : int, optional
        Sampling frequency of the data.
    plot_dir : str, optional
        Directory to save plots.
    """
    # Validation
    if data1.shape[1] != data2.shape[1]:
        raise ValueError("Input data must have the same number of channels.")

    # Set channels to plot
    selected_channels = sel_ch or np.arange(data1.shape[1])
    n_channels = len(selected_channels)

    # Initialize figures
    fig, axes = plt.subplots(n_channels, 1, figsize=(15, 5 * n_channels))

    for i, ch in enumerate(selected_channels):
        axes[i].psd(
            data1[:, ch],
            Fs=sampling_frequency,
            NFFT=1024,
            color="tab:red",
            label="Subject 1" if i == 0 else None,
        )
        axes[i].psd(
            data2[:, ch],
            Fs=sampling_frequency,
            NFFT=1024,
            color="tab:blue",
            label="Subject 2" if i == 0 else None,
        )
        axes[i].set_ylabel(f"Channel {ch}")
        if i == 0: axes[i].legend()
    axes[-1].set_xlabel("Frequency (Hz)")
    axes[0].set_title("Power spectral density")

    # Save figure
    fig.tight_layout()
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(
            f"{plot_dir}/psd.png", bbox_inches="tight", dpi=300
        )
        plt.close(fig)


def _plot_tde_conn(
    data: np.ndarray,
    mtc: np.ndarray,
    true_tde_conns: List[np.ndarray],
    n_embeddings: int,
    conn_type: str = "corr",
    plot_dir: str = None,
) -> None:
    """
    Plots TDE connectivity matrices.

    Parameters
    ----------
    data : np.ndarray
        The input data array. Shape must be (T, C).
    mtc : np.ndarray
        The mode time courses. Shape must be (T, M).
    true_tde_conns : List[np.ndarray]
        The true TDE connectivity matrices for each mode.
        Each element has shape (CE, CE).
    n_embeddings : int
        The number of embeddings.
    conn_type : str, optional
        The type of connectivity to compute.
    plot_dir : str, optional
        Directory to save plots.
    """
    # Get number of modes and channels
    n_modes = len(true_tde_conns)
    n_channels = data.shape[1]

    # Compute TDE connectivity matrices for the generated data
    def _get_tde_conn(x: np.ndarray, n_embeddings: int) -> np.ndarray:
        x_tde = time_delay_embed(x, n_embeddings)
        x_tde = standardize(x_tde, axis=0)
        tde_conn = functional_connectivity(x_tde, conn_type=conn_type)
        return tde_conn

    gen_tde_conns = [
        _get_tde_conn(data[mtc[:, m] == 1, :], n_embeddings)
        for m in range(n_modes)
    ]

    # Initialize figures
    fig, axes = plt.subplots(2, n_modes, figsize=(5 * n_modes, 8))

    for i in range(n_modes):
        true_conn = true_tde_conns[i].copy()
        gen_conn = gen_tde_conns[i].copy()

        # Zero out (n_embeddings x n_embeddings) diagonal blocks
        for c in range(n_channels):
            true_conn[
                c * n_embeddings : (c + 1) * n_embeddings,
                c * n_embeddings : (c + 1) * n_embeddings
            ] = 0
            gen_conn[
                c * n_embeddings : (c + 1) * n_embeddings,
                c * n_embeddings : (c + 1) * n_embeddings
            ] = 0

        # Plot connectivity matrices
        sns.heatmap(
            true_conn, 
            cmap="viridis",
            vmin=np.min(true_conn),
            vmax=np.max(true_conn),
            cbar=True,
            ax=axes[0, i],
        )
        axes[0, i].set_title(f"Mode {i} True")

        sns.heatmap(
            gen_conn,
            cmap="viridis",
            vmin=np.min(gen_conn),
            vmax=np.max(gen_conn),
            cbar=True,
            ax=axes[1, i],
        )
        axes[1, i].set_title(f"Mode {i} Gen.")

    # Save figure
    fig.tight_layout()
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(
            f"{plot_dir}/tde_conn.png", bbox_inches="tight", dpi=300
        )
        plt.close(fig)
