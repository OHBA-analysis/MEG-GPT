"""Utility helper functions for post-hoc analysis."""

# Import packages
import logging
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from pqdm.threads import pqdm
from sklearn.covariance import LedoitWolf
from tqdm.auto import trange
from typing import Optional, Dict
from meg_gpt.utils import array_ops
from meg_gpt.utils.processing import (
    amplitude_envelope, moving_average, standardize, temporal_filter
)


_logger = logging.getLogger(__name__)


def functional_connectivity(
    data: np.ndarray,
    conn_type: str = "corr",
    demean: bool = True,
    window_length: int = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Calculates functional connectivity networks.

    This function uses the `LedoitWolf <https://scikit-learn.org/stable\
    /modules/generated/sklearn.covariance.LedoitWolf.html>`_ estimator
    to calculate covariance. If another measure is requested it is calculated
    from this covariance matrix.

    Parameters
    ----------
    data : np.ndarray
        Time series data. Shape must be (n_sessions, n_samples, n_channels)
        or (n_samples, n_channels).
    conn_type : str or func, optional
        What measure should we use? Must be one of:
            - "cov": covariance.
            - "pcov": partial covariance.
            - "corr": Pearson correlation.
            - "pcorr": partial correlation.
            - "prec": precision (inverse of the covariance matrix).
        Or can be a function that accepts a (n_samples, n_channels) array and
        returns a (n_channels, n_channels) array.
    demean : bool, optional
        Should we demean the data before calculating the functional connectivity?
    window_length : int, optional
        Window length (in samples) to split the data and average the functional
        connectivity network for. If None, no averaging is done. Windows are
        non-overlapping.
    n_jobs : int, optional
        Number of jobs.

    Returns
    -------
    fc : np.ndarray
        Functional connectivity network.
        Shape is (n_sessions, n_channels, n_channels) or (n_channels, n_channels).
    """

    # Validation
    if isinstance(conn_type, str):
        allowed_conn_types = ["cov", "pcov", "corr", "pcorr", "prec"]
        if conn_type not in allowed_conn_types:
            raise ValueError(
                f"conn_type must be one of: {allowed_conn_types}, got: {conn_type}."
            )
    elif not callable(conn_type):
        raise TypeError("conn_type must be a str, list or callable function.")

    if isinstance(data, np.ndarray):
        data = [data]

    def _calc_cov(x):
        # Function to calculate covariance
        lw = LedoitWolf(assume_centered=(not demean), store_precision=False)

        # Use all data to calculate the covariance
        if window_length is None:
            lw.fit(x)
            return lw.covariance_

        # Validation
        L = int(window_length)
        T = x.shape[0]
        n_windows = T // L
        if n_windows == 0:
            raise ValueError("window_length is larger than number of samples.")

        # Average the covariance for multiple windows
        covs = []
        for w in range(n_windows):
            start = w * L
            end = start + L
            chunk = x[start:end]
            lw.fit(chunk)
            covs.append(lw.covariance_)

        return np.mean(covs, axis=0)

    # Covariance
    if conn_type == "cov":
        measure = _calc_cov

    # Partial covariance
    elif conn_type == "pcov":

        def measure(x):
            cov = _calc_cov(x)
            return array_ops.cov2partialcov(cov)

    # Correlation
    elif conn_type == "corr":

        def measure(x):
            cov = _calc_cov(x)
            return array_ops.cov2corr(cov)

    # Partial correlation
    elif conn_type == "pcorr":

        def measure(x):
            cov = _calc_cov(x)
            return array_ops.cov2partialcorr(cov)

    # Precision
    elif conn_type == "prec":

        def measure(x):
            cov = _calc_cov(x)
            return np.linalg.inv(cov)

    # User specified function
    else:
        measure = conn_type

    # Calculate networks
    if len(data) == 1:
        # Don't need to parallelise
        results = [measure(data[0])]

    elif n_jobs == 1:
        # We have multiple arrays but we're running in serial
        results = []
        for i in trange(len(data), desc="Calculating FC"):
            results.append(measure(data[i]))

    else:
        # Calculate in parallel
        _logger.info("Calculating FC")
        results = pqdm(data, measure, n_jobs=n_jobs)

    return np.squeeze(results)


def get_history(
    log_dir: str, save_dir: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Loads training history from a CSV log file.

    Parameters
    ----------
    log_dir : str
        Directory containing the log files.
    save_dir : str, optional
        Directory to save the training history pickle file.
        If None, the history will not be saved.

    Returns
    -------
    history : Dict[str, np.ndarray]
        Dictionary containing the training history.
    """
    # Read metric log file
    log_path = Path(log_dir) / "metrics.csv"
    if log_path.exists():
        df = pd.read_csv(log_path)
    else:
        raise FileNotFoundError(
            f"No metrics.csv found in {log_dir}. Cannot load training history."
        )

    # Initialize history
    history = {}
    to_npy = lambda x: x.dropna().to_numpy()
    # NOTE: We use `.dropna()` to remove the interleaved NaN values. During training,
    #       validation metrics are logged as NaN; during validation, training metrics
    #       are logged as NaN.

    # Collect training history
    history["loss"] = to_npy(df["train/loss"])
    history["cross_entropy_loss"] = to_npy(df["train/cross_entropy_loss"])
    for col in df.columns:
        if col.startswith("train/") and col.endswith("_acc"):
            key = col.replace("train/", "")
            history[key] = to_npy(df[col])
        if col == "train/learning_rate":
            history["learning_rate"] = to_npy(df[col])

    # Collect validation history
    history["val_loss"] = to_npy(df["val/loss"])
    history["val_cross_entropy_loss"] = to_npy(df["val/cross_entropy_loss"])
    for col in df.columns:
        if col.startswith("val/") and col.endswith("_acc"):
            key = col.replace("val/", "val_")
            history[key] = to_npy(df[col])

    # Save history
    if save_dir:
        save_path = Path(save_dir) / "history.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(history, f)

    return history


def compute_aec(
    data: np.ndarray,
    sampling_frequency: int,
    low_freq: Optional[float] = None,
    high_freq: Optional[float] = None,
    n_window: Optional[int] = None,
    do_standardize: bool = False,
) -> np.ndarray:
    """
    Computes the amplitude envelope correlation (AEC) of the input data.

    Parameters
    ----------
    data : np.ndarray
        Input data array to compute AEC. Shape should be (n_samples, n_channels).
    sampling_frequency : int
        Sampling frequency of the input data.
    low_freq : float
        Lower frequency bound for bandpass filtering.
    high_freq : float
        Upper frequency bound for bandpass filtering.
    n_window : int
        Window size for moving average in samples. Must be odd.
    do_standardize : bool
        Whether to standardize the input data first.

    Returns
    -------
    aec : np.ndarray
        AEC matrix of shape (n_channels, n_channels).
    """
    # Standardize data
    if do_standardize:
        data = standardize(data, axis=0)

    # Apply bandpass filter
    if low_freq is not None or high_freq is not None:
        data_filt = temporal_filter(
            x=data,
            sampling_frequency=sampling_frequency,
            low_freq=low_freq,
            high_freq=high_freq,
        )
    else:
        data_filt = data

    # Compute amplitude envelope
    data_env = amplitude_envelope(data_filt)

    # Apply moving average
    if n_window is not None:
        data_env_ma = moving_average(data_env, n_window=n_window)
    else:
        data_env_ma = data_env

    # Compute correlation
    aec = functional_connectivity(data_env_ma, conn_type="corr")

    return aec
