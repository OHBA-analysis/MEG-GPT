"""Utility helper functions for data processing."""

# Import packages
import numpy as np
import torch
from scipy import signal
from typing import Optional
from meg_gpt.utils.array_ops import sliding_window_view


def standardize(
    x: np.ndarray,
    axis: int = 0,
    create_copy: bool = True
) -> np.ndarray:
    """
    Standardizes a time series.

    Returns a time series with zero mean and unit variance.

    Parameters
    ----------
    x : np.ndarray
        Time series data. Shape must be (n_samples, n_channels).
    axis : int, optional
        Axis on which to perform the transformation.
    create_copy : bool, optional
        Should we return a new array containing the standardized data or modify
        the original time series array?

    Returns
    -------
    X :  np.ndarray
        Standardized data.
    """
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    if create_copy:
        return (np.copy(x) - mean) / std
    return (x - mean) / std


def time_delay_embed(
    x: np.ndarray, n_embeddings: int
) -> np.ndarray:
    """
    Performs time delay embedding.

    Parameters
    ----------
    x : np.ndarray
        Time series data. Shape must be (n_samples, n_channels).
    n_embeddings : int
        Number of samples in which to shift the data.

    Returns
    -------
    X : np.ndarray
        Time embedded data. Shape is (n_samples - n_embeddings + 1,
        n_channels * n_embeddings).
    """
    if n_embeddings % 2 == 0:
        raise ValueError("n_embeddings must be an odd number.")

    # Shape of time embedded data
    te_shape = (x.shape[0] - (n_embeddings - 1), x.shape[1] * n_embeddings)

    # Perform time embedding
    X = (
        sliding_window_view(x=x, window_shape=te_shape[0], axis=0)
        .T[..., ::-1]
        .reshape(te_shape)
    )

    return X


def temporal_filter(
    x: np.ndarray,
    sampling_frequency: float,
    low_freq: Optional[float] = None,
    high_freq: Optional[float] = None,
    order: int = 5,
) -> np.ndarray:
    """
    Applies IIR Butterworth filtering across the time axis.

    Parameters
    ----------
    x : np.ndarray
        Time series data. Shape must be (n_samples, n_channels).
    sampling_frequency : float
        Sampling frequency in Hz.
    low_freq : float
        Frequency in Hz for a high pass filter.
    high_freq : float
        Frequency in Hz for a low pass filter.
    order : int, optional
        Order for a butterworth filter.

    Returns
    -------
    x_filt : np.ndarray
        Filtered time series. Shape is (n_samples, n_channels).
    """
    # Input validation
    if low_freq is None and high_freq is None:
        raise ValueError("low_freq and high_freq cannot both be None.")

    # Define filter type
    if low_freq is None and high_freq is not None:
        btype = "lowpass"
        Wn = high_freq
    elif low_freq is not None and high_freq is None:
        btype = "highpass"
        Wn = low_freq
    else:
        btype = "bandpass"
        Wn = [low_freq, high_freq]

    # Create the filter
    b, a = signal.butter(order, Wn=Wn, btype=btype, fs=sampling_frequency)

    # Apply the filter
    x_filt = signal.filtfilt(b, a, x, axis=0)

    return x_filt.astype(x.dtype)


def amplitude_envelope(x: np.ndarray) -> np.ndarray:
    """
    Calculates amplitude envelope.

    Parameters
    ----------
    x : np.ndarray
        Time series data. Shape must be (n_samples, n_channels).

    Returns
    -------
    x_env : np.ndarray
        Amplitude envelope data. Shape is (n_samples, n_channels).
    """
    # Apply Hilbert transform
    x_env = np.abs(signal.hilbert(x, axis=0))
    return x_env.astype(x.dtype)


def welch_psd(
    x: torch.Tensor,
    segment_length: int,
    overlap: float = 0.5,
    fs: float = 1.0,
    detrend: bool = True,
) -> torch.Tensor:
    """
    Welch PSD estimate over the last axis.

    Pure-PyTorch implementation (Hann window, density-scaled, one-sided rFFT)
    that matches ``scipy.signal.welch(..., window="hann", scaling="density",
    return_onesided=True, average="mean")`` to within floating-point
    tolerance. Lives here so both the online sampler and offline
    target-PSD computation share one definition of the frequency axis.

    Parameters
    ----------
    x : torch.Tensor
        Real-valued input. Shape is ``(*batch_dims, T)``.
        Welch is computed along the last axis.
    segment_length : int
        Length of each Welch segment (``nperseg``).
    overlap : float, optional
        Fractional overlap in ``[0, 1)`` between consecutive segments.
        Defaults to 0.5.
    fs : float, optional
        Sampling frequency. Only affects the absolute scale of the returned
        PSD (density: units of x^2/fs). Defaults to 1.0.
    detrend : bool, optional
        If True, subtract the mean from each segment before windowing
        (matches ``scipy.signal.welch(detrend="constant")``). Defaults to True.

    Returns
    -------
    psd : torch.Tensor
        One-sided PSD of shape ``(*batch_dims, F)`` where
        ``F = segment_length // 2 + 1``.
    """
    # Validation
    if segment_length <= 0:
        raise ValueError(f"segment_length must be > 0; got {segment_length}.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1); got {overlap}.")
    T = x.shape[-1]
    if T < segment_length:
        raise ValueError(
            f"Input length ({T}) is shorter than segment_length ({segment_length})."
        )

    n_overlap = int(segment_length * overlap)  # truncate to match scipy
    step = max(1, segment_length - n_overlap)
    n_segments = 1 + (T - segment_length) // step

    # Build segments
    segments = x.unfold(dimension=-1, size=segment_length, step=step)
    segments = segments[..., :n_segments, :]

    if detrend:
        segments = segments - segments.mean(dim=-1, keepdim=True)

    window = torch.hann_window(
        segment_length, periodic=True, dtype=x.dtype, device=x.device
    )  # matches scipy.signal.get_window("hann", N, fftbins=True)
    win_energy = (window * window).sum()  # scalar

    windowed = segments * window
    spectrum = torch.fft.rfft(windowed, n=segment_length, dim=-1)
    power = spectrum.real.pow(2) + spectrum.imag.pow(2)

    # Density scaling (one-sided, so double interior bins but not DC / Nyquist)
    F = segment_length // 2 + 1
    scale = 1.0 / (fs * win_energy)
    psd = power * scale
    if F > 1:
        # Interior bins
        if segment_length % 2 == 0:
            psd[..., 1:-1] = psd[..., 1:-1] * 2.0
        else:
            psd[..., 1:] = psd[..., 1:] * 2.0

    psd = psd.mean(dim=-2)  # average across segments
    return psd


def moving_average(x: np.ndarray, n_window: int) -> np.ndarray:
    """
    Calculates a moving average over a sliding window along the time axis.

    This function uses a cumulative-sum trick for efficiency and returns only
    resulting values where the full window fits.

    Parameters
    ----------
    x : np.ndarray
        Time series data. Shape must be (n_samples, n_channels).
    n_window : int
        Number of data points in the sliding window. Must be odd.

    Returns
    -------
    x_ma : np.ndarray
        Time series with sliding window applied.
        Shape is (n_samples - n_window + 1, n_channels).
    """
    if n_window % 2 == 0:
        raise ValueError("n_window must be odd.")

    # Calculate cumulative sum
    c = np.cumsum(x, axis=0)

    # Pad cumulative sum with leading zeros
    c = np.vstack([np.zeros((1, x.shape[1])), c])

    # Calculate moving average
    x_ma = (c[n_window:] - c[:-n_window]) / float(n_window)
    return x_ma.astype(x.dtype)
