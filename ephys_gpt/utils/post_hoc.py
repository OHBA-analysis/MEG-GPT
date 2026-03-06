"""Utility helper functions for post-hoc analysis."""

# Import packages
import logging
import numpy as np
from pqdm.threads import pqdm
from sklearn.covariance import LedoitWolf
from tqdm.auto import trange
from ephys_gpt.utils import array_ops


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
