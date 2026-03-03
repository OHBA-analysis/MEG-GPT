"""Utility helper functions for array operations."""

# Import packages
import numpy as np
from typing import Tuple


def get_one_hot(values: np.ndarray, n_states: int = None) -> np.ndarray:
    """
    Expands a categorical variable to a series of boolean columns (one-hot encoding).

    +----------------------+
    | Categorical Variable |
    +======================+
    |           A          |
    +----------------------+
    |           C          |
    +----------------------+
    |           D          |
    +----------------------+
    |           B          |
    +----------------------+

    becomes

    +---+---+---+---+
    | A | B | C | D |
    +===+===+===+===+
    | 1 | 0 | 0 | 0 |
    +---+---+---+---+
    | 0 | 0 | 1 | 0 |
    +---+---+---+---+
    | 0 | 0 | 0 | 1 |
    +---+---+---+---+
    | 0 | 1 | 0 | 0 |
    +---+---+---+---+

    Parameters
    ----------
    values : np.ndarray
        1D array of categorical values with shape (n_samples,). The values
        should be integers (0, 1, 2, 3, ... , `n_states` - 1). Or 2D
        array of shape (n_samples, n_states) to be binarized.
    n_states : int, optional
        Total number of states in `values`. Must be at least the number
        of states present in `values`. Default is the number of unique
        values in `values`.

    Returns
    -------
    one_hot : np.ndarray
        A 2D array containing the one-hot encoded form of `values`.
        Shape is (n_samples, n_states).
    """
    if values.ndim == 2:
        values = values.argmax(axis=1)
    if n_states is None:
        n_states = values.max() + 1
    res = np.eye(n_states)[np.array(values).reshape(-1)]
    return res.reshape([*list(values.shape), n_states]).astype(int)


def sliding_window_view(
    x: np.ndarray,
    window_shape: Tuple[int],
    axis: int = None,
    *,
    subok: bool = False,
    writeable: bool = False,
) -> np.ndarray:
    """
    Creates a sliding window over an array in arbitrary dimensions.

    Unceremoniously ripped from numpy 1.20,
    `np.lib.stride_tricks.sliding_window_view \
    <https://numpy.org/doc/1.20/reference/generated/\
    numpy.lib.stride_tricks.sliding_window_view.html>`_.
    """
    if np.iterable(window_shape):
        window_shape = tuple(window_shape)
    else:
        window_shape = (window_shape,)

    # First convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError("`window_shape` cannot contain negative values")

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(
                f"Since axis is `None`, must provide "
                f"window_shape for all dimensions of `x`; "
                f"got {len(window_shape)} window_shape elements "
                f"and `x.ndim` is {x.ndim}."
            )
    else:
        axis = np.core.numeric.normalize_axis_tuple(
            axis,
            x.ndim,
            allow_duplicate=True,
        )
        if len(window_shape) != len(axis):
            raise ValueError(
                f"Must provide matching length window_shape and "
                f"axis; got {len(window_shape)} window_shape "
                f"elements and {len(axis)} axes elements."
            )

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # Note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            msg = "window shape cannot be larger than input array shape"
            raise ValueError(msg)
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape

    return np.lib.stride_tricks.as_strided(
        x,
        strides=out_strides,
        shape=out_shape,
        subok=subok,
        writeable=writeable,
    )


def cov2corr(cov: np.ndarray) -> np.ndarray:
    """
    Covariance to correlation.

    Converts batches of covariance matrices into batches of correlation
    matrices.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrices. Shape must be (..., N, N).

    Returns
    -------
    corr : np.ndarray
        Correlation matrices. Shape is (..., N, N).
    """
    # Validation
    cov = np.array(cov)
    if cov.ndim < 2:
        raise ValueError("input covariances must have more than 1 dimension.")

    # Extract batches of standard deviations
    std = np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))
    normalisation = np.expand_dims(std, -1) @ np.expand_dims(std, -2)
    return cov / normalisation


def cov2partialcorr(cov: np.ndarray) -> np.ndarray:
    """
    Covariance to partial correlation.

    Converts batches of covariance matrices into batches of partial correlation
    matrices.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrices. Shape must be (..., N, N).

    Returns
    -------
    partial_corr : np.ndarray
        Partial correlation matrices. Shape is (..., N, N).
    """
    cov = np.array(cov)
    if cov.ndim < 2:
        raise ValueError("input covariances must have more than 1 dimension.")
    N = cov.shape[-1]
    precision = np.linalg.inv(cov)
    diag = np.diagonal(precision, axis1=-2, axis2=-1)
    outer_diag = diag[..., :, np.newaxis] * diag[..., np.newaxis, :]
    partial_corr = -precision / np.sqrt(outer_diag)
    partial_corr[..., range(N), range(N)] = 1.0 / diag
    return partial_corr


def cov2partialcov(cov: np.ndarray, use_pinv: bool = False) -> np.ndarray:
    """
    Covariance to partial covariance.

    Converts batches of covariance matrices into batches of partial covariance
    matrices.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix or batch of covariance matrices. Shape (..., N, N).
    use_pinv : bool, optional
        If True, use np.linalg.pinv to calculate inverse of the covariance.
        If False, we use np.linalg.inv.

    Returns
    -------
    partial_cov : np.ndarray
        Partial covariance matrices. Shape (..., N, N).
        Off-diagonals: -Omega_ij / (Omega_ii * Omega_jj - Omega_ij^2)
        Diagonals: 1 / Omega_ii  (conditional variances)
    """
    cov = np.asarray(cov)
    if cov.ndim < 2:
        raise ValueError(
            "input covariances must have at least 2 dimensions (..., N, N)."
        )
    if cov.shape[-1] != cov.shape[-2]:
        raise ValueError("last two dimensions must be square (N, N).")

    if use_pinv:
        precision = np.linalg.pinv(cov)
    else:
        precision = np.linalg.inv(cov)

    diag = np.diagonal(precision, axis1=-2, axis2=-1)  # (..., N)
    outer_diag = diag[..., :, np.newaxis] * diag[..., np.newaxis, :]
    denom = outer_diag - precision * precision  # (..., N, N)

    denom_safe = denom.copy()
    eps = np.finfo(cov.dtype if np.issubdtype(cov.dtype, np.floating) else float).eps
    small_mask = np.abs(denom_safe) <= eps
    denom_safe[small_mask] = 1.0

    partial = -precision / denom_safe

    diag_safe = diag.copy()
    tiny_diag_mask = np.abs(diag_safe) <= eps
    diag_safe[tiny_diag_mask] = np.nan  # will produce nan for truly degenerate cases
    diag_inv = 1.0 / diag_safe

    N = cov.shape[-1]
    idx = np.arange(N)
    partial[..., idx, idx] = diag_inv

    return partial
