"""Helpers for pre-computing initial weights for extra-label embeddings."""

# Import packages
from __future__ import annotations

import h5py
import logging
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meg_gpt.data.datasets import H5Dataset


_logger = logging.getLogger(__name__)


def per_class_token_freqs(
    dataset: "H5Dataset",
    n_tokens: int,
    label_name: str = "session_labels",
    smoothing: float = 0.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Computes a ``(n_classes, n_tokens)`` array of per-class normalised token
    frequency distributions.

    Each :class:`H5Session` contributes its bincount to the row indexed by
    ``session.label_idxs[label_name]``. Sessions sharing a class (e.g.
    multiple sessions per subject when ``label_name == "subject_labels"``)
    are summed before normalisation, so the result is always a valid
    distribution over tokens.

    Parameters
    ----------
    dataset : H5Dataset
        Dataset produced by :func:`meg_gpt.data.datasets.build_h5_dataset`
        with the appropriate ``label_cols`` configured.
    n_tokens : int
        Vocabulary size (including any sentinel token). Bincount columns
        beyond this index are dropped.
    label_name : str
        Label key to compute frequencies for. Must exist in
        ``dataset.label_vocabs``.
    smoothing : float
        Additive (Laplace) smoothing applied to the counts before
        normalisation. ``0.0`` disables smoothing.
    eps : float
        Numerical floor on the per-row denominator.

    Returns
    -------
    freqs : np.ndarray, shape (n_classes, n_tokens), dtype float32
        Row ``k`` is the normalised distribution for class index ``k``.
        Rows for classes with zero observed tokens fall back to a uniform
        distribution.
    """
    if label_name not in dataset.label_vocabs:
        raise KeyError(
            f"label_name {label_name!r} not in dataset.label_vocabs "
            f"({list(dataset.label_vocabs.keys())}). Did you pass label_cols "
            "to build_h5_dataset?"
        )

    vocab = dataset.label_vocabs[label_name]
    n_classes = len(vocab)
    counts = np.zeros((n_classes, n_tokens), dtype=np.float64)

    sessions = dataset.dataset.datasets
    for sess in sessions:
        idx = sess.label_idxs.get(label_name)
        if idx is None:
            raise RuntimeError(
                f"Session {sess.h5_path} has no label_idxs entry for "
                f"{label_name!r}."
            )
        with h5py.File(sess.h5_path, "r") as f:
            tokens = f[sess.data_key][...].ravel().astype(np.int64, copy=False)
        bc = np.bincount(tokens, minlength=n_tokens)
        counts[idx] += bc[:n_tokens]

    if smoothing > 0:
        counts += float(smoothing)

    denom = counts.sum(axis=1, keepdims=True)
    empty_rows = (denom <= eps).ravel()
    if empty_rows.any():
        _logger.warning(
            "per_class_token_freqs: %d/%d rows have zero observed tokens; "
            "filling with uniform distribution.",
            int(empty_rows.sum()), n_classes,
        )
        counts[empty_rows] = 1.0
        denom[empty_rows] = float(n_tokens)

    freqs = (counts / np.maximum(denom, eps)).astype(np.float32)
    return freqs
