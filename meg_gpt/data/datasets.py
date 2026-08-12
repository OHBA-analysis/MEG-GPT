"""PyTorch Datasets for MEG-GPT training."""

# Import packages
import os
import h5py
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any, Dict, List, Optional, Sequence
from torch.utils.data import ConcatDataset, Dataset


_logger = logging.getLogger(__name__)


class SimulationDataset(Dataset):
    """
    PyTorch dataset for the simulated synthetic dataset.

    Note: `exclude_subjects` and `include_subjects` arguments
          are mutually exclusive.

    Parameters
    ----------
    data_path : str
        Path to the directory containing the data files.
    window_len : int
        Length of the windows to extract (in samples).
    sampling_frequency : int
        The sampling frequency of the data (in Hz).
    info : List[str]
        Additional information to provide: subject | session | dataset
    standardize : bool
        Whether to standardize the data (over time).
    exclude_subjects : List[str], optional
        List of subjects to exclude.
    include_subjects : List[str], optional
        List of subjects to include.
    """
    def __init__(
        self,
        data_path: str,
        window_len: int,
        sampling_frequency: int,
        info: List[str],
        standardize: bool=True,
        exclude_subjects: Optional[List[int]] = None,
        include_subjects: Optional[List[int]] = None,
    ):
        # Set attributes
        self.data_path = Path(data_path)
        self.window_len = window_len
        self.info = info

        exclude_subjects = set(exclude_subjects or ())
        include_subjects = set(include_subjects or ())

        # Validate attributes
        if exclude_subjects and include_subjects:
            raise ValueError(
                "Only one of exclude_subjects or include_subjects may be provided."
            )

        # Get the list of subject IDs
        file_paths = sorted(self.data_path.glob("*.npy"))
        subject_paths = {int(f.stem.split("_")[1]): f for f in file_paths}

        # Include/exclude subjects
        subject_sets = set(subject_paths.keys())
        if exclude_subjects:
            subject_sets = subject_sets - exclude_subjects
        if include_subjects:
            subject_sets = subject_sets & include_subjects

        self.subjects = sorted(list(subject_sets))
        self.n_subjects = len(self.subjects)
        _logger.info("Number of subjects: %d", self.n_subjects)

        # Load simulated data
        datasets = []
        for n, subject in enumerate(tqdm(
            self.subjects, desc="Loading simulated data", total=self.n_subjects
        )):
            try:
                dataset = SimulationDataSubloader(
                    subject=subject,
                    subject_idx=n,
                    file_path=subject_paths[subject],
                    window_len=window_len,
                    sampling_frequency=sampling_frequency,
                    info=info,
                    standardize=standardize,
                )
                datasets.append(dataset)
            except Exception as exc:
                _logger.error(f"Failed to open a file for {subject}.", exc)

        self.dataset = ConcatDataset(datasets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)


class SimulationDataSubloader(Dataset):
    """
    Loads a recording of a specific subject from cache.

    Parameters
    ----------
    subject : int
        The subject ID.
    subject_idx : int
        The index of the subject. Unlike subject ID, the index is continuous
        for the embedding lookup table.
    file_path : Path
        The path to the cache file for the subject.
    window_len : float
        The length of the windows to extract (in samples).
    sampling_frequency : int
        The sampling frequency of the data (in Hz).
    info : List[str]
        Additional information to provide: subject | session | dataset
    standardize : bool
        Whether to standardize the data (over time).
    """
    def __init__(
        self,
        subject: int,
        subject_idx: int,
        file_path: Path,
        window_len: float,
        sampling_frequency: int,
        info: List[str],
        standardize: bool,
    ):
        # Set attributes
        self.subject = subject
        self.subject_idx = subject_idx
        self.window_len = window_len
        self.sampling_frequency = sampling_frequency
        self.file_path = file_path
        self.standardize = standardize

        # Write in additional information
        self.info = {}
        if "dataset" in info:
            self.info["dataset"] = "Simulation"
        if "subject" in info:
            self.info["subject"] = subject
            self.info["subject_idx"] = subject_idx
        if "session" in info:
            self.info["session"] = "None"

        # Load data
        output = self._read_cache_file()
        self.data = output[0]
        self.times = output[1]
        self.n_windows = self.data.shape[0] // self.window_len

        # Capture subject labels if they were returned
        if len(output) == 3:
            self.subject_labels = output[2]
        else:
            self.subject_labels = None

    def _standardize(self, array, eps=1e-8):
        """
        Standardizes the input array to have zero mean and unit variance.
        Assumes the input has shape (n_samples, n_channels).
        """
        mean = np.mean(array, axis=0, keepdims=True)
        std = np.std(array, axis=0, keepdims=True)
        return (array - mean) / (std + eps)

    def _read_cache_file(self):
        """
        Reads the cache file for the specified subject.
        """
        data = np.load(self.file_path, mmap_mode="c")
        times = np.arange(data.shape[0]) / self.sampling_frequency
        # data.shape: (n_samples, n_channels)
        # times.shape: (n_samples,)

        if self.standardize:
            data = self._standardize(data)

        assert data.shape[0] == times.shape[0], "Data and times must have the same length."

        if "subject" in self.info:
            subject_labels = np.full((data.shape[0],), self.subject_idx)  # shape: (n_samples,)
            return data, times, subject_labels

        return data, times

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.window_len
        end = start + self.window_len

        # Build the batch dictionary dynamically
        batch = {
            "data": self.data[start:end, :],
            "times": self.times[start:end],
            "info": self.info,
        }

        # Add subject labels if they exist
        if self.subject_labels is not None:
            batch["subject_labels"] = self.subject_labels[start:end]

        return batch


# ---------------------------------------------------------------------------
# Generic h5-backed Dataset
# ---------------------------------------------------------------------------
# Mirrors the interface consumed by :class:`MEGGPTDataModule`:
#   - Top-level object exposes ``.dataset`` as a ``ConcatDataset`` of
#     per-session sub-datasets.
#   - Each sub-dataset has a ``.subject`` attribute for subject-level splits.
#   - Each item is ``{"data": (L, C) float32, "times": (L,) float32,
#     "info": dict}``.
#
# Each h5 file is expected to hold a single ``"data"`` dataset of shape
# ``(n_samples, n_channels)``.


class H5Session(Dataset):
    """
    Non-overlapping windowed view of one h5 session file.

    Parameters
    ----------
    h5_path : str
        Path to a session h5 file containing a single dataset (named by
        ``data_key``) of shape ``(n_samples, n_channels)``.
    window_len : int
        Window length in samples; windows are non-overlapping. Any trailing
        samples shorter than ``window_len`` are dropped.
    sfreq : float
        Sample rate in Hz, used only to populate the ``times`` array.
    info : dict
        Metadata dict copied into every item's ``info`` field. Must contain a
        ``"subject"`` key if subject-level splitting will be used downstream.
    standardize : bool
        If True, apply per-channel, per-session z-score standardization.
        Leave False for integer token data.
    data_key : str
        Name of the h5 dataset to read. Defaults to ``"tokens"``.
    label_idxs : Optional[Dict[str, int]]
        Optional mapping ``{label_name: integer_index}``. For each entry,
        every window emits a constant ``(window_len,)`` int64 array under
        ``label_name`` (e.g. ``"session_labels": 12``). Used to feed the
        model's ``extra_label_specs`` (e.g. session / subject embeddings).
    """
    def __init__(
        self,
        h5_path: str,
        window_len: int,
        sfreq: float,
        info: Dict[str, Any],
        standardize: bool = True,
        data_key: str = "tokens",
        label_idxs: Optional[Dict[str, int]] = None,
    ):
        self.h5_path = str(h5_path)
        self.window_len = int(window_len)
        self.sfreq = float(sfreq)
        self.info = dict(info)
        self.standardize = bool(standardize)
        self.data_key = str(data_key)
        self.label_idxs: Dict[str, int] = (
            {str(k): int(v) for k, v in label_idxs.items()} if label_idxs else {}
        )

        # Required by MEGGPTDataModule subject-splitting logic
        self.subject = self.info.get("subject")

        with h5py.File(self.h5_path, "r") as f:
            ds = f[self.data_key]
            self.n_samples = int(ds.shape[0])
            self.n_channels = int(ds.shape[1])
            if self.standardize:
                data = ds[...].astype(np.float64)
                self._mean = data.mean(axis=0).astype(np.float32)
                self._std = data.std(axis=0).astype(np.float32)
            else:
                self._mean = None
                self._std = None

        self.n_windows = self.n_samples // self.window_len
        # Opened lazily per process; re-opened if the PID changes so a handle
        # opened in the main process isn't reused (and silently desynced) by a
        # forked DataLoader worker
        self._h5: Optional[h5py.File] = None
        self._h5_pid: Optional[int] = None

    def __len__(self) -> int:
        return self.n_windows

    def __getstate__(self) -> Dict[str, Any]:
        # Strip the handle so the dataset pickles cleanly into workers
        # (spawn multiprocessing context)
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_h5_pid"] = None
        return state

    def _handle(self) -> h5py.File:
        pid = os.getpid()
        if self._h5 is None or self._h5_pid != pid:
            self._h5 = h5py.File(self.h5_path, "r")
            self._h5_pid = pid
        return self._h5

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.n_windows:
            raise IndexError(idx)
        start = idx * self.window_len
        end = start + self.window_len
        data = self._handle()[self.data_key][start:end, :].astype(np.float32, copy=False)
        if self.standardize:
            data = (data - self._mean) / self._std
        times = np.arange(start, end, dtype=np.float32) / self.sfreq
        batch: Dict[str, Any] = {"data": data, "times": times, "info": self.info}
        for name, label_idx in self.label_idxs.items():
            batch[name] = np.full((self.window_len,), label_idx, dtype=np.int64)
        return batch


class H5Dataset(Dataset):
    """
    Concatenates :class:`H5Session` datasets.

    Here, ``.dataset`` is exposed as a ``ConcatDataset`` so it can be
    used with :class:`MEGGPTDataModule`.

    Parameters
    ----------
    sessions : Sequence[H5Session]
        Per-session datasets to concatenate.
    label_vocabs : Optional[Dict[str, Dict[Any, int]]]
        Optional mapping ``{label_name: {csv_value: integer_index}}`` kept for
        inspection (e.g. logging ``n_classes`` or persisting the vocabulary
        alongside a trained model). Computed by :func:`build_h5_dataset`.
    """
    def __init__(
        self,
        sessions: Sequence[H5Session],
        label_vocabs: Optional[Dict[str, Dict[Any, int]]] = None,
    ):
        if len(sessions) == 0:
            raise ValueError("H5Dataset requires at least one session.")
        self.dataset = ConcatDataset(list(sessions))
        self.label_vocabs: Dict[str, Dict[Any, int]] = dict(label_vocabs or {})

    @property
    def label_n_classes(self) -> Dict[str, int]:
        """Number of unique classes per label, derived from ``label_vocabs``."""
        return {name: len(vocab) for name, vocab in self.label_vocabs.items()}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[idx]


def build_h5_dataset(
    sessions_csv: str,
    h5_dir: str,
    window_len: int,
    sfreq: float = 250.0,
    standardize: bool = True,
    include_sessions: Optional[Sequence[str]] = None,
    info_cols: Sequence[str] = (
        "session",
        "dataset",
        "subject",
        "task",
        "system",
        "age",
        "sex",
    ),
    data_key: str = "tokens",
    label_cols: Optional[Sequence[str]] = None,
) -> H5Dataset:
    """
    Builds an :class:`H5Dataset` from a sessions CSV.

    Parameters
    ----------
    sessions_csv : str
        Path to a CSV indexing h5 sessions. Must contain a ``session`` column
        and columns named in ``info_cols``.
    h5_dir : str
        Directory containing ``{session}.h5`` files.
    window_len : int
        Number of samples per window.
    sfreq : float
        Sampling frequency in Hz. Defaults to 250 Hz.
    standardize : bool
        Apply per-session, per-channel z-score standardization.
    include_sessions : Optional[Sequence[str]]
        If given, restrict to these session IDs.
    info_cols : Sequence[str]
        Columns copied from the CSV into each item's ``info`` dict.
    data_key : str
        Name of the h5 dataset inside each session file.
        Defaults to ``"tokens"``.
    label_cols : Optional[Sequence[str]]
        CSV columns to expose as integer label streams. For each column
        ``col`` listed, every window emits ``{col}_labels`` (shape
        ``(window_len,)``, int64) carrying the session's continuous index
        for that column. The label vocabularies are stored on the returned
        :class:`H5Dataset` as ``label_vocabs``. Use this together with the
        model's ``extra_label_specs`` (e.g. ``name: session_labels``,
        ``n_classes: <number of unique sessions>``).
    """
    df = pd.read_csv(sessions_csv)
    if include_sessions is not None:
        df = df[df["session"].isin(set(include_sessions))]
    if df.empty:
        raise ValueError("No sessions selected from CSV.")

    # Build label vocabularies (deterministic via sorted unique values)
    label_cols = tuple(label_cols or ())
    label_vocabs: Dict[str, Dict[Any, int]] = {}
    for col in label_cols:
        if col not in df.columns:
            raise KeyError(f"label_cols entry {col!r} is not a column in {sessions_csv}.")
        unique_vals = sorted(df[col].dropna().unique().tolist())
        label_vocabs[f"{col}_labels"] = {val: i for i, val in enumerate(unique_vals)}

    h5_dir_path = Path(h5_dir)
    sessions: List[H5Session] = []
    for _, row in df.iterrows():
        info = {c: row[c] for c in info_cols if c in row}
        h5_path = h5_dir_path / f"{row['session']}.h5"
        label_idxs = {
            f"{col}_labels": label_vocabs[f"{col}_labels"][row[col]]
            for col in label_cols
        }
        sessions.append(
            H5Session(
                h5_path=str(h5_path),
                window_len=window_len,
                sfreq=sfreq,
                info=info,
                standardize=standardize,
                data_key=data_key,
                label_idxs=label_idxs,
            )
        )
    return H5Dataset(sessions, label_vocabs=label_vocabs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Set data directory
    DATA_DIR = Path("../../examples/simulations/data")

    # Create the simulation dataset
    sim_data = SimulationDataset(
        data_path=DATA_DIR,
        window_len=81,
        sampling_frequency=250,
        info=["subject", "dataset"],
        include_subjects=[0, 1, 2],
    )

    # Get a data sample (i.e., one window segment)
    sample = sim_data[0]
    data, times = sample["data"], sample["times"]
    subject_labels = sample["subject_labels"]

    print(f"Shape of data: {data.shape}")
    print(f"Shape of times: {times.shape}")
    print(f"Shape of subject labels: {subject_labels.shape}")

    # Plot a data sample
    fig, ax = plt.subplots(figsize=(12, 6))
    for ch in range(data.shape[1]):
        ax.plot(times, data[:, ch], label=f"Channel {ch + 1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.legend()
    plt.savefig("sim_data.png")
