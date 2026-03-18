"""PyTorch Datasets for EphysGPT training."""

# Import packages
import logging
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Optional
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

        # Capture subject lables if they were returned
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
