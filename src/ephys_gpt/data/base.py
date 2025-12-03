


# Import packages
import numpy as np
import os
import torch
from pqdm.threads import pqdm
from torch.utils.data import Dataset
from typing import Union, Optional

import ephys_gpt.data.rw as rw


class EphysDataset(Dataset):
    def __init__(
        self,
        data: Union[list, str, np.ndarray],
        labels: Union[list, str, np.ndarray] = None,
        data_field: Optional[str] = "X",
        picks: Optional[Union[str, list[str]]] = None,
        reject_by_annotation: Optional[str] = None,
        sampling_frequency: Optional[float] = None,
        load_memmaps: Optional[bool] = None,
        store_dir: Optional[str] = "tmp",
        n_jobs: Optional[int] = 1,
    ):
        """
        A PyTorch Dataset class for electrophysiological data.
        """
        self._identifier = id(self)
        self.labels = labels
        self.data_field = data_field
        self.picks = picks
        self.reject_by_annotation = reject_by_annotation
        self.sampling_frequency = sampling_frequency
        self.load_memmaps = load_memmaps
        self.store_dir = store_dir
        self.n_jobs = n_jobs

        # Validate input data
        self.data = rw.validate_inputs(data)
        if len(self.data) == 0:
            raise ValueError("No valid inputs were passed.")
        
        # Load and validate the raw data
        self.raw_data_arrays, self.raw_data_filenames = self.load_raw_data()
        self.validate_raw_data()

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]
    
    def load_raw_data(self):
        """
        Imports raw data into memory or as memory maps.

        Returns
        -------
        raw_data : Union[list[np.ndarray], list[np.memmap]]
            List of raw data arrays or memory maps.
        raw_data_filenames : list[str]
            List of filenames corresponding to each raw data array.
        """
        raw_data_pattern = "raw_data_{{i:0{width}d}}_{identifier}.npy".format(
            width=len(str(len(self.data))), identifier=self._identifier
        )
        raw_data_filenames = [
            os.path.join(self.store_dir, raw_data_pattern.format(i=i))
            for i in range(len(self.data))
        ]
        # not used if self.data is a list of strings, where the strings are paths to .npy files

        # Function to save a single memory map
        def _load_data(raw_data, mmap_location):
            if not self.load_memmaps:  # do not load into the memory maps
                mmap_location = None
            loaded_data = rw.load_data(
                raw_data,
                self.data_field,
                self.picks,
                self.reject_by_annotation,
                mmap_location,
                mmap_mode="r",
            )
            return loaded_data

        # Load data
        raw_data = pqdm(
            array=zip(self.data, raw_data_filenames),
            function=_load_data,
            n_jobs=self.n_jobs,
            desc="Loading files",
            argument_type="args",
            total=len(self.data),
        )
        return raw_data, raw_data_filenames

    def validate_raw_data(self):
        """Validates raw data arrays."""
        n_channels = [array.shape[-1] for array in self.raw_data_arrays]
        if not all(np.equal(n_channels, n_channels[0])):
            raise ValueError("All inputs should have the same number of channels.")
