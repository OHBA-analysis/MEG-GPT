"""Functions for reading and writing data."""

# Import packages
import logging
import mne
import numpy as np
import os
from scipy import ndimage
from scipy.io import loadmat
from typing import Union, Optional


_logger = logging.getLogger("ephys_gpt.data")
ALLOWED_EXT = [".npy", ".mat", ".txt", ".fif"]


def validate_inputs(inputs: Union[np.ndarray, str, list[str]]) -> list:
    """
    Validates input types and shapes.

    Parameters
    ----------
    inputs : Union[np.ndarray, str, list[str]]
        Inputs files or data.

    Returns
    -------
    validated_inputs : list
        Validated inputs.
    """
    # Case 1: String
    if isinstance(inputs, str):
        return (
            list_dir(inputs, keep_ext=ALLOWED_EXT)
            if os.path.isdir(inputs) else [inputs]
        )

    # Case 2: NumPy array
    if isinstance(inputs, np.ndarray):
        if inputs.ndim == 1:
            return [inputs[:, np.newaxis]]
        elif inputs.ndim == 2:
            return [inputs]
        return inputs
    
    # Case 3: List
    if isinstance(inputs, list):
        if not inputs:
            raise ValueError("Empty list passed.")
        # If the list contains only strings, treat them as paths/dirs
        if all(isinstance(x, str) for x in inputs):
            validated_inputs = []
            for x in inputs:
                if os.path.isdir(x):
                    validated_inputs.extend(list_dir(x, keep_ext=ALLOWED_EXT))
                elif os.path.exists(x):
                    validated_inputs.append(x)
                else:
                    _logger.warning(f"{x} not found")
            return validated_inputs
        # Otherwise, assume it's a list of arrays or other valid objects
        return inputs

    raise ValueError("inputs must be str, np.ndarray, or list.")


def file_ext(filename: str) -> Union[str, None]:
    """
    Returns the extension of a file.

    Parameters
    ----------
    filename : str
        Path to a file.
    """
    if not isinstance(filename, str):
        return None
    _, ext = os.path.splitext(filename)
    return ext


def list_dir(path: str, keep_ext: Optional[Union[str, list[str]]] = None) -> list[str]:
    """
    Lists files in a directory, optionally filtering by extension.

    Parameters
    ----------
    path : str
        Directory containing files to list.
    keep_ext : Optional[Union[str, list[str]]]
        Extensions of files to include in the returned list (including
        the leading dot). Default is None, which includes all files.

    Returns
    -------
    files : list[str]
        Full paths to matching files.
    """
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a directory")
    if isinstance(keep_ext, str):
        keep_ext = {keep_ext}
    elif keep_ext is not None:
        keep_ext = set(keep_ext)
    files = []
    for entry in sorted(os.scandir(path), key=lambda e: e.name):
        if not entry.is_file():
            continue
        ext = file_ext(entry.name)
        if keep_ext is None or ext in keep_ext:
            files.append(os.path.join(path, entry.name))
    return files


def load_data(
    data: Union[np.ndarray, str, list[str]],
    data_field: Optional[str] = "X",
    picks: Optional[Union[str, list[str]]] = None,
    reject_by_annotation: Optional[str] = None,
    mmap_location: Optional[str] = None,
    mmap_mode: Optional[str] = "c",
) -> Union[np.ndarray, np.memmap]:
    """
    Loads a time series data, after checking data shape and type.

    Data shape is expected to be (n_samples, n_channels).
    Data type is expected to be :code:`float32`.

    Parameters
    ----------
    data : Union[np.ndarray, str, list[str]]
        An array or path to a :code:`.npy`, :code:`.mat`, :code:`.txt` or
        :code:`.fif` file containing the data.
    data_field : Optional[str]
        Data field to load from a MATLAB file if a MATLAB filename is passed.
    picks : Optional[Union[list[str], str]]
        Channels to include. Passed to `mne.io.Raw.get_data
        <https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw\
        .get_data>`_ or `mne.Epochs.get_data <https://mne.tools/stable\
        /generated/mne.Epochs.html#mne.Epochs.get_data>`_.
        Only used if a .fif file is provided.
    reject_by_annotation : Optional[str]
        Whether to reject by annotation. Passed to `mne.io.Raw.get_data <\
        https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.get_data>`_.
        Only used if a .fif file is provided.
    mmap_location : Optional[str]
        If provided, the location to save a NumPy memory map of the data.
        Default is None.
    mmap_mode : Optional[str]
        Mode to load memory maps in. Default is :code:`'c'`.

    Returns
    -------
    data : Union[np.ndarray, np.memmap]
        Loaded data array.
    """
    # Define helper function
    def _save_for_mmap(array: np.ndarray, location: str) -> Union[str, None]:
        if not location:
            return None
        store_dir = os.path.dirname(location)
        if store_dir:
            os.makedirs(store_dir, exist_ok=True, mode=0o700)
        np.save(location, array)  # save to a file to load as memmap later
        return location

    # Case 1: NumPy array
    if isinstance(data, np.ndarray):
        data = data.astype(np.float32)
        mmap_loc = _save_for_mmap(data, mmap_location)
        if mmap_loc is None:
            return data
        data = mmap_loc

    # Case 2: String file path(s)
    if isinstance(data, str):
        # Check if file exists
        if not os.path.exists(data):
            raise FileNotFoundError(data)

        # Check extension
        ext = file_ext(data)
        if ext not in ALLOWED_EXT:
            raise ValueError(f"Data file must have an extension in {ALLOWED_EXT}.")

        # Load data based on extension
        if ext == ".npy":
            if mmap_location is None:
                return np.load(data).astype(np.float32)
            mmap_location = data  # use existing .npy as memmap source
        
        elif ext == ".mat":
            data = load_matlab(data, data_field).astype(np.float32)
            mmap_loc = _save_for_mmap(data, mmap_location)
            if mmap_loc is None:
                return data
            data = mmap_loc

        elif ext == ".txt":
            data = np.loadtxt(data).astype(np.float32)
            mmap_loc = _save_for_mmap(data, mmap_location)
            if mmap_loc is None:
                return data
            data = mmap_loc

        elif ext == ".fif":
            data = load_fif(data, picks, reject_by_annotation).astype(np.float32)
            mmap_loc = _save_for_mmap(data, mmap_location)
            if mmap_loc is None:
                return data
            data = mmap_loc

    # Load data as memory map from .npy file
    data_mm = np.load(mmap_loc, mmap_mode=mmap_mode)
    return data_mm.astype(np.float32)


def load_fif(
    filename: str,
    picks: Optional[Union[list[str], str]] = None,
    reject_by_annotation: Optional[str] = None,
) -> np.ndarray:
    """
    Loads a fif file.

    Parameters
    ----------
    filename : str
        Path to fif file. Must end with :code:`'raw.fif'` or :code:`'epo.fif'`.
    picks : Optional[Union[list[str], str]]
        Argument passed to `mne.io.Raw.get_data
        <https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw\
        .get_data>`_ or `mne.Epochs.get_data <https://mne.tools/stable\
        /generated/mne.Epochs.html#mne.Epochs.get_data>`_.
    reject_by_annotation : Optional[str]
        Argument passed to `mne.io.Raw.get_data <https://mne.tools/stable\
        /generated/mne.io.Raw.html#mne.io.Raw.get_data>`_.

    Returns
    -------
    data : np.ndarray
        Time series data array of shape (n_samples, n_channels).
        If an :code:`mne.Epochs` fif file is passed (:code:`'epo.fif'`), the epochs
        are concatenated along the first axis.
    """
    if "raw.fif" in filename:
        raw = mne.io.read_raw_fif(filename, verbose=False)
        data = raw.get_data(
            picks=picks,
            reject_by_annotation=reject_by_annotation,
            verbose=False,
        ).T
    elif "epo.fif" in filename:
        epochs = mne.read_epochs(filename, verbose=False)
        data = epochs.get_data(picks=picks)
        data = np.swapaxes(data, 1, 2).reshape(-1, data.shape[1])
    else:
        raise ValueError("a fif file must end with 'raw.fif' or 'epo.fif'.")
    return data


def save_fif(
    data: np.ndarray,
    filename: str,
    sampling_frequency: Optional[float] = None,
    bad_samples: Optional[np.ndarray] = None,
    original_fif: Optional[str] = None,
    verbose: Optional[bool] = True,
) -> None:
    """
    Saves a fif file.

    Parameters
    ----------
    data : np.ndarray
        Data to save. Shape must be (n_good_samples, n_channels).
    filename : str
        Output filename. Recommended to end with 'raw.fif'.
    sampling_frequency : Optional[float]
        Sampling frequency in Hz. If None, the sampling frequency
        is taken from :code:`original_fif`.
    bad_samples : Optional[np.ndarray]
        Boolean numpy array of shape (n_samples,) which indicates
        if a time point is good (False) or bad (True).
    original_fif : Optional[str]
        Path to the original fif file containing the data.
        We take the timing info from this file if passed.
    verbose : Optional[bool]
        If True, prints verbose output.
    """
    if sampling_frequency is None and original_fif is None:
        raise ValueError("Either sampling_frequency or original_fif must be passed.")

    if original_fif is not None:
        # Load original fif file and get sampling frequency
        original_raw = mne.io.read_raw_fif(original_fif, verbose=verbose)
        if sampling_frequency is not None:
            if sampling_frequency != original_raw.info["sfreq"]:
                raise ValueError(
                    "sampling_frequency does not match original_fif.info['sfreq']."
                )
        sampling_frequency = original_raw.info["sfreq"]

    # Create Info object
    n_channels = data.shape[1]
    info = mne.create_info(
        ch_names=[f"ch_{i}" for i in range(n_channels)],
        sfreq=sampling_frequency,
        ch_types=["misc"] * n_channels,
    )

    if bad_samples is not None:
        # Create the full time series with nans during bad segments
        x = np.full([bad_samples.shape[0], data.shape[1]], np.nan)
        x[~bad_samples] = data

        # Create the Raw object
        raw = mne.io.RawArray(x.T, info, verbose=verbose)

        if original_fif is not None:
            # Set timing info
            raw.set_meas_date(original_raw.info["meas_date"])
            raw.__dict__["_first_samps"] = original_raw.__dict__["_first_samps"]
            raw.__dict__["_last_samps"] = original_raw.__dict__["_last_samps"]
            raw.__dict__["_cropped_samp"] = original_raw.__dict__["_cropped_samp"]

        # Annotate bad segments
        _, times = raw.get_data(return_times=True)
        labels, n_bad_segments = ndimage.label(bad_samples)
        annotations = []
        for i in range(1, n_bad_segments + 1):
            indices = np.where(labels == i)[0]
            onset = times[indices[0]]
            duration = times[indices[-1]] - onset + raw.times[1] - raw.times[0]
            annotations.append(
                mne.Annotations(onset=onset, duration=duration, description="bad")
            )
        if annotations:
            raw.set_annotations(sum(annotations[1:], annotations[0]))

    else:
        # Create Raw object (all time points are good)
        raw = mne.io.RawArray(data.T, info, verbose=verbose)

    # Save .fif file
    raw.save(filename, overwrite=True, verbose=verbose)


def load_matlab(
    filename: str, field: Optional[str] = None, return_dict: bool = False
) -> Union[np.ndarray, dict]:
    """
    Loads a MATLAB file.

    Parameters
    ----------
    filename : str
        Filename of a MATLAB file to read.
    field : Optional[str]
        Field in the MATLAB file to read.
    return_dict : bool
        If True, returns selected fields as a dictionary.
        If False, returns the data as a numpy array, if only one field
        is present. Default is False.

    Returns
    -------
    data : Union[np.ndarray, dict]
        Data in the MATLAB file.
    """
    try:
        mat = loadmat(filename, simplify_cells=True)
    except NotImplementedError:
        raise NotImplementedError("MATLAB v7.3 files are not supported.")
    
    # Get fields without MATLAB metadata
    fields = {k: v for k, v in mat.items() if "__" not in k}

    if field is not None:
        if field not in fields:
            raise KeyError(f"field '{field}' missing from MATLAB file.")
        return fields[field]

    if return_dict:
        return fields
    
    if len(fields) == 1:
        return next(iter(fields.values()))
