import numpy as np
import iisignature
import importlib
from data_loading_utils import (
    minmax_scale_features,
    clip_quantiles,
    clip_outliers,
)


def get_function_from_string(function_path):
    """
    Dynamically import a function from a given module path.

    Args:
        function_path (str): The full path to the function, e.g., 'module.submodule.function'.

    Returns:
        function: The imported function.
    """
    module_name, function_name = function_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def load_and_save_data(
    name, data_path, preprocessing_fn, seq_len, dim, shuffle, real_path_folder
):
    """
    Load data, apply preprocessing, optionally shuffle, and save to a file.
    """
    data = preprocessing_fn(data_path, seq_len, dim)
    no = len(data)

    if shuffle:
        idx = np.random.permutation(no)
        data = data[idx]

    np.save(f"{real_path_folder}/{name}.npy", data)
    return data


def augment_for_fourier_inversion(data, no, seq_len, dim, mirror_augmentation):
    """
    Augment the paths with time, sin(time), cos(time) - 1 for Fourier inversion.
    """
    time_range = seq_len + 1
    if mirror_augmentation:
        time_range *= 2
    time = np.linspace(0, 2 * np.pi, time_range)
    time = np.tile(time, (no, 1, 1)).reshape(no, time_range, 1)
    sin_time = np.sin(time)
    cos_time_minus_1 = np.cos(time) - 1
    data = np.concatenate((np.zeros((no, 1, dim)), data), axis=1)
    if mirror_augmentation:
        data = np.concatenate((data, data[:, ::-1, :]), axis=1)
    data = np.concatenate((time, sin_time, cos_time_minus_1, data), axis=2)
    return data


def take_log_signatures(
    data, sig_depth, by_channel, mirror_augmentation, no, seq_len, dim
):
    """
    Compute log-signatures of the data.

    Args:
        data (np.ndarray): The input time series.
        sig_depth (int): The depth of the signature.
        by_channel (bool): Whether to take a signature of each channel individually.
        mirror_augmentation (bool): Whether to augment with the reverse path.
        no (int): Number of samples.
        seq_len (int): Sequence length.
        dim (int): Number of time series channels.

    Returns:
        np.ndarray: The log-signatures of the data.
    """
    if by_channel:
        prev_no = no
        prev_dim = dim
        no = no * dim
        dim = 1
        data = data.swapaxes(1, 2).reshape(no, seq_len, 1)

    data = augment_for_fourier_inversion(data, no, seq_len, dim, mirror_augmentation)

    s = iisignature.prepare(dim + 3, sig_depth)
    logsigs = iisignature.logsig(data, s)

    logsigs = clip_quantiles(logsigs, 0.001, 0.999)
    logsigs = clip_outliers(logsigs, n_std=5)

    if by_channel:
        logsigs = logsigs.reshape(prev_no, -1)
        no = prev_no
        dim = prev_dim
    return logsigs


def compute_signatures(config, name):
    """
    Compute and save log-signatures for a given dataset configuration.

    Args:
        config (dict): Configuration dictionary containing dataset and logging details.
        name (str): Experiment name.

    Returns:
        np.ndarray: The computed log-signatures.
    """
    data_path = config["dataset"]["data_path"]
    preprocessing_fn_path = config["dataset"]["preprocessing_fn"]
    preprocessing_fn = get_function_from_string(preprocessing_fn_path)
    seq_len = config["dataset"]["seq_len"]
    dim = config["dataset"]["dim"]
    scaler = config["dataset"]["scaler"]
    shuffle = config["dataset"]["shuffle"]
    sig_depth = config["dataset"]["sig_depth"]
    by_channel = config["dataset"]["by_channel"]
    mirror_augmentation = config["dataset"]["mirror_augmentation"]

    real_path_folder = config["logging_folders"]["real_paths"]
    real_sig_folder = config["logging_folders"]["real_sigs"]
    seed = config["seed"]

    np.random.seed(seed)

    data = load_and_save_data(
        name, data_path, preprocessing_fn, seq_len, dim, shuffle, real_path_folder
    )
    if scaler == "minmax":
        data, data_min, data_max = minmax_scale_features(data)

    print(f"Computing log-signatures for {name}...")
    logsigs = take_log_signatures(
        data, sig_depth, by_channel, mirror_augmentation, len(data), seq_len, dim
    )
    np.save(f"{real_sig_folder}/{name}.npy", logsigs)
    return logsigs
