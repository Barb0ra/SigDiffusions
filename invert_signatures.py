import numpy as np
import iisignature
from tqdm import tqdm
from signature_inversion_utils.fourier_inversion import (
    get_fourier_coeffs_from_sig,
    reconstruct_from_fourier_coeffs,
)
from data_loading_utils import reverse_minmax_scaler


def fourier_inversion(sigs, seq_len, dim, sig_depth, t):
    """
    Perform Fourier inversion on a list of signatures to reconstruct paths.

    Args:
        sigs (list): List of signatures to be inverted.
        seq_len (int): Length of the sequence to be reconstructed.
        dim (int): Dimensionality of the original path.
        sig_depth (int): Depth of the signature.
        t (array-like): Time points at which to evaluate the reconstructed path.

    Returns:
        np.ndarray: Array of reconstructed paths.
    """
    paths = []
    for i in tqdm(range(len(sigs)), desc="Inverting signatures"):
        a_n, b_n = get_fourier_coeffs_from_sig(sigs[i], dim, sig_depth, sig_depth - 2)
        inv = reconstruct_from_fourier_coeffs(t, a_n, b_n)
        # remove basepoint and mirror augmentation
        inv = inv[1 : seq_len + 1, :]
        paths.append(inv)
    return np.array(paths)


def reverse_scaler(real_path_folder, name, generated_paths, scaler):
    if scaler == "minmax":
        paths = reverse_minmax_scaler(real_path_folder, name, generated_paths)
    else:
        paths = generated_paths
    return paths


def invert_signatures(config, name, logsigs=None):
    """
    Inverts log-signatures to paths using Fourier inversion and saves the resulting paths.

    Parameters:
    config (dict): Configuration dictionary described in the README.

    name (str): Experiment name.
    logsigs (numpy.ndarray, optional): Array of log-signatures. If None, it will be loaded from file.

    Returns:
    numpy.ndarray: Array of inverted paths.
    """
    real_path_folder = config["logging_folders"]["real_paths"]
    generated_sig_folder = config["logging_folders"]["generated_sigs"]
    generated_path_folder = config["logging_folders"]["generated_paths"]
    seq_len = config["dataset"]["seq_len"]
    dim = config["dataset"]["dim"]
    scaler = config["dataset"]["scaler"]
    sig_depth = config["dataset"]["sig_depth"]
    by_channel = config["dataset"]["by_channel"]
    mirror_augmentation = config["dataset"]["mirror_augmentation"]

    if logsigs is None:
        logsigs = np.load(f"{generated_sig_folder}/{name}.npy")
    no = len(logsigs)

    if by_channel:
        # each channel is treated individually
        logsigs = logsigs.reshape((no * dim, -1))
        prev_dim = dim
        dim = 1

    # convert logsignatures to signatures
    # dim+3 is due to the augmentation
    s = iisignature.prepare(dim + 3, sig_depth, "COSX2")
    sigs = iisignature.logsigtosig(logsigs, s)

    time_range = seq_len + 1
    if mirror_augmentation:
        time_range *= 2
    t = np.linspace(0, 2 * np.pi, time_range)

    paths = fourier_inversion(sigs, seq_len, dim, sig_depth, t)

    if by_channel:
        paths = paths.reshape((no, prev_dim, seq_len)).swapaxes(1, 2)

    paths = reverse_scaler(real_path_folder, name, paths, scaler)

    np.save(f"{generated_path_folder}/{name}.npy", paths)

    return paths
