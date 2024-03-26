import numpy as np
import torch
from torchdiffeq import odeint
from torch import nn


def sine_data_generation(no, seq_len, dim):
    """Inspired by https://github.com/jsyoon0823/TimeGAN/blob/master/data_loading.py"""
    data = list()
    t = np.linspace(0, 24, seq_len)
    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = np.sin(freq * t + phase)
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)

    return data


def generate_predator_prey(no=10_000):
    """Method from https://github.com/morganstanley/MSML/blob/main/papers/Stochastic_Process_Diffusion/tsdiff/data/generate.py."""

    class PredatorPrey(nn.Module):
        def forward(self, t, y):
            y1, y2 = y.chunk(2, dim=-1)
            dy = torch.cat(
                [
                    2 / 3 * y1 - 2 / 3 * y1 * y2,
                    y1 * y2 - y2,
                ],
                -1,
            )
            return dy

    f = PredatorPrey()
    t = torch.linspace(0, 10, 1000)
    x0 = torch.rand(no, 2)
    with torch.no_grad():
        x = odeint(f, x0, t, method="dopri5").transpose(0, 1)

    return x.numpy()


def load_sines_data(data_path, seq_len, dim):
    no = 10000
    data = np.array(sine_data_generation(no, seq_len, dim))
    return data


def load_predator_prey_data(data_path, seq_len, dim):
    no = 10000
    data = generate_predator_prey(no)
    return data


def load_HEPC_data(data_path, seq_len, dim):
    data = np.load(data_path)
    return chop_into_windows(data, seq_len, stride=200)


def load_exchange_rates_data(data_path, seq_len, dim):
    data = np.load(data_path)
    return chop_into_windows(data, seq_len, stride=1)


def load_weather_data(data_path, seq_len, dim):
    data = np.load(data_path)
    return chop_into_windows(data[:, :, 0], seq_len, stride=5)


def minmax_scale_features(data):
    """
    Scales the features of the data to the range [0, 1] using min-max scaling.

    Args:
        data (np.ndarray): The data to be scaled.

    Returns:
        tuple: A tuple containing the scaled data, the minimum values, and the maximum values for each feature.
    """
    data_min = data.min(axis=(0, 1), keepdims=True)
    data_max = data.max(axis=(0, 1), keepdims=True)
    data = (data - data_min) / (data_max - data_min + 1e-6)
    return data, data_min, data_max


def reverse_minmax_scaler(real_path_folder, name, data):
    """
    Reverses the min-max scaling of the data using the scaling factors from the real data.

    Args:
        real_path_folder (str): The folder containing the real data.
        name (str): Experiment name.
        data (np.ndarray): The data to be rescaled.

    Returns:
        np.ndarray: The rescaled data.
    """
    real_paths = np.load(f"{real_path_folder}/{name}.npy")
    real_paths_scaled, data_min, data_max = minmax_scale_features(real_paths)
    return data * (data_max - data_min + 1e-6) + data_min


def clip_quantiles(data, q1, q2):
    data = np.clip(data, np.quantile(data, q1, axis=0), np.quantile(data, q2, axis=0))
    return data


def clip_outliers(data, n_std=4):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return np.clip(data, mean - n_std * std, mean + n_std * std)


def chop_into_windows(ts, window_size, stride):
    """
    Chops a time series into overlapping windows.

    Parameters:
    ts (numpy.ndarray): A numpy array of shape (length, dim) representing the time series.
    window_size (int): The size of each window.
    stride (int): The number of steps to move the window forward.

    Returns:
    numpy.ndarray: A numpy array of shape (num_windows, window_size, dim) containing the windows.
    """
    windows = []
    for i in range(0, len(ts) - window_size, stride):
        windows.append(np.array(ts[i : i + window_size, :]))
    windows = np.array(windows)
    return windows
