import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from training_utils import train_loop, save
from ode_lib import VPODE
from model import Transformer


def preprocess_for_training(logsigs, test_set_size):
    # preprocess data
    train_data = logsigs[:-test_set_size]
    test_data = logsigs[-test_set_size:]

    train_data = jnp.array(train_data)
    test_data = jnp.array(test_data)

    data_mean = jnp.mean(train_data, axis=0)
    data_std = jnp.std(train_data, axis=0)
    train_data_standardized = (train_data - data_mean) / (data_std + 1e-6)
    test_data_standardized = (test_data - data_mean) / (data_std + 1e-6)

    return train_data_standardized, test_data_standardized, data_mean, data_std


def train(config, name, logsigs=None):
    """
    Trains a Transformer model using the provided configuration and data.

    Args:
        config (dict): Configuration dictionary described in the README.
        name (str): Experiment name.
        logsigs (np.ndarray, optional): Precomputed log-signatures. If None, they will be loaded from file.

    Returns:
        tuple: A tuple containing the best trained model, data mean, and data standard deviation.
    """
    test_set_size = config["dataset"]["test_set_size"]
    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]
    print_every = config["training"]["print_every"]
    lr = config["training"]["lr"]

    hidden_size = config["model"]["hidden_size"]
    hidden_size_multiplier = config["model"]["hidden_size_multiplier"]
    num_layers = config["model"]["num_layers"]
    num_heads = config["model"]["num_heads"]

    dim = config["dataset"]["dim"]
    by_channel = config["dataset"]["by_channel"]
    seed = config["seed"]
    model_checkpoints_folder = config["logging_folders"]["model_checkpoints"]
    real_sigs_folder = config["logging_folders"]["real_sigs"]

    if logsigs is None:
        logsigs = np.load(f"{real_sigs_folder}/{name}.npy").astype(np.float32)

    key = jr.PRNGKey(seed)
    key, model_key = jr.split(key)

    train_data, test_data, data_mean, data_std = preprocess_for_training(
        logsigs, test_set_size
    )

    hyperparams = {
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * hidden_size_multiplier,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "sig_length": train_data.shape[-1],
        "dim": dim,
        "by_channel": by_channel,
    }
    model = Transformer(
        **hyperparams,
        key=model_key,
    )

    # count parameters
    num_params = sum(
        np.prod(x.shape)
        for x in jax.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    )
    print(f"Number of parameters: {num_params}")

    ode = VPODE()
    model = train_loop(
        model, ode, train_data, num_epochs, batch_size, print_every, test_data, lr, key
    )

    save(f"{model_checkpoints_folder}/{name}", hyperparams, model)

    return model, data_mean, data_std
