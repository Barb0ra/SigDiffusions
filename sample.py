import numpy as np
import functools as ft
import jax
import jax.numpy as jnp
import jax.random as jr

from training_utils import ode_sampler, load
from ode_lib import VPODE
from model import Transformer
from tqdm import tqdm


def sample(config, name, model=None, data_mean=None, data_std=None, data_shape=None):
    """
    Generates samples using a specified model and configuration.

    Parameters:
    config (dict): Configuration dictionary described in the README.
    name (str): Name of the experiment.
    model (optional): Pre-trained model to use for sampling. If None, the model will be loaded from the checkpoint.
    data_mean (optional): Mean of the real data signatures. If None, it will be calculated from the real signatures.
    data_std (optional): Standard deviation of the real data signatures. If None, it will be calculated from the real signatures.
    data_shape (optional): Shape of the real data signatures. If None, it will be inferred from the real signatures.

    Returns:
    jnp.ndarray: Generated samples.
    """
    model_checkpoints_folder = config["logging_folders"]["model_checkpoints"]
    real_sig_folder = config["logging_folders"]["real_sigs"]
    generated_sig_folder = config["logging_folders"]["generated_sigs"]

    num_steps = config["sampling"]["num_steps"]
    sample_size = config["sampling"]["sample_size"]
    sample_batch_size = config["sampling"]["sample_batch_size"]
    seed = config["seed"]

    key = jr.PRNGKey(seed)
    key, model_key = jr.split(key)
    key, sample_key = jr.split(key)
    sample_key_split = jr.split(sample_key, sample_size)

    if model is None:
        model = load(f"{model_checkpoints_folder}/{name}", Transformer, key=model_key)
        # load real signatures and calculate data attributes
        logsigs = np.load(f"{real_sig_folder}/{name}.npy").astype(np.float32)
        data_mean = jnp.mean(logsigs, axis=0)
        data_std = jnp.std(logsigs, axis=0)
        data_shape = logsigs.shape[1:]

    ode = VPODE()
    sampler = ode_sampler
    sample_fn = ft.partial(sampler, model, ode, data_shape, (1 - 1e-3) / num_steps)
    samples = []
    for batch in tqdm(range(sample_size // sample_batch_size), desc="Sampling"):
        sample_normalized = jax.vmap(sample_fn)(
            sample_key_split[
                batch * sample_batch_size : (batch + 1) * sample_batch_size
            ]
        )
        sample = data_mean + (data_std) * sample_normalized
        samples.append(sample)

    samples = jnp.concatenate(samples, axis=0)

    np.save(f"{generated_sig_folder}/{name}.npy", samples)

    return samples
