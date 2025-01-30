"""Some functions taken from https://docs.kidger.site/equinox/examples/score_based_diffusion/"""

import functools as ft
import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax as dfx
import equinox as eqx
import optax
import json
from tqdm import tqdm


def single_loss_fn(model, ode, data, t, key):
    key, modelkey = jr.split(key)
    mean, var = ode.marginal_prob(data, t)
    std = jnp.sqrt(jnp.maximum(var, 1e-5))
    noise = jr.normal(key, data.shape)
    y = mean + std * noise
    pred = model(t, y, key=modelkey)
    return jnp.mean((pred * std + noise) ** 2)


def batch_loss_fn(model, ode, data, key):
    batch_size = data.shape[0]
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)
    # Low-discrepancy sampling over t to reduce variance
    t = jr.uniform(tkey, (batch_size,), minval=10e-4, maxval=1 / batch_size)
    t = t + (1 / batch_size) * jnp.arange(batch_size)
    loss_fn = ft.partial(single_loss_fn, model, ode)
    loss_fn = jax.vmap(loss_fn)
    return jnp.mean(loss_fn(data, t, losskey))


def dataloader(data, batch_size, *, key):
    dataset_size = data.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        key, subkey = jr.split(key, 2)
        perm = jr.permutation(subkey, indices)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield data[batch_perm]
            start = end
            end = start + batch_size


@eqx.filter_jit
def make_step(model, ode, data, key, opt_state, opt_update):
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model, ode, data, key)
    updates, opt_state = opt_update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return loss, model, key, opt_state


def ode_sampler(model, ode, data_shape, dt0, key):
    keys = jr.split(key, 2)

    def drift(t, y, args):
        return ode.drift(model, t, y, key=keys[0])

    term = dfx.ODETerm(drift)
    solver = dfx.Tsit5()
    t0 = 10e-3
    y1 = jr.normal(keys[1], data_shape)
    # reverse time, solve from t1 to t0
    sol = dfx.diffeqsolve(term, solver, 1, t0, -dt0, y1)
    return sol.ys[0]


def train_loop(
    model,
    ode,
    data,
    num_epochs,
    batch_size,
    print_every,
    test_data,
    lr,
    key,
):
    train_key, loader_key, test_key = jr.split(key, 3)
    num_steps = num_epochs * (data.shape[0] // batch_size)
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    total_value = 0
    total_size = 0
    with tqdm(total=num_steps, desc="Training") as pbar:
        for step, data in zip(
            range(num_steps), dataloader(data, batch_size, key=loader_key)
        ):
            value, model, train_key, opt_state = make_step(
                model, ode, data, train_key, opt_state, opt.update
            )
            total_value += value.item()
            total_size += 1

            if (step % print_every) == 0 or step == num_steps - 1:
                # Evaluate the model on the test data
                test_loss = batch_loss_fn(model, ode, test_data, test_key)
                pbar.set_postfix(
                    {
                        "Train Loss": total_value / total_size,
                        "Test Loss": test_loss.item(),
                    }
                )
                total_value = 0
                total_size = 0

            pbar.update(1)
    return model


def save(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load(filename, model_class, key):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = model_class(**hyperparams, key=key)
        return eqx.tree_deserialise_leaves(f, model)
