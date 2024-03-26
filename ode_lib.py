"""Partially adapted from https://github.com/yang-song/score_sde/blob/main/sde_lib.py."""

import abc
import jax.numpy as jnp
import jax
import numpy as np


class VPODE:
    """Varience Preserving ODE."""

    def __init__(self, beta_min=0.1, beta_max=5):
        super().__init__()
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    @property
    def T(self):
        return 1

    def beta(self, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        return beta_t

    def drift(self, model, t, y, key):
        beta = self.beta(t)
        return -0.5 * beta * (y + model(t, y, key))

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = jnp.exp(log_mean_coeff) * x
        var = 1 - jnp.exp(2.0 * log_mean_coeff)
        return mean, var

    def beta_i(self, num_steps):
        """Function that discretizes beta for sampling."""
        return lambda i: self.beta_0 / num_steps + (i - 1) / (
            num_steps * (num_steps - 1)
        ) * (self.beta_1 - self.beta_0)
