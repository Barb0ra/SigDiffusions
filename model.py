"""Some functions taken from https://docs.kidger.site/equinox/examples/bert/ and https://github.com/Y-debug-sys/Diffusion-TS"""

import math
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float, Int  # https://github.com/google/jaxtyping
from typing import List, Optional


def upsample_1d(y, factor=2):
    C, W = y.shape
    y = jnp.reshape(y, [C, W, 1])
    y = jnp.tile(y, [1, 1, factor])
    return jnp.reshape(y, [C, W * factor])


class LearnablePositionalEncoding(eqx.Module):
    pe: eqx.nn.Embedding

    def __init__(self, hidden_size, sig_length, key):
        self.pe = eqx.nn.Embedding(
            num_embeddings=sig_length, embedding_size=hidden_size, key=key
        )

    def __call__(
        self,
        x,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        position_ids = jnp.array(range(x.shape[0]))
        positions = jax.vmap(self.pe)(position_ids)
        return x + positions


class SinusoidalPosEmb(eqx.Module):
    dim: int

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim) * -emb)
        emb = t * emb
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb


class AdaLayerNorm(eqx.Module):
    emb: SinusoidalPosEmb
    silu: jax.nn.silu
    linear: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm

    def __init__(self, n_embd, key):
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = jax.nn.silu
        self.linear = eqx.nn.Linear(n_embd, n_embd * 2, key=key)
        self.layernorm = eqx.nn.LayerNorm(n_embd, elementwise_affine=False)

    def __call__(self, x, t):
        emb = self.emb(t)
        emb = self.linear(self.silu(emb))
        scale, shift = jnp.split(emb, 2, 0)
        x = jax.vmap(self.layernorm)(x) * (1 + scale) + shift
        return x


class AttentionBlock(eqx.Module):
    """A single transformer attention block."""

    attention: eqx.nn.MultiheadAttention
    layernorm: AdaLayerNorm
    num_heads: int = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        key: jax.random.PRNGKey,
    ):
        att_key, layernorm_key = jax.random.split(key)
        self.num_heads = num_heads
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            qk_size=8,
            vo_size=8,
            key=att_key,
        )
        self.layernorm = AdaLayerNorm(hidden_size, layernorm_key)

    def __call__(
        self,
        x: Float[Array, "seq_len hidden_size"],
        t,
        key: "jax.random.PRNGKey" = None,
    ) -> Float[Array, "seq_len hidden_size"]:

        attention_key, dropout_key = (
            (None, None) if key is None else jax.random.split(key)
        )

        att_input = self.layernorm(x, t)

        attention_output = self.attention(
            query=att_input,
            key_=att_input,
            value=att_input,
            mask=None,
            key=attention_key,
        )

        return x + attention_output


class FeedForwardBlock(eqx.Module):
    """A single transformer feed forward block."""

    mlp: eqx.nn.Linear
    output: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        key: jax.random.PRNGKey,
    ):
        mlp_key, output_key = jax.random.split(key)
        self.mlp = eqx.nn.Linear(
            in_features=hidden_size, out_features=intermediate_size, key=mlp_key
        )
        self.output = eqx.nn.Linear(
            in_features=intermediate_size, out_features=hidden_size, key=output_key
        )

        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)

    def __call__(
        self,
        x: Float[Array, " hidden_size"],
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, " hidden_size"]:
        # Feed-forward.
        inputs = self.layernorm(x)
        hidden = self.mlp(inputs)
        hidden = jax.nn.gelu(hidden)
        output = self.output(hidden)
        return x + output


class TransformerLayer(eqx.Module):
    """A single transformer layer."""

    attention_block: AttentionBlock
    ff_block: FeedForwardBlock

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        key: jax.random.PRNGKey,
    ):
        attention_key, ff_key = jax.random.split(key)

        self.attention_block = AttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            key=attention_key,
        )
        self.ff_block = FeedForwardBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            key=ff_key,
        )

    def __call__(
        self,
        inputs,
        t: Float,
        mask: Optional[Int[Array, " seq_len"]] = None,
        *,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        attn_key, ff_key = (None, None) if key is None else jax.random.split(key)
        attention_output = self.attention_block(inputs, t, key=attn_key)
        seq_len = inputs.shape[0]
        ff_keys = None if ff_key is None else jax.random.split(ff_key, num=seq_len)
        output = jax.vmap(self.ff_block, in_axes=(0, 0))(attention_output, ff_keys)
        return output


class Transformer(eqx.Module):
    """A transformer model."""

    layers: List[TransformerLayer]
    pos_enc: LearnablePositionalEncoding
    input_proj: eqx.nn.Conv1d
    output_proj: eqx.nn.Conv1d
    dim: int

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        sig_length: int,
        dim: int,
        by_channel: bool,
        key: jax.random.PRNGKey,
    ):
        # if we have taken signatures of each channel separately, stack them before the convolution layers
        if by_channel:
            self.dim = dim
        else:
            self.dim = 1

        layer_key, enc_key1, enc_key2, input_key, output_key = jax.random.split(
            key, num=5
        )

        self.pos_enc = LearnablePositionalEncoding(
            hidden_size, sig_length, key=enc_key1
        )

        self.input_proj = eqx.nn.Conv1d(
            self.dim, hidden_size, kernel_size=1, stride=1, padding=0, key=input_key
        )
        self.output_proj = eqx.nn.Conv1d(
            hidden_size, self.dim, kernel_size=1, padding=0, key=output_key
        )

        layer_keys = jax.random.split(layer_key, num=num_layers)
        self.layers = []
        for layer_key in layer_keys:
            self.layers.append(
                TransformerLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    key=layer_key,
                )
            )

    def __call__(self, t, x, key=None):
        shape = x.shape[0]
        x = x.reshape(self.dim, -1)

        x = self.input_proj(x).swapaxes(0, 1)
        x = self.pos_enc(x, key=key)

        layer_outputs = []
        for layer in self.layers:
            cl_key, key = (None, None) if key is None else jax.random.split(key)
            x = layer(x, t, None, key=cl_key)
            layer_outputs.append(x)

        x = x.swapaxes(0, 1)
        x = self.output_proj(x)
        x = x.reshape(shape)
        return x
