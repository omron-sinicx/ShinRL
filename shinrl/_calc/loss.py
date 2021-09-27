"""JAX functions to compute losses.
Author: Toshinori Kitamura
Affiliation: NAIST
"""

import chex
import jax
import optax
from chex import Array


@jax.jit
def l2_loss(pred: Array, target: Array) -> float:
    chex.assert_equal_shape((pred, target))
    loss = optax.l2_loss(pred, target)
    return loss.mean()


@jax.jit
def huber_loss(pred: Array, target: Array) -> float:
    chex.assert_equal_shape((pred, target))
    loss = optax.huber_loss(pred, target)
    return loss.mean()


@jax.jit
def cross_entropy_loss(logits: Array, targ_logits: Array) -> float:
    chex.assert_equal_shape((logits, targ_logits))
    targ_idx = targ_logits.argmax(axis=-1)[:, None]
    loss = optax.softmax_cross_entropy(logits, targ_idx)
    return loss.mean()


@jax.jit
def kl_loss(logits: Array, targ_logits: Array) -> float:
    chex.assert_equal_shape((logits, targ_logits))
    log_pol = jax.nn.log_softmax(logits, axis=1)
    pol = jax.nn.softmax(logits, axis=-1)
    loss = (pol * (log_pol - targ_logits)).sum(-1, keepdims=True)
    return loss.mean()
