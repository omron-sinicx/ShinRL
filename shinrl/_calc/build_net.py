""" Functions to build networks. 
Author: Toshinori Kitamura
Affiliation: NAIST
"""
from __future__ import annotations

from typing import Any, Callable, Tuple, Type

import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from distrax import Categorical
from mypy_extensions import KwArg

import shinrl as srl


def build_forward_fc(
    n_out: int, depth: int, hidden: int, act_layer: Any
) -> hk.Transformed:
    """Build a forward step of fully-connected network.

    Args:
        n_out (int): Number of outputs.
        depth (int): Depth of layers.
        hidden (int): # of hidden units of fc.
        act_layer (Any): Activation layer.

    Returns:
        hk.Transformed:
            Takes [batch x ?] input and returns [batch x n_out] Array.
    """

    @jax.vmap
    def forward(input: Array) -> Array:
        modules = []
        if depth > 0:
            modules.append(hk.Linear(hidden))
            for _ in range(depth - 1):
                modules += [act_layer, hk.Linear(hidden)]
            modules += [act_layer, hk.Linear(n_out)]
        else:
            modules.append(hk.Linear(n_out))
        return hk.Sequential(modules)(input.astype(float))

    return hk.without_apply_rng(hk.transform(forward))


def build_forward_conv(
    n_out: int, depth: int, hidden: int, act_layer: Any
) -> hk.Transformed:
    """Build a forward step of simple cnn.

    Args:
        n_out (int): Number of outputs.
        depth (int): Depth of layers.
        hidden (int): # of hidden units of fc.
        act_layer (Any): Activation layer.

    Returns:
        hk.Transformed:
            Takes [batch x H x W x C] input and returns [batch x n_out] Array.
    """

    def forward(input: Array) -> Array:
        modules = [
            hk.Conv2D(16, kernel_shape=3, stride=1, padding="VALID"),
            hk.Flatten(),
        ]
        for _ in range(depth):
            modules += [act_layer, hk.Linear(hidden)]
        modules += [act_layer, hk.Linear(n_out)]
        return hk.Sequential(modules)(input.astype(float))

    return hk.without_apply_rng(hk.transform(forward))


NET_ACT_FN = Callable[
    [PRNGKey, Array, hk.Params, KwArg(Array)], Tuple[PRNGKey, Array, Array]
]


def build_net_act(Dist: Type[Categorical], net: hk.Transformed) -> NET_ACT_FN:
    """Build an act function with a network.

    Args:
        Dist (Categorical): Type of policy distribution.
        net (hk.Transformed): A network which takes an observation and returns logits.

    Returns:
        net_act (NET_ACT_FN)
    """

    @jax.jit
    def net_act(
        key: PRNGKey, obs: Array, params: hk.Params, **dist_kwargs: Array
    ) -> Tuple[PRNGKey, Array, Array]:
        new_key, key = jax.random.split(key)
        obs = jnp.expand_dims(obs, axis=0)  # (1, obs.shape)
        dist = Dist(net.apply(params, obs), **dist_kwargs)
        act = jnp.squeeze(dist.sample(seed=key), axis=0)  # (, )
        log_prob = jnp.squeeze(dist.log_prob(act), axis=0)  # (, )
        return new_key, act, log_prob

    return net_act
