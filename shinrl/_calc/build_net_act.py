"""Functions to build net_act functions.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from __future__ import annotations

from typing import Callable, Tuple

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array, Numeric, PRNGKey
from mypy_extensions import VarArg

import shinrl as srl

NETACT = Callable[
    [PRNGKey, Array, hk.Params, VarArg(Numeric)], Tuple[PRNGKey, Numeric, Numeric]
]


# ----- Discrete action -----


def build_discrete_greedy_net_act(net: hk.Transformed) -> NETACT:
    """Build a net_act function which acts greedily.

    Args:
        net(hk.Transformed):
            Takes [1 x obs] and returns [1 x dA] Array.

    Returns:
        NETACT
    """

    def net_act(
        key: PRNGKey, obs: Array, params: hk.Params
    ) -> Tuple[PRNGKey, Array, Array]:
        new_key, key = jax.random.split(key)
        obs = jnp.expand_dims(obs, axis=0)  # (1, obs.shape)
        dist = distrax.Greedy(net.apply(params, obs))
        act = jnp.squeeze(dist.sample(seed=key), axis=0)  # (, )
        log_prob = jnp.squeeze(dist.log_prob(act), axis=0)  # (, )
        return new_key, act, log_prob

    return jax.jit(net_act)


def build_eps_greedy_net_act(net: hk.Transformed) -> NETACT:
    """Build a net_act function with epsilon-greedy policy.

    Args:
        net(hk.Transformed):
            Takes [1 x obs] and returns [1 x dA] Array.

    Returns:
        NETACT
    """

    def net_act(
        key: PRNGKey,
        obs: Array,
        params: hk.Params,
        n_step: int,
        eps_decay: int,
        eps_warmup: int,
        eps_end: float,
    ) -> Tuple[PRNGKey, Array, Array]:
        new_key, key = jax.random.split(key)
        obs = jnp.expand_dims(obs, axis=0)  # (1, obs.shape)
        eps = srl.calc_eps(n_step, eps_decay, eps_warmup, eps_end)
        dist = distrax.EpsilonGreedy(net.apply(params, obs), epsilon=eps)
        act = jnp.squeeze(dist.sample(seed=key), axis=0)  # (, )
        log_prob = jnp.squeeze(dist.log_prob(act), axis=0)  # (, )
        return new_key, act, log_prob

    return jax.jit(net_act)


def build_softmax_net_act(net: hk.Transformed) -> NETACT:
    """Build a net_act function with softmax policy.

    Args:
        net(hk.Transformed):
            Takes [1 x obs] and returns [1 x dA] Array.

    Returns:
        NETACT
    """

    def net_act(
        key: PRNGKey, obs: Array, params: hk.Params, softmax_tmp: float
    ) -> Tuple[PRNGKey, Array, Array]:
        new_key, key = jax.random.split(key)
        obs = jnp.expand_dims(obs, axis=0)  # (1, obs.shape)
        dist = distrax.Softmax(net.apply(params, obs), softmax_tmp)
        act = jnp.squeeze(dist.sample(seed=key), axis=0)  # (, )
        log_prob = jnp.squeeze(dist.log_prob(act), axis=0)  # (, )
        return new_key, act, log_prob

    return jax.jit(net_act)


# ----- Continuous action -----


def build_normal_diagonal_net_act(net: hk.Transformed) -> NETACT:
    """Build a net_act function of a normal distribution with
    diagonal covariance matrix. Return its mean values if scale == 0.0.

    Args:
        net(hk.Transformed):
            Takes [1 x obs] and returns [1 x dA] Array.

    Returns:
        NETACT
    """

    def net_act(
        key: PRNGKey, obs: Array, params: hk.Params, scale: Numeric
    ) -> Tuple[PRNGKey, Array, Array]:
        new_key, key = jax.random.split(key)
        obs = jnp.expand_dims(obs, axis=0)  # (1, obs.shape)
        dist = distrax.Normal(net.apply(params, obs), scale)
        act = jnp.squeeze(dist.sample(seed=key), axis=0)  # (, )
        log_prob = jnp.squeeze(dist.log_prob(act), axis=0)  # (, )
        return new_key, act, log_prob

    return jax.jit(net_act)


def build_squashed_normal_net_act(net: hk.Transformed) -> NETACT:
    """Build a net_act function of a squashed normal distribution with
    diagonal covariance matrix.

    Args:
        net(hk.Transformed):
            Takes [1 x obs] and returns [1 x dA] Array.

    Returns:
        NETACT
    """

    def net_act(
        key: PRNGKey, obs: Array, params: hk.Params, scale: Numeric
    ) -> Tuple[PRNGKey, Array, Array]:
        new_key, key = jax.random.split(key)
        obs = jnp.expand_dims(obs, axis=0)  # (1, obs.shape)
        dist = srl.SquashedNormal(net.apply(params, obs), scale)
        act = jnp.squeeze(dist.sample(seed=key), axis=0)  # (, )
        log_prob = jnp.squeeze(dist.log_prob(act), axis=0)  # (, )
        return new_key, act, log_prob

    return jax.jit(net_act)
