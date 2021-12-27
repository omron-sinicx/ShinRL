""" Functions to build networks. 
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from typing import Any, Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array


def build_obs_forward_fc(
    n_out: int,
    depth: int,
    hidden: int,
    act_layer: Any,
    last_layer: Optional[Any] = None,
) -> hk.Transformed:
    """Build a simple fully-connected forward step that takes an observation.

    Args:
        n_out (int): Number of outputs.
        depth (int): Depth of layers.
        hidden (int): # of hidden units of fc.
        act_layer (Any): Activation layer.
        last_layer (Any): Last activation layer.

    Returns:
        hk.Transformed:
            Takes [batch x ?] observation and returns [batch x n_out] Array.
    """

    @jax.vmap
    def forward(obs: Array) -> Array:
        modules = []
        if depth > 0:
            modules.append(hk.Linear(hidden))
            for _ in range(depth - 1):
                modules += [act_layer, hk.Linear(hidden)]
            modules += [act_layer, hk.Linear(n_out)]
        else:
            modules.append(hk.Linear(n_out))
        if last_layer is not None:
            modules.append(last_layer)
        return hk.Sequential(modules)(obs.astype(float))

    return hk.without_apply_rng(hk.transform(forward))


def build_obs_forward_conv(
    n_out: int,
    depth: int,
    hidden: int,
    act_layer: Any,
    last_layer: Optional[Any] = None,
) -> hk.Transformed:
    """Build a simple cnn forward step that takes an observation.

    Args:
        n_out (int): Number of outputs.
        depth (int): Depth of layers.
        hidden (int): # of hidden units of fc.
        act_layer (Any): Activation layer.
        last_layer (Any): Last activation layer.

    Returns:
        hk.Transformed:
            Takes [batch x H x W x C] observation and returns [batch x n_out] Array.
    """

    def forward(obs: Array) -> Array:
        modules = [
            hk.Conv2D(16, kernel_shape=3, stride=1, padding="VALID"),
            hk.Flatten(),
        ]
        for _ in range(depth):
            modules += [act_layer, hk.Linear(hidden)]
        modules += [act_layer, hk.Linear(n_out)]
        if last_layer is not None:
            modules.append(last_layer)
        return hk.Sequential(modules)(obs.astype(float))

    return hk.without_apply_rng(hk.transform(forward))


def build_obs_act_forward_fc(
    n_out: int,
    depth: int,
    hidden: int,
    act_layer: Any,
    last_layer: Optional[Any] = None,
) -> hk.Transformed:
    """Build a simple fully-connected forward step that takes an observation & an action.

    Args:
        n_out (int): Number of outputs.
        depth (int): Depth of layers.
        hidden (int): # of hidden units of fc.
        act_layer (Any): Activation layer.
        last_layer (Any): Last activation layer.

    Returns:
        hk.Transformed:
            Takes [batch x ?] observation and [batch x ?] actions.
            Returns [batch x n_out] Array.
    """

    @jax.vmap
    def forward(obs: Array, act: Array) -> Array:
        # concat observation and action
        chex.assert_equal_rank((obs, act))
        obs_act = jnp.hstack((obs, act))

        # set up layers
        modules = []
        if depth > 0:
            modules.append(hk.Linear(hidden))
            for _ in range(depth - 1):
                modules += [act_layer, hk.Linear(hidden)]
            modules += [act_layer, hk.Linear(n_out)]
        else:
            modules.append(hk.Linear(n_out))
        if last_layer is not None:
            modules.append(last_layer)
        return hk.Sequential(modules)(obs_act.astype(float))

    return hk.without_apply_rng(hk.transform(forward))


def build_obs_act_forward_conv(
    n_out: int,
    depth: int,
    hidden: int,
    act_layer: Any,
    last_layer: Optional[Any] = None,
) -> hk.Transformed:
    """Build a simple cnn forward step that takes an observation & an action.

    Args:
        n_out (int): Number of outputs.
        depth (int): Depth of layers.
        hidden (int): # of hidden units of fc.
        act_layer (Any): Activation layer.
        last_layer (Any): Last activation layer.

    Returns:
        hk.Transformed:
            Takes [batch x H x W x C] observation and returns [batch x n_out] Array.
    """

    def forward(obs: Array, act: Array) -> Array:
        # set up cnn-layers
        cnn = [
            hk.Conv2D(16, kernel_shape=3, stride=1, padding="VALID"),
            hk.Flatten(),
        ]
        cnn_out = hk.Sequential(cnn)(obs.astype(float))

        # concat the output of cnn and the action
        chex.assert_equal_rank((cnn_out, act))
        fc_in = jnp.hstack((cnn_out, act))

        # set up fc-layers
        modules = []
        for _ in range(depth):
            modules += [act_layer, hk.Linear(hidden)]
        modules += [act_layer, hk.Linear(n_out)]
        if last_layer is not None:
            modules.append(last_layer)
        fc_out = hk.Sequential(modules)(fc_in.astype(float))
        return fc_out

    return hk.without_apply_rng(hk.transform(forward))
