"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from chex import Array

import shinrl as srl

from .config import MountainCarConfig


@jax.jit
def to_discrete_act(config: MountainCarConfig, c_act: float) -> int:
    """Convert a continuous action to a discrete action.

    Args:
        config (MountainCarConfig)
        c_act (float): Continuous action in range [-1, 1].

    Returns:
        A discretized action id.
    """
    chex.assert_type(c_act, float)
    dA = config.dA
    c_act = jnp.clip(c_act, -1.0, 1.0)
    c_step = 2 / dA
    act = jnp.floor((c_act + 1.0) / c_step + 1e-5).astype(jnp.uint32)
    return jnp.clip(act, 0, dA - 1)


@jax.jit
def to_continuous_act(config: MountainCarConfig, act: int) -> float:
    """Convert a discrete action to a continuous action.

    Args:
        config (MountainCarConfig)
        act (int): Discrete action in [0, ..., dA-1].

    Returns:
        A continuous action in range [-1.0, 1.0]
    """
    chex.assert_type(act, int)
    dA = config.dA
    c_step = 2 / dA
    c_act = act * c_step - 1.0
    return jnp.clip(c_act, -1.0, 1.0)


@jax.jit
def state_to_pos_vel(config: MountainCarConfig, state: int) -> Tuple[float, float]:
    """Convert a state id to position and velocity.

    Args:
        config (MountainCarConfig)
        state (int)

    Returns:
        position and velocity
    """
    pos_res, vel_res = config.pos_res, config.vel_res
    pos_max, vel_max = config.pos_max, config.vel_max
    pos_min, vel_min = config.pos_min, config.vel_min
    pos_idx = state % pos_res
    vel_idx = state // vel_res
    pos = pos_min + (pos_max - pos_min) / (pos_res - 1) * pos_idx
    pos = jnp.clip(pos, pos_min, pos_max)
    vel = vel_min + (vel_max - vel_min) / (vel_res - 1) * vel_idx
    vel = jnp.clip(vel, vel_min, vel_max)
    return pos, vel


@jax.jit
def pos_vel_to_state(config: MountainCarConfig, pos: float, vel: float) -> float:
    """Convert position and velocity to state id

    Args:
        config (MountainCarConfig)
        pos (float): pos value
        vel (float): velocity value

    Returns:
        state id (int)
    """
    pos_res, vel_res = config.pos_res, config.vel_res
    pos_max, vel_max = config.pos_max, config.vel_max
    pos_min, vel_min = config.pos_min, config.vel_min
    pos_step = (pos_max - pos_min) / (pos_res - 1)
    vel_step = (vel_max - vel_min) / (vel_res - 1)
    pos_idx = jnp.floor((pos - pos_min) / pos_step + 1e-5)
    vel_idx = jnp.floor((vel - vel_min) / vel_step + 1e-5)
    state = (pos_idx + pos_res * vel_idx).astype(jnp.uint32)
    return jnp.clip(state, 0, pos_res * vel_res - 1)


@jax.jit
def transition(
    config: MountainCarConfig, state: int, action: int
) -> Tuple[Array, Array]:
    chex.assert_type([state, action], int)
    c_act = to_continuous_act(config, action)
    force = jnp.squeeze(c_act) * config.force_mag

    def body_fn(_, pos_vel):
        pos, vel = pos_vel
        vel = vel + force + jnp.cos(3 * pos) * (-0.0025)
        vel = jnp.clip(vel, config.vel_min, config.vel_max)
        pos = pos + vel
        pos = jnp.clip(pos, config.pos_min, config.pos_max)
        return (pos, vel)

    pos, vel = state_to_pos_vel(config, state)
    # one step is not enough when state is discretized
    pos, vel = jax.lax.fori_loop(0, 8, body_fn, (pos, vel))
    vel = jax.lax.cond(pos == config.pos_min, lambda _: 0.0, lambda _: vel, None)
    next_state = pos_vel_to_state(config, pos, vel).reshape((1,))
    prob = jnp.array((1.0,), dtype=float)
    return next_state, prob


@jax.jit
def reward(config: MountainCarConfig, state: int, action: int) -> float:
    pos, vel = state_to_pos_vel(config, state)
    goal = pos >= config.goal_pos
    return jax.lax.cond(goal, lambda _: 0.0, lambda _: -1.0, None)


@jax.jit
def observation_tuple(config: MountainCarConfig, state: int) -> Array:
    """Make the tuple observation."""
    pos, vel = state_to_pos_vel(config, state)
    return jnp.array([pos, vel], dtype=float)


@jax.jit
def observation_image(config: MountainCarConfig, state: int) -> Array:
    """Make the image observation."""
    pos, vel = state_to_pos_vel(config, state)
    image = jnp.zeros((28, 28), dtype=float)
    pos2pxl = 28 / (config.pos_max - config.pos_min)
    to_hight = lambda _x: jnp.sin(3 * _x) * 0.45 + 0.75
    x = ((pos - config.pos_min) * pos2pxl).astype(jnp.uint32)
    y = (to_hight(pos - config.pos_min) * pos2pxl).astype(jnp.uint32)
    pos_circle = srl.draw_circle(image, x, y, 4)
    image = image + pos_circle * 0.8

    x = ((pos - vel * 5.0 - config.pos_min) * pos2pxl).astype(jnp.uint32)
    y = (to_hight(pos - vel * 5.0 - config.pos_min) * pos2pxl).astype(jnp.uint32)
    vel_circle = srl.draw_circle(image, x, y, 4)
    image = image + vel_circle * 0.2
    return jnp.expand_dims(image, axis=-1)  # 28x28x1
