"""
Author: Toshinori Kitamura
Affiliation: NAIST
"""
from typing import Tuple, Union

import jax
import jax.numpy as jnp
from chex import Array

import shinrl as srl

from .config import PendulumConfig


@jax.jit
def torque_to_act(config: PendulumConfig, torque: float) -> int:
    """Convert torque to a discrete action.

    Args:
        config (PendulumConfig)
        torque (float): Continuous action.

    Returns:
        A discretized action id.
    """
    torque_max, dA = config.torque_max, config.dA
    action = jnp.clip(torque, -torque_max, torque_max - 1e-5)
    torque_step = (2 * torque_max) / dA
    return jnp.floor((action + torque_max) / torque_step).astype(int)


@jax.jit
def act_to_torque(config: PendulumConfig, act: int) -> float:
    """Convert a discrete action to a continuous action.

    Args:
        config (PendulumConfig)
        act (int): Discrete action.

    Returns:
        A continuous action
    """
    torque_max, dA = config.torque_max, config.dA
    torque_step = (2 * torque_max) / dA
    return act * torque_step - torque_max


@jax.jit
def state_to_th_vel(config: PendulumConfig, state: int) -> Tuple[float, float]:
    """Convert a state id to the angle and the angle velocity.

    Args:
        config (PendulumConfig)
        state (int)

    Returns:
        theta and vel_theta values.
    """
    th_res, vel_res = config.theta_res, config.vel_res
    th_max, vel_max = config.theta_max, config.vel_max
    th_idx = state % th_res
    vel_idx = state // vel_res
    th = -th_max + (2 * jnp.pi) / (th_res - 1) * th_idx
    vel = -vel_max + (2 * vel_max) / (vel_res - 1) * vel_idx
    return th, vel


@jax.jit
def th_vel_to_state(config: PendulumConfig, th: float, vel: float) -> float:
    """Convert the angle and the angle velocity to state id

    Args:
        config (PendulumConfig)
        th (float): theta value
        vel (float): velocity value

    Returns:
        state id (int)
    """
    th_res, vel_res = config.theta_res, config.vel_res
    th_max, vel_max = config.theta_max, config.vel_max
    th_step = (2 * jnp.pi) / (th_res - 1)
    vel_step = (2 * vel_max) / (vel_res - 1)
    th_round = jnp.floor((th + th_max) / th_step).astype(int)
    th_vel = jnp.floor((vel + vel_max) / vel_step).astype(int)
    return th_round + th_res * th_vel


@jax.jit
def transition(
    config: PendulumConfig, state: int, action: Union[int, float]
) -> Tuple[Array, Array]:
    g, m, l, dt = config.gravity, config.mass, config.length, 0.05
    is_continuous = config.act_mode == config.ACT_MODE.continuous
    torque = jax.lax.cond(
        is_continuous,
        lambda _: action.astype(float),
        lambda _: act_to_torque(config, action),
        None,
    )

    def body_fn(_, th_vel):
        th, vel = th_vel
        vel = (
            vel
            + (-3 * g / (2 * l) * jnp.sin(th + jnp.pi) + 3.0 / (m * l ** 2) * torque)
            * dt
        )
        th = th + vel * dt
        vel = jnp.array((vel, config.vel_max - 1e-8)).min()
        vel = jnp.array((vel, -config.vel_max)).max()
        sign = -1 + (th < -jnp.pi) * 2  # 1 if th < -jnp.pi else -1
        th = th + sign * 2 * jnp.pi
        return (th, vel)

    th, vel = state_to_th_vel(config, state)
    # one step is not enough when state is discretized
    th, vel = jax.lax.fori_loop(0, 4, body_fn, (th, vel))
    next_state = th_vel_to_state(config, th, vel).reshape((1,))
    prob = jnp.array((1.0,), dtype=float)
    return next_state, prob


@jax.jit
def reward(config: PendulumConfig, state: int, action: int) -> float:
    is_continuous = config.act_mode == config.ACT_MODE.continuous
    torque = jax.lax.cond(
        is_continuous,
        lambda _: action.astype(float),
        lambda _: act_to_torque(config, action),
        None,
    )
    th, vel = state_to_th_vel(config, state)
    # OpenAI gym reward
    normed_th = ((th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
    cost = normed_th ** 2 + 0.1 * (vel ** 2) + 0.001 * (torque ** 2)
    return -cost


@jax.jit
def observation_tuple(config: PendulumConfig, state: int) -> Array:
    """Make the tuple observation."""
    th, vel = state_to_th_vel(config, state)
    return jnp.array([jnp.cos(th), jnp.sin(th), vel], dtype=float)


@jax.jit
def observation_image(config: PendulumConfig, state: int) -> Array:
    """Make the image observation."""
    th, vel = state_to_th_vel(config, state)
    image = jnp.zeros((28, 28), dtype=float)
    length = 9
    x = (14 + length * jnp.cos(th + jnp.pi / 2)).astype(int)
    y = (14 - length * jnp.sin(th + jnp.pi / 2)).astype(int)
    cc, rr, val = srl.line_aa(10, 14, 14, x, y)
    image = image.at[cc, rr].add(val * 0.8)

    vx = (14 + length * jnp.cos((th - vel * 0.15) + jnp.pi / 2)).astype(int)
    vy = (14 - length * jnp.sin((th - vel * 0.15) + jnp.pi / 2)).astype(int)
    cc, rr, val = srl.line_aa(10, 14, 14, vx, vy)
    image = image.at[cc, rr].add(val * 0.2)
    return jnp.expand_dims(image, axis=-1)  # 28x28x1
