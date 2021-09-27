"""
Author: Toshinori Kitamura
Affiliation: NAIST
"""
import enum
from typing import ClassVar, Type

import chex
import jax.numpy as jnp

from shinrl import EnvConfig


class OBS_MODE(enum.IntEnum):
    tuple = enum.auto()
    image = enum.auto()


class ACT_MODE(enum.IntEnum):
    discrete = enum.auto()
    continuous = enum.auto()


@chex.dataclass
class PendulumConfig(EnvConfig):
    """Config for Pendulum.

    Args:
        torque_max (float): Maximum torque.
        theta_max (float): Maximum angle.
        theta_res (int): Resolution of discretizing angle.
        vel_max (float): Maximum velocity.
        vel_res (int): Resolution of discretizing velocity.
        obs_mode (OBS_MODE): Type of observation.
        act_mode (ACT_MODE): Type of action.
        gravity (float): gravity.
        mass (float): pendulum's mass.
        length (float): pendulum's length.
    """

    # class variables
    OBS_MODE: ClassVar[Type[OBS_MODE]] = OBS_MODE
    ACT_MODE: ClassVar[Type[ACT_MODE]] = ACT_MODE

    # fields
    torque_max: float = 2.0
    theta_max: float = jnp.pi
    theta_res: int = 32
    vel_max: float = 8.0
    vel_res: int = 32
    obs_mode: OBS_MODE = OBS_MODE.tuple
    act_mode: ACT_MODE = ACT_MODE.discrete
    dA: int = 3
    discount: float = 0.99
    horizon: int = 200
    gravity: float = 10.0
    mass: float = 1.0
    length: float = 1.0
