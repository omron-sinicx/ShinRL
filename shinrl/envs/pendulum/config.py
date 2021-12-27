"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
import enum
from typing import ClassVar, Type

import chex

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
        torque_mag (float): Magnitude of the input torque.
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
    dt: float = 0.06
    torque_mag: float = 3.0
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
