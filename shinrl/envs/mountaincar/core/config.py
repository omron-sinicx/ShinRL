"""
Author: Toshinori Kitamura
Affiliation: NAIST
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
class MountainCarConfig(EnvConfig):
    """Config for MountainCar.

    Args:
    """

    OBS_MODE: ClassVar[Type[OBS_MODE]] = OBS_MODE
    ACT_MODE: ClassVar[Type[ACT_MODE]] = ACT_MODE
    obs_mode: OBS_MODE = OBS_MODE.tuple
    act_mode: ACT_MODE = ACT_MODE.discrete

    force_max: float = 2.0
    goal_pos: float = 0.5

    pos_max: float = 0.6
    pos_min: float = -1.2
    pos_res: int = 32

    vel_max: float = 0.07
    vel_min: float = -0.07
    vel_res: int = 32

    dA: int = 3
    discount: float = 0.99
    horizon: int = 200
