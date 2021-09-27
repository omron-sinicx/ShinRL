"""
Author: Toshinori Kitamura
Affiliation: NAIST
"""
import enum
from typing import ClassVar, Type

import chex

from shinrl import EnvConfig


class OBS_MODE(enum.IntEnum):
    random = enum.auto()
    onehot = enum.auto()


@chex.dataclass
class MazeConfig(EnvConfig):
    """Config for Maze.

    Args:
    """

    OBS_MODE: ClassVar[Type[OBS_MODE]] = OBS_MODE
    obs_mode: OBS_MODE = OBS_MODE.random

    rew_default: float = 0.0
    eps: float = 0.1
    discount: float = 0.99
    horizon: int = 30
    random_obs_dim: int = 5
    random_obs_seed: int = 0
