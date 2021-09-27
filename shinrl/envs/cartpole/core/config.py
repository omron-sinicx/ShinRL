"""
Author: Toshinori Kitamura
Affiliation: NAIST
"""
import enum
from typing import ClassVar, Type

import chex

from shinrl import EnvConfig


class ACT_MODE(enum.IntEnum):
    discrete = enum.auto()
    continuous = enum.auto()


@chex.dataclass
class CartPoleConfig(EnvConfig):
    """Config for CartPole.

    Args:
    """

    ACT_MODE: ClassVar[Type[ACT_MODE]] = ACT_MODE
    act_mode: ACT_MODE = ACT_MODE.discrete

    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5
    force_max: float = 10.0
    tau: float = 0.2

    x_res: float = 32
    x_dot_res: float = 32
    th_res: float = 32
    th_dot_res: float = 32

    x_max: float = 1.0
    x_dot_max: float = 0.5
    th_max: float = 0.25
    th_dot_max: float = 0.5

    dA: int = 3
    discount: float = 0.99
    horizon: int = 200
