"""
Author: Toshinori Kitamura
Affiliation: NAIST
"""
import chex

from shinrl import Config


@chex.dataclass
class EnvConfig(Config):
    """Config for ShinEnv.

    Args:
        discount (float): MDP's discount factor.
        horizon (int, optional): Environment's Horizon.
    """

    discount: float
    horizon: int
