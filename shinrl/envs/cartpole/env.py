"""
Author: Toshinori Kitamura
Affiliation: NAIST
"""
from functools import cached_property
from typing import Tuple

import gym
import jax.numpy as jnp
import numpy as np
from chex import Array

from shinrl import ShinEnv

from .core import calc
from .core.config import CartPoleConfig


class CartPole(ShinEnv):
    """Dynamics and reward are based on OpenAI gym's implementation of CartPole-v0"""

    DefaultConfig = CartPoleConfig

    @property
    def config(self) -> CartPoleConfig:
        return self._config

    @property
    def dS(self) -> int:
        return (
            self.config.x_res
            * self.config.x_dot_res
            * self.config.th_res
            * self.config.th_dot_res
        )

    @property
    def dA(self) -> int:
        return self.config.dA

    @cached_property
    def init_probs(self) -> Array:
        ini_x = 0
        ini_x_dot = 0
        th_step = 2 * self.config.th_max / (self.config.th_res - 1)
        th_dot_step = 2 * self.config.th_dot_max / (self.config.th_dot_res - 1)
        ini_th = np.arange(-0.02, 0.02, th_step)
        ini_th_dot = np.arange(-0.02, 0.02, th_dot_step)

        idxs = []
        for i_th in ini_th:
            for i_th_dot in ini_th_dot:
                idxs.append(
                    calc.x_th_to_state(self.config, ini_x, ini_x_dot, i_th, i_th_dot)
                )
        idxs = np.unique(np.array(idxs))
        probs = np.ones_like(idxs, dtype=float) / len(idxs)
        init_probs = np.zeros(self.dS)
        np.put(init_probs, idxs, probs)
        return jnp.array(init_probs)

    @cached_property
    def observation_space(self) -> gym.spaces.Space:
        high = jnp.array(
            [
                self.config.x_max,
                self.config.x_dot_max,
                self.config.th_max,
                self.config.th_dot_max,
            ]
        )
        space = gym.spaces.Box(low=-high, high=high, dtype=float)
        return space

    @cached_property
    def action_space(self) -> gym.spaces.Space:
        if self.config.act_mode == CartPoleConfig.ACT_MODE.discrete:
            space = gym.spaces.Discrete(self.config.dA)
        elif self.config.act_mode == CartPoleConfig.ACT_MODE.continuous:
            space = gym.spaces.Box(
                low=np.array(-self.config.force_max),
                high=np.array(self.config.force_max),
                dtype=float,
            )
        return space

    def transition(self, state: int, action: int) -> Tuple[Array, Array]:
        return calc.transition(self.config, state, action)

    def reward(self, state: int, action: int) -> float:
        return calc.reward(self.config, state, action)

    def observation(self, state: int) -> Array:
        return calc.observation_tuple(self.config, state)
