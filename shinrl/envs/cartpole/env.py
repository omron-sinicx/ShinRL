"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""

import gym
import jax.numpy as jnp
import numpy as np
from chex import Array

from shinrl import ShinEnv

from .calc import (
    observation_tuple,
    reward,
    to_continuous_act,
    to_discrete_act,
    transition,
    x_th_to_state,
)
from .config import CartPoleConfig


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

    @property
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

    @property
    def action_space(self) -> gym.spaces.Space:
        if self.config.act_mode == CartPoleConfig.ACT_MODE.discrete:
            space = gym.spaces.Discrete(self.config.dA)
        elif self.config.act_mode == CartPoleConfig.ACT_MODE.continuous:
            space = gym.spaces.Box(
                low=np.array((-1.0,), dtype=float),
                high=np.array((1.0,), dtype=float),
                dtype=float,
            )
        return space

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
                    x_th_to_state(self.config, ini_x, ini_x_dot, i_th, i_th_dot)
                )
        idxs = np.unique(np.array(idxs, dtype=int))
        probs = np.ones_like(idxs, dtype=float) / len(idxs)
        init_probs = np.zeros(self.dS, dtype=float)
        np.put(init_probs, idxs, probs)
        return jnp.array(init_probs)

    def transition(self, state, action):
        return transition(self.config, state, action)

    def reward(self, state, action):
        return reward(self.config, state, action)

    def observation(self, state):
        return observation_tuple(self.config, state)

    def continuous_action(self, act):
        return to_continuous_act(self.config, jnp.array([act]))

    def discrete_action(self, c_act):
        return to_discrete_act(self.config, c_act)
