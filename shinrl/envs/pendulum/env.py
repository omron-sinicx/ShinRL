"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from typing import Any, Optional

import gym
import jax.numpy as jnp
import numpy as np
from chex import Array
from matplotlib.axes import Axes

from shinrl import ShinEnv

from .calc import (
    observation_image,
    observation_tuple,
    reward,
    th_vel_to_state,
    to_continuous_act,
    to_discrete_act,
    transition,
)
from .config import PendulumConfig
from .plot import plot_S


class Pendulum(ShinEnv):
    """Dynamics and reward are based on OpenAI gym's implementation of Pendulum-v0"""

    DefaultConfig = PendulumConfig

    @property
    def config(self) -> PendulumConfig:
        return self._config

    @property
    def dS(self) -> int:
        return self.config.theta_res * self.config.vel_res

    @property
    def dA(self) -> int:
        return self.config.dA

    @property
    def observation_space(self) -> gym.spaces.Space:
        if self.config.obs_mode == PendulumConfig.OBS_MODE.tuple:
            space = gym.spaces.Box(
                low=np.array([0, 0, -self.config.vel_max]),
                high=np.array([1, 1, self.config.vel_max]),
                dtype=float,
            )
        elif self.config.obs_mode == PendulumConfig.OBS_MODE.image:
            space = gym.spaces.Box(
                low=np.zeros((28, 28, 1)),
                high=np.ones((28, 28, 1)),
                dtype=float,
            )
        return space

    @property
    def action_space(self) -> gym.spaces.Space:
        if self.config.act_mode == PendulumConfig.ACT_MODE.discrete:
            space = gym.spaces.Discrete(self.config.dA)
        elif self.config.act_mode == PendulumConfig.ACT_MODE.continuous:
            space = gym.spaces.Box(
                low=np.array((-1.0,)),
                high=np.array((1.0,)),
                dtype=float,
            )
        return space

    def init_probs(self) -> Array:
        th_step = (2 * jnp.pi) / (self.config.theta_res - 1)
        vel_step = (2 * self.config.vel_max) / (self.config.vel_res - 1)
        ini_ths = jnp.arange(-jnp.pi, jnp.pi, th_step)
        ini_vels = jnp.arange(-1, 1, vel_step)
        idxs = []
        for ini_th in ini_ths:
            for ini_vel in ini_vels:
                idxs.append(th_vel_to_state(self.config, ini_th, ini_vel))
        idxs = np.unique(np.array(idxs))
        probs = np.ones_like(idxs, dtype=float) / len(idxs)
        init_probs = np.zeros(self.dS)
        np.put(init_probs, idxs, probs)
        return jnp.array(init_probs)

    def transition(self, state, action):
        return transition(self.config, state, action)

    def reward(self, state, action):
        return reward(self.config, state, action)

    def observation(self, state):
        if self.config.obs_mode == PendulumConfig.OBS_MODE.tuple:
            obs_fn = observation_tuple
        elif self.config.obs_mode == PendulumConfig.OBS_MODE.image:
            obs_fn = observation_image
        return obs_fn(self.config, state)

    def continuous_action(self, act):
        return to_continuous_act(self.config, jnp.array([act]))

    def discrete_action(self, c_act):
        return to_discrete_act(self.config, c_act)

    def plot_S(
        self,
        tb: Array,
        title: Optional[str] = None,
        ax: Optional[Axes] = None,
        cbar_ax: Optional[Axes] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        fontsize: Optional[int] = 10,
        **kwargs: Any,
    ) -> None:
        plot_S(tb, self.config, title, ax, cbar_ax, vmin, vmax, fontsize, **kwargs)
