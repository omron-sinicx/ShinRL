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

from shinrl import OBS_FN, REW_FN, TRAN_FN, ShinEnv

from .core import calc, plot
from .core.config import PendulumConfig


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
                low=np.array(-self.config.torque_max),
                high=np.array(self.config.torque_max),
                dtype=float,
            )
        return space

    def _init_probs(self) -> Array:
        th_step = (2 * jnp.pi) / (self.config.theta_res - 1)
        vel_step = (2 * self.config.vel_max) / (self.config.vel_res - 1)
        ini_ths = jnp.arange(-jnp.pi, jnp.pi, th_step)
        ini_vels = jnp.arange(-1, 1, vel_step)
        idxs = []
        for ini_th in ini_ths:
            for ini_vel in ini_vels:
                idxs.append(calc.th_vel_to_state(self.config, ini_th, ini_vel))
        idxs = np.unique(np.array(idxs))
        probs = np.ones_like(idxs, dtype=float) / len(idxs)
        init_probs = np.zeros(self.dS)
        np.put(init_probs, idxs, probs)
        return jnp.array(init_probs)

    def _make_transition_fn(self) -> TRAN_FN:
        return lambda state, action: calc.transition(self.config, state, action)

    def _make_reward_fn(self) -> REW_FN:
        return lambda state, action: calc.reward(self.config, state, action)

    def _make_observation_fn(self) -> OBS_FN:
        if self.config.obs_mode == PendulumConfig.OBS_MODE.tuple:
            obs_fn = calc.observation_tuple
        elif self.config.obs_mode == PendulumConfig.OBS_MODE.image:
            obs_fn = calc.observation_image
        return lambda state: obs_fn(self.config, state)

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
        plot.plot_S(tb, self.config, title, ax, cbar_ax, vmin, vmax, fontsize, **kwargs)
