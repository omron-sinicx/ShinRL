import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.axes import Axes
from typing import Optional
from shinrl.envs import ShinEnv


class Pendulum(ShinEnv):
    """Dynamics and reward are based on OpenAI gym's implementation of Pendulum-v0

    Args:
        state_disc (int, optional): Resolution of :math:`\\theta` and :math:`\\dot{\\theta}`.
        action_mode (str): Specify the type of action. "discrete" or "continuous".
        obs_mode (str): Specify the type of observation. "tuple" or "image".
    """

    def __init__(
        self,
        state_disc=32,
        dA=5,
        init_dist=None,
        horizon=200,
        action_mode="discrete",
        obs_mode="tuple",
    ):
        self.state_disc = state_disc
        self.dA = dA
        self.max_vel = 8.0
        self.max_torque = 2.0
        self.min_torque = -self.max_torque
        self.torque_step = (2 * self.max_torque) / dA

        self.action_map = np.linspace(-self.max_torque, self.max_torque, num=dA)
        self.state_min = -np.pi
        self.state_step = (2 * np.pi) / (state_disc - 1)
        self.vel_min = -self.max_vel
        self.vel_step = (2 * self.max_vel) / (state_disc - 1)
        # for rendering
        gym.make("Pendulum-v0")
        self.render_env = gym.envs.classic_control.PendulumEnv()
        self.render_env.reset()

        # observation
        if obs_mode != "tuple" and obs_mode != "image":
            raise ValueError("Invalid obs_mode: {}".format(obs_mode))
        self.obs_mode = obs_mode
        self.base_image = np.zeros((28, 28))

        # set initial state probability
        ini_ths = np.arange(-np.pi, np.pi, self.state_step)
        ini_thvs = np.arange(-1, 1, self.vel_step)
        idxs = []
        for ini_th in ini_ths:
            for ini_thv in ini_thvs:
                idxs.append(self.to_state_id(ini_th, ini_thv))
        idxs = set(idxs)
        random_init = {idx: 1 / len(idxs) for idx in idxs}
        init_dist = random_init if init_dist is None else init_dist
        super().__init__(state_disc * state_disc, dA, init_dist, horizon=horizon)

        self.action_mode = action_mode
        if action_mode == "discrete":
            self.action_space = gym.spaces.Discrete(dA)
        elif action_mode == "continuous":
            self.action_space = gym.spaces.Box(
                low=np.array((self.min_torque,)), high=np.array((self.max_torque,))
            )
        else:
            raise ValueError

        if obs_mode == "tuple":
            self.observation_space = gym.spaces.Box(
                low=np.array([0, 0, -self.max_vel]),
                high=np.array([1, 1, self.max_vel]),
                dtype=np.float32,
            )
        elif obs_mode == "image":
            self.observation_space = gym.spaces.Box(
                low=np.expand_dims(np.zeros((28, 28)), axis=0),
                high=np.expand_dims(np.ones((28, 28)), axis=0),
                dtype=np.float32,
            )

    def discretize_action(self, action):
        # continuous to discrete action
        action = np.clip(action, self.action_space.low, self.action_space.high - 1e-5)
        return int(np.floor((action - self.action_space.low) / self.torque_step))

    def to_continuous_action(self, action):
        return action * self.torque_step + self.action_space.low

    def th_thv_from_state_id(self, state):
        th_idx = state % self.state_disc
        vel_idx = state // self.state_disc
        th = self.state_min + self.state_step * th_idx
        thv = self.vel_min + self.vel_step * vel_idx
        return th, thv

    def to_state_id(self, theta, thetav):
        # round
        th_round = int(np.floor((theta - self.state_min) / self.state_step))
        th_vel = int(np.floor((thetav - self.vel_min) / self.vel_step))
        return th_round + self.state_disc * th_vel

    def transition(self, state, action):
        transition = {}

        # pendulum dynamics
        g, m, l, dt = 10.0, 1.0, 1.0, 0.05
        torque = self.action_map[action]
        theta, thetav = self.th_thv_from_state_id(state)

        for _ in range(3):  # one step is not enough when state is discretized
            thetav = (
                thetav
                + (
                    -3 * g / (2 * l) * np.sin(theta + np.pi)
                    + 3.0 / (m * l ** 2) * torque
                )
                * dt
            )
            theta = theta + thetav * dt
            thetav = max(min(thetav, self.max_vel - 1e-8), -self.max_vel)
            if theta < -np.pi:
                theta += 2 * np.pi
            if theta >= np.pi:
                theta -= 2 * np.pi
        next_state = self.to_state_id(theta, thetav)

        transition[next_state] = 1.0
        return transition

    def reward(self, state, action):
        if self.action_mode == "continuous":
            action = self.discretize_action(action)
        torque = self.action_map[action]
        theta, thetav = self.th_thv_from_state_id(state)
        # OpenAI gym reward
        normed_theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        cost = normed_theta ** 2 + 0.1 * (thetav ** 2) + 0.001 * (torque ** 2)
        return -cost

    def observation(self, state):
        theta, thetav = self.th_thv_from_state_id(state)
        if self.obs_mode == "tuple":
            return np.array([np.cos(theta), np.sin(theta), thetav], dtype=np.float32)
        elif self.obs_mode == "image":
            image = self.base_image.copy()
            length = 9
            x = int(14 + length * np.cos(theta + np.pi / 2))
            y = int(14 - length * np.sin(theta + np.pi / 2))
            image = cv2.line(image, (14, 14), (x, y), 0.8, thickness=1)

            vx = int(14 + length * np.cos((theta - thetav * 0.15) + np.pi / 2))
            vy = int(14 - length * np.sin((theta - thetav * 0.15) + np.pi / 2))
            image = cv2.line(image, (14, 14), (vx, vy), 0.2, thickness=1)

            return np.expand_dims(image, axis=0)  # 1x28x28

    def disc_th_thv(self, theta, thetav):
        th_round = int(np.floor((theta - self.state_min) / self.state_step))
        th_vel = int(np.floor((thetav - self.vel_min) / self.vel_step))
        return th_round, th_vel

    def undisc_th_thv(self, th_round, th_vel):
        th = th_round * self.state_step + self.state_min
        thv = th_vel * self.vel_step + self.vel_min
        return th, thv

    def render(self):
        theta, thetav = self.th_thv_from_state_id(self._state)
        self.render_env.state = (theta, thetav)
        self.render_env.render()

    def close(self):
        self.render_env.close()

    def plot_S(
        self,
        values: np.ndarray,
        title: Optional[str] = None,
        ax: Optional[Axes] = None,
        cbar_ax: Optional[Axes] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        fontsize: Optional[int] = 10,
        **kwargs,
    ) -> None:
        reshaped_values = np.empty((self.state_disc, self.state_disc))
        reshaped_values[:] = np.nan
        for s in range(len(values)):
            th, thv = self.th_thv_from_state_id(s)
            th, thv = self.disc_th_thv(th, thv)
            reshaped_values[th, thv] = values[s]

        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        vmin = reshaped_values.min() if vmin is None else vmin
        vmax = reshaped_values.max() if vmax is None else vmax

        th_ticks, thv_ticks = [], []
        for i in range(self.state_disc):
            th, thv = self.undisc_th_thv(i, i)
            th_ticks.append(round(th, 3))
            thv_ticks.append(round(thv, 3))

        data = pd.DataFrame(reshaped_values, index=th_ticks, columns=thv_ticks)
        data = data.ffill(axis=0)
        data = data.ffill(axis=1)
        show_cbar = True if cbar_ax is not None else False
        sns.heatmap(
            data,
            ax=ax,
            cbar=show_cbar,
            cbar_ax=cbar_ax,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
        ax.set_title(title, fontsize=fontsize)
        ax.set_ylabel(r"$\theta$", fontsize=fontsize)
        ax.set_xlabel(r"$\dot{\theta}$", fontsize=fontsize)
