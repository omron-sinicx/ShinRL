import cv2
import gym
import numpy as np

from shinrl.envs import ShinEnv


class CartPole(ShinEnv):
    """Dynamics and reward are based on OpenAI gym's implementation of CartPole-v0

    Args:
        x_disc (int, optional): Resolution of :math:`x`.
        x_dot_disc (int, optional): Resolution of :math:`\\dot{x}`.
        th_disc (int, optional): Resolution of :math:`\\theta`.
        th_dot_disc (int, optional): Resolution of :math:`\\dot{\\theta}`.
        action_mode (str): Specify the type of action. "discrete" or "continuous".
        obs_mode (str): Specify the type of observation. "tuple" or "image".
    """

    def __init__(
        self,
        x_disc=128,
        x_dot_disc=4,
        th_disc=64,
        th_dot_disc=4,
        dA=5,
        init_dist=None,
        horizon=100,
        action_mode="discrete",
        obs_mode="tuple",
    ):
        # env parameters
        self.x_disc = x_disc
        self.x_dot_disc = x_dot_disc
        self.th_disc = th_disc
        self.th_dot_disc = th_dot_disc
        self.dA = dA

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.force_list = np.linspace(-self.force_mag, self.force_mag, num=self.dA)
        self.tau = 0.1  # seconds between state updates

        # Angle at which to fail the episode
        self.max_x = 2.0
        self.max_x_dot = 0.5
        self.max_th = 0.25
        self.max_th_dot = 0.2

        self.action_step = (self.force_mag * 2) / self.dA
        self.x_step = (self.max_x * 2) / (self.x_disc - 1)
        self.x_dot_step = (self.max_x_dot * 2) / (self.x_dot_disc - 1)
        self.th_step = (self.max_th * 2) / (self.th_disc - 1)
        self.th_dot_step = (self.max_th_dot * 2) / (self.th_dot_disc - 1)

        # assert self.max_x_dot*self.tau > self.x_step, "x resolution is not enough "
        # assert self.max_th_dot*self.tau > self.th_step, "th resolution is not enough "

        # set initial state probability
        ini_x = 0
        ini_x_dot = 0
        ini_ths = np.arange(-0.02, 0.02, self.th_step)
        ini_th_dots = np.arange(-0.02, 0.02, self.th_dot_step)

        idxs = []
        for ini_th in ini_ths:
            for ini_th_dot in ini_th_dots:
                idxs.append(self.to_state_id(ini_x, ini_x_dot, ini_th, ini_th_dot))
        idxs = set(idxs)
        random_init = {idx: 1 / len(idxs) for idx in idxs}
        init_dist = random_init if init_dist is None else init_dist
        super().__init__(
            x_disc * x_dot_disc * th_disc * th_dot_disc, dA, init_dist, horizon=horizon
        )

        high = np.array(
            [self.max_x, self.max_x_dot, self.max_th, self.max_th_dot], dtype=np.float32
        )

        self.obs_mode = obs_mode
        if obs_mode == "tuple":
            self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        else:
            raise NotImplementedError

        self.action_mode = action_mode
        if action_mode == "discrete":
            self.action_space = gym.spaces.Discrete(dA)
        elif action_mode == "continuous":
            self.action_space = gym.spaces.Box(
                low=np.array((-self.force_mag,)), high=np.array((self.force_mag,))
            )
        else:
            raise ValueError

        # for rendering
        gym.make("CartPole-v0")
        self.render_env = gym.envs.classic_control.CartPoleEnv()
        self.render_env.reset()

    def discretize_action(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high - 1e-5)
        return int(np.floor((action - self.action_space.low) / self.action_step))

    def to_continuous_action(self, action):
        return action * self.action_step + self.action_space.low

    def transition(self, state, action):
        transition = {}

        x, x_dot, theta, theta_dot = self.from_state_id(state)
        force = self.force_list[action]

        if np.abs(x) >= self.max_x or np.abs(theta) >= self.max_th:
            transition[state] = 1.0
            return transition

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        x = np.clip(x, -self.max_x, self.max_x)
        x_dot = np.clip(x_dot, -self.max_x_dot, self.max_x_dot)
        theta = np.clip(theta, -self.max_th, self.max_th)
        theta_dot = np.clip(theta_dot, -self.max_th_dot, self.max_th_dot)

        next_state = self.to_state_id(x, x_dot, theta, theta_dot)
        transition[next_state] = 1.0
        return transition

    def reward(self, state, action):
        if self.action_mode == "continuous":
            action = self.discretize_action(action)
        x, _, th, _ = self.from_state_id(state)
        if np.abs(x) >= self.max_x or np.abs(th) >= self.max_th:
            return 0.0
        else:
            return 1.0

    def observation(self, state):
        x, x_dot, th, th_dot = self.from_state_id(state)
        if self.obs_mode == "tuple":
            return np.array([x, x_dot, th, th_dot], dtype=np.float32)
        elif self.obs_mode == "image":
            raise NotImplementedError

    def from_state_id(self, state):
        x_idx = state % self.x_disc
        state = (state - x_idx) / self.x_disc
        x_dot_idx = state % self.x_dot_disc
        state = (state - x_dot_idx) / self.x_dot_disc
        th_idx = state % self.th_disc
        th_dot_idx = (state - th_idx) / self.th_disc

        x = -self.max_x + self.x_step * x_idx
        x_dot = -self.max_x_dot + self.x_dot_step * x_dot_idx
        th = -self.max_th + self.th_step * th_idx
        th_dot = -self.max_th_dot + self.th_dot_step * th_dot_idx
        return x, x_dot, th, th_dot

    def to_state_id(self, x, x_dot, th, th_dot):
        # round
        x_idx = int(np.floor((x + self.max_x) / self.x_step))
        x_dot_idx = int(np.floor((x_dot + self.max_x_dot) / self.x_dot_step))
        th_idx = int(np.floor((th + self.max_th) / self.th_step))
        th_dot_idx = int(np.floor((th_dot + self.max_th_dot) / self.th_dot_step))
        state_id = x_idx + self.x_disc * (
            x_dot_idx + self.x_dot_disc * (th_idx + self.th_disc * th_dot_idx)
        )
        return state_id

    def render(self):
        state = self.get_state()
        x, x_dot, th, th_dot = self.from_state_id(state)
        self.render_env.state = (x, x_dot, th, th_dot)
        self.render_env.render()

    def close(self):
        self.render_env.close()
