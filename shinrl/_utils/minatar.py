"""
MinAtar environment made compatible for Gym.
See https://github.com/kenjyoung/MinAtar to install minatar.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""

import gym
from gym.wrappers import TimeLimit

try:
    import minatar
except ImportError:
    pass
import numpy as np


class MinAtarEnv(gym.Env):
    def __init__(self, game_name):
        self.env = minatar.Environment(env_name=game_name)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=self.env.state_shape(), dtype=np.bool
        )
        self.action_space = gym.spaces.Discrete(self.env.num_actions())

    def reset(self):
        self.env.reset()
        return self.env.state()

    def step(self, action):
        r, terminal = self.env.act(action)
        return self.env.state(), r, terminal, {}

    def render(self):
        self.env.display_state(10)
        self.env.close_display()


def make_minatar(game_name):
    return TimeLimit(MinAtarEnv(game_name), max_episode_steps=10000)
