import gym
import numpy as np


class NormalizeActionWrapper(gym.Wrapper):
    """Gym wrapper to normalize actions into [-1, 1]."""

    def __init__(self, env):
        super().__init__(env)
        ones = np.ones_like(env.action_space.high)
        self.action_space = gym.spaces.Box(
            low=-ones,
            high=ones,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )

    def step(self, action):
        action = self.unnormalize(action)
        return self.env.step(action)

    def unnormalize(self, action: np.ndarray) -> np.ndarray:
        low = self.env.action_space.low
        high = self.env.action_space.high

        zero_to_one = (action + 1.0) / 2
        action = low + zero_to_one * (high - low)
        action = np.clip(action, low, high)

        return action
