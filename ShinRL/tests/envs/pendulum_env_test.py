import numpy as np
import pytest

from shinrl.envs import Pendulum

from .misc import check_continuous_env, check_discrete_env


def test_pendulum_env():
    env = Pendulum(state_disc=5, dA=3)
    check_discrete_env(env)


def test_pendulum_continuous_env():
    env = Pendulum(state_disc=5, dA=3, action_mode="continuous")
    check_continuous_env(env)


def test_image_obs():
    env = Pendulum(state_disc=5, dA=3, obs_mode="image")
    env.reset()
    for i in range(env.horizon - 1):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        assert obs.shape == (1, 28, 28)
