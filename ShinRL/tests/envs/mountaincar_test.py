import numpy as np
import pytest

from shinrl.envs import MountainCar

from .misc import check_continuous_env, check_discrete_env


def test_mountaincar_env():
    env = MountainCar(state_disc=5)
    check_discrete_env(env)


def test_mountaincar_continuous_env():
    env = MountainCar(state_disc=5, action_mode="continuous")
    check_continuous_env(env)


def test_image_obs():
    env = MountainCar(state_disc=5, obs_mode="image")
    env.reset()
    for i in range(env.horizon - 1):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        assert obs.shape == (1, 28, 28)
