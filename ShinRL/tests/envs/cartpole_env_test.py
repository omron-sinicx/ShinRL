import numpy as np
import pytest

from shinrl.envs import CartPole

from .misc import check_continuous_env, check_discrete_env


def test_cartpole_env():
    env = CartPole(x_disc=5, x_dot_disc=5, th_disc=8, th_dot_disc=16, horizon=10)
    check_discrete_env(env)


def test_cartpole_continuous_env():
    env = CartPole(
        x_disc=5,
        x_dot_disc=5,
        th_disc=8,
        th_dot_disc=16,
        horizon=10,
        action_mode="continuous",
    )
    check_continuous_env(env)
