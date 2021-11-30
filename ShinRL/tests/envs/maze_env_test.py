import numpy as np
import pytest

from shinrl.envs.maze import Maze, spec_from_str

from .misc import check_discrete_env


def test_maze_env():
    maze = spec_from_str("SOOO\\" + "OLLL\\" + "OOOO\\" + "OLRO\\")
    env = Maze(maze, trans_eps=0.0)
    check_discrete_env(env)


def test_onehot():
    maze = spec_from_str("SOOO\\" + "OLLL\\" + "OOOO\\" + "OLRO\\")
    env = Maze(maze, trans_eps=0.0, obs_mode="onehot")
    env.reset()
    for i in range(10):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
    assert obs.shape == (8,)


def test_random():
    maze = spec_from_str("SOOO\\" + "OLLL\\" + "OOOO\\" + "OLRO\\")
    obs_dim = 5
    env = Maze(maze, trans_eps=0.0, obs_dim=obs_dim, obs_mode="random")
    env.reset()
    for i in range(10):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
    assert obs.shape == (obs_dim,)
