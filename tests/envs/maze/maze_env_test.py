import distrax
import jax.numpy as jnp
import pytest

from shinrl import Maze


@pytest.fixture
def setUp():
    maze_str = Maze.create_random_maze_str(3, 3)
    maze = Maze.str_to_maze_array(maze_str)
    config = Maze.DefaultConfig(eps=0.0)
    return maze, config


def test_step_reset(setUp):
    maze, config = setUp
    env = Maze(maze, config)
    env.reset()
    for _ in range(env.config.horizon - 1):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        assert not done
    obs, rew, done, info = env.step(a)
    assert info["TimeLimit.truncated"]


def test_q(setUp):
    maze, config = setUp
    env = Maze(maze, config)
    pol = jnp.ones((env.mdp.dS, env.mdp.dA)) / env.mdp.dA
    ret = env.calc_return(pol)
    assert ret < 3
    q = env.calc_optimal_q()
    pol = distrax.Greedy(q).probs
    assert q.max() > 10
    assert env.calc_return(pol) > 10
