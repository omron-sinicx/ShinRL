import chex
import jax.numpy as jnp
import numpy.testing as npt


def test_str_to_maze_array():
    from shinrl.envs.maze.core.calc import str_to_maze_array

    str_maze = "SOOO\\" + "O###\\" + "OOOO\\" + "O#RO\\"
    array = str_to_maze_array(str_maze)
    chex.assert_shape(array, (4, 4))


def test_out_of_bounds():
    from shinrl.envs.maze.core.calc import out_of_bounds

    maze = jnp.zeros((4, 4), dtype=int)
    assert out_of_bounds(maze, 5, 7)


def test_state_to_xy():
    from shinrl.envs.maze.core.calc import state_to_xy

    maze = jnp.zeros((4, 4), dtype=int)
    x, y = state_to_xy(maze, 2)
    assert x == 2
    assert y == 0


def test_xy_to_state():
    from shinrl.envs.maze.core.calc import xy_to_state

    maze = jnp.zeros((4, 4), dtype=int)
    state = xy_to_state(maze, 2, 2)
    assert state == 10


def test_reward():
    from shinrl.envs.maze.core.calc import reward

    maze = jnp.zeros((4, 4), dtype=int)
    rew = reward(maze, 1, 0)
    assert rew == 0


def test_transition():
    from shinrl.envs.maze.core.calc import MazeConfig, transition

    maze = jnp.zeros((4, 4), dtype=int)
    config = MazeConfig(eps=0.0)
    ns, probs = transition(config, maze, 0, 0)
    npt.assert_allclose(probs, jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]))

    config = MazeConfig(eps=0.1)
    ns, probs = transition(config, maze, 0, 1)
    npt.assert_allclose(ns, jnp.array([0, 0, 4, 0, 1]))
    npt.assert_allclose(probs, jnp.array([0.1, 0.7, 0.1, 0.0, 0.1]))


def test_onehot_observation():
    from shinrl.envs.maze.core.calc import onehot_observation

    maze = jnp.zeros((4, 4), dtype=int)
    onehot = onehot_observation(maze, 2)
    npt.assert_allclose(onehot, jnp.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]))


def test_init_probs():
    from shinrl.envs.maze.core.calc import init_probs

    maze = jnp.zeros((4, 4), dtype=int)
    maze = maze.at[1, 0].set(3)
    maze = maze.at[0, 1].set(3)
    init_states, probs = init_probs(maze)
    npt.assert_allclose(init_states, jnp.array([4, 1]))
    npt.assert_allclose(probs, jnp.array([0.5, 0.5]))
