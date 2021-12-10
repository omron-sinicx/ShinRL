"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from __future__ import annotations

import enum
import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array

from .config import MazeConfig


class MOVE(enum.IntEnum):
    noop = 0
    up = 1
    down = 2
    left = 3
    right = 4


MOVE_TO_DXY = jnp.array(
    [
        [0, 0],
        [0, -1],
        [0, 1],
        [-1, 0],
        [1, 0],
    ]
)


class TILE(enum.IntEnum):
    empty = 0
    rew = 1
    wall = 2
    start = 3


TILE_TO_REW = jnp.array([0.0, 1.0, 0.0, 0.0])
STR_TO_TILE = {"O": TILE.empty, "#": TILE.wall, "S": TILE.start, "R": TILE.rew}


def str_to_maze_array(s: str) -> Array:
    if s.endswith("\\"):
        s = s[:-1]
    rows = s.split("\\")
    rowlens = np.array([len(row) for row in rows])
    assert np.all(rowlens == rowlens[0])
    w, h = len(rows[0]), len(rows)

    array = np.zeros((w, h))
    for i in range(h):
        for j in range(w):
            array[j, i] = STR_TO_TILE[rows[i][j]]
    maze = jnp.array(array)
    return maze


@jax.jit
def out_of_bounds(maze: Array, x: int, y: int):
    """ Return true if x, y is out of bounds """
    w, h = maze.shape
    is_x_out = (x < 0) + (x >= w)
    is_y_out = (y < 0) + (y >= h)
    return is_x_out + is_y_out


@jax.jit
def state_to_xy(maze: Array, state: int) -> Array:
    w = maze.shape[0]
    x, y = state % w, state // w
    return x.astype(jnp.uint32), y.astype(jnp.uint32)


@jax.jit
def xy_to_state(maze: Array, x: int, y: int) -> int:
    w, h = maze.shape[0], maze.shape[1]
    state = x + y * w
    return state.astype(jnp.uint32)


@jax.jit
def reward(maze: Array, state: int, act: int):
    x, y = state_to_xy(maze, state)
    val = maze[x, y].astype(jnp.uint32)
    return TILE_TO_REW[val]


@jax.jit
def transition(config: MazeConfig, maze: Array, state: int, act: int) -> Array:
    w, h = maze.shape[0], maze.shape[1]

    def cannot_enter(xy):
        x, y = xy
        tile = maze[x, y]
        is_out = out_of_bounds(maze, x, y)
        is_wall = tile == 3
        return is_out + is_wall

    def xy_to_state(xy) -> Array:
        x, y = xy
        state = x + y * w
        return state.astype(jnp.uint32)

    cannot_enter = jax.vmap(cannot_enter)
    xy_to_state = jax.vmap(xy_to_state)
    x, y = state_to_xy(maze, state)
    next_xy = jnp.array([x, y]) + MOVE_TO_DXY  # (5, 2)
    zero_flags = cannot_enter(next_xy)  # (5, )
    next_states = jnp.where(zero_flags, state, xy_to_state(next_xy))  # (5, )
    probs = jnp.where(zero_flags, 0.0, 1.0) * config.eps  # (5, )
    probs = probs.at[act].add(1 - probs.sum())
    return next_states, probs


@functools.partial(jax.jit, static_argnums=(1,))
def flat_to_one_hot(val: int, ndim: int) -> Array:
    """
    >>> flat_to_one_hot(2, ndim=4)
    array([ 0.,  0.,  1.,  0.])
    >>> flat_to_one_hot(4, ndim=5)
    array([ 0.,  0.,  0.,  0.,  1.])
    """
    onehot = jnp.zeros((ndim,))
    onehot = onehot.at[val].set(1.0)
    return onehot


@jax.jit
def onehot_observation(maze: Array, state: int) -> Array:
    w, h = maze.shape[0], maze.shape[1]
    x, y = state_to_xy(maze, state)
    x = flat_to_one_hot(x, w)
    y = flat_to_one_hot(y, h)
    obs = jnp.hstack([x, y])
    return obs


def init_probs(maze: Array) -> Tuple[Array, Array]:
    w, h = maze.shape

    def xy_to_state(xy) -> Array:
        x, y = xy
        state = x + y * w
        return state.astype(jnp.uint32)

    start_xy = jnp.array(jnp.where(maze == TILE.start))
    start_states = xy_to_state(start_xy)
    assert len(start_states) > 0, "Start position is not specified."
    probs = jnp.ones_like(start_states) / len(start_states)
    return start_states, probs
