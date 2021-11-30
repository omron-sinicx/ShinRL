import random
import sys
from typing import Optional

import numpy as np
from gym.spaces import Box

from shinrl.envs import ShinEnv

from .plotter import plot_SA
from .spec import LAVA, RENDER_DICT, REWARD, START, WALL, Spec, spec_from_str

ACT_NOOP = 0
ACT_UP = 1
ACT_DOWN = 2
ACT_LEFT = 3
ACT_RIGHT = 4
ACT_DICT = {
    ACT_NOOP: [0, 0],
    ACT_UP: [0, -1],
    ACT_LEFT: [-1, 0],
    ACT_RIGHT: [+1, 0],
    ACT_DOWN: [0, +1],
}


class TransitionModel(object):
    def __init__(self, spec, eps=0.2):
        self.ms = spec
        self.eps = eps

    def get_aprobs(self, s, a):
        # TODO: could probably output a matrix over all states...
        legal_moves = self._get_legal_moves(s)
        p = np.zeros(len(ACT_DICT))
        p[legal_moves] = self.eps / (len(legal_moves))
        if a in legal_moves:
            p[a] += 1.0 - self.eps
        else:
            # p = np.array([1.0,0,0,0,0])  # NOOP
            p[ACT_NOOP] += 1.0 - self.eps
        return p

    def _get_legal_moves(self, s):
        xy = np.array(self.ms.idx_to_xy(s))
        moves = [
            move
            for move in ACT_DICT
            if not self.ms.out_of_bounds(xy + ACT_DICT[move])
            and self.ms[xy + ACT_DICT[move]] != WALL
        ]
        return moves


class RewardFunction(object):
    def __init__(self, default=0):
        rew_map = {
            REWARD: 1.0,
            LAVA: -1.0,
        }
        self.default = default
        self.rew_map = rew_map

    def __call__(self, spec, s, a):
        val = spec[spec.idx_to_xy(s)]
        if val in self.rew_map:
            return self.rew_map[val]
        return self.default


def flat_to_one_hot(val, ndim):
    """

    >>> flat_to_one_hot(2, ndim=4)
    array([ 0.,  0.,  1.,  0.])
    >>> flat_to_one_hot(4, ndim=5)
    array([ 0.,  0.,  0.,  0.,  1.])
    >>> flat_to_one_hot(np.array([2, 4, 3]), ndim=5)
    array([[ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.],
           [ 0.,  0.,  0.,  1.,  0.]])
    """
    shape = np.array(val).shape
    v = np.zeros(shape + (ndim,))
    if len(shape) == 1:
        v[np.arange(shape[0]), val] = 1.0
    else:
        v[val] = 1.0
    return v


class Maze(ShinEnv):
    """
    Args:
        spec (shinrl.envs.Maze.spec.Spec)
        maze_size (int, optional): Maze size. Used if spec is None.
        trans_eps (float, optional): Randomness of the transition
        default_rew (int, optional): Default reward
        obs_mode (str, optional): Type of observation. "onehot" or "random".
        obs_dim (int, optional): Dimension of the observation when obs_mode=="random"
    """

    def __init__(
        self,
        spec: Optional[Spec] = None,
        horizon: Optional[int] = 20,
        trans_eps=0.0,
        obs_mode="onehot",
        default_rew=0,
        maze_size: Optional[int] = None,
        obs_dim=5,
    ):
        if spec is None and maze_size is not None:
            maze_str = create_maze_str(maze_size, maze_size)
            self.ms = spec_from_str(maze_str)
        else:
            self.ms = spec
        self.model = TransitionModel(self.ms, eps=trans_eps)
        self.rew_fn = RewardFunction(default=default_rew)
        self.action_mode = "discrete"

        # compute init_state_dist
        start_idxs = np.array(np.where(self.ms.spec == START)).T
        num_starts = start_idxs.shape[0]
        initial_distribution = {}
        for i in range(num_starts):
            initial_distribution[self.ms.xy_to_idx(start_idxs[i])] = 1.0 / num_starts

        super().__init__(
            dS=len(self.ms),
            dA=5,
            init_state_dist=initial_distribution,
            horizon=horizon,
        )

        if obs_mode == "onehot":
            self.observation_space = Box(0, 1, (self.ms.width + self.ms.height,))
        elif obs_mode == "random":
            self.obs_dim = obs_dim
            self.obs_matrix = self.np_random.randn(len(self.ms), self.obs_dim)
            self.observation_space = Box(
                np.min(self.obs_matrix),
                np.max(self.obs_matrix),
                shape=(self.obs_dim,),
                dtype=np.float32,
            )
        else:
            raise ValueError("Invalid obs_mode: {}".format(obs_mode))
        self.obs_mode = obs_mode

    def transition(self, s, a):
        tile_type = self.ms[self.ms.idx_to_xy(s)]
        if tile_type == LAVA:  # Lava gets you stuck
            return {s: 1.0}

        aprobs = self.model.get_aprobs(s, a)
        t_dict = {}
        for sa in range(5):
            if aprobs[sa] > 0:
                next_s = self.ms.idx_to_xy(s) + ACT_DICT[sa]
                next_s_idx = self.ms.xy_to_idx(next_s)
                t_dict[next_s_idx] = t_dict.get(next_s_idx, 0.0) + aprobs[sa]
        return t_dict

    def reward(self, s, a):
        return self.rew_fn(self.ms, s, a)

    def render(self, ostream=sys.stdout):
        state = self.__state
        ostream.write("-" * (self.ms.width + 2) + "\n")
        for h in range(self.ms.height):
            ostream.write("|")
            for w in range(self.ms.width):
                if self.ms.xy_to_idx((w, h)) == state:
                    ostream.write("*")
                else:
                    val = self.ms[w, h]
                    ostream.write(RENDER_DICT[val])
            ostream.write("|\n")
        ostream.write("-" * (self.ms.width + 2) + "\n")

    def observation(self, s):
        if self.obs_mode == "onehot":
            xy = self.ms.idx_to_xy(s)
            x = flat_to_one_hot(xy[0], self.ms.width)
            y = flat_to_one_hot(xy[1], self.ms.height)
            obs = np.hstack([x, y])
            return obs
        else:
            return self.obs_matrix[s]

    def plot_SA(self, values, title=None, ax=None, **kwargs):
        plot_SA(self.ms, values, title=title, ax=ax)


def create_maze_str(width: int, height: int) -> str:
    direction = [(0, -2), (0, 2), (-2, 0), (2, 0)]
    maze = np.ones((height, width))
    maze[0, 0] = 0  # start position

    def fill_maze(updX, updY):
        rnd_array = list(range(4))
        random.shuffle(rnd_array)
        for index in rnd_array:
            # if it reaches out of the maze or visited cell
            if (
                updY + direction[index][1] < 0
                or updY + direction[index][1] > maze.shape[0]
            ):
                continue
            elif (
                updX + direction[index][0] < 0
                or updX + direction[index][0] > maze.shape[1]
            ):
                continue
            elif maze[updY + direction[index][1]][updX + direction[index][0]] == 0:
                continue

            maze[updY + direction[index][1]][updX + direction[index][0]] = 0
            if index == 0:
                maze[updY + direction[index][1] + 1][updX + direction[index][0]] = 0
            elif index == 1:
                maze[updY + direction[index][1] - 1][updX + direction[index][0]] = 0
            elif index == 2:
                maze[updY + direction[index][1]][updX + direction[index][0] + 1] = 0
            elif index == 3:
                maze[updY + direction[index][1]][updX + direction[index][0] - 1] = 0
            fill_maze(updX + direction[index][0], updY + direction[index][1])

    fill_maze(0, 0)
    maze[0, 0] = -1  # start position
    maze[height - 1, width - 1] = 2  # goal position

    string = ""
    for dy_list in maze:
        for item in dy_list:
            if item == -1:
                string += "S"
            elif item == 0:
                string += "O"
            elif item == 1:
                string += random.choice(["#", "L"])
            elif item == 2:
                string += "R"
        string += "\\"
    return string
