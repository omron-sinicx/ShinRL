from __future__ import annotations

from typing import Dict

import numpy as np

EMPTY = 110
WALL = 111
START = 112
REWARD = 113
OUT_OF_BOUNDS = 114
LAVA = 118

TILES = {EMPTY, WALL, START, REWARD, LAVA}

STR_MAP = {"O": EMPTY, "#": WALL, "S": START, "R": REWARD, "L": LAVA}

RENDER_DICT = {v: k for k, v in STR_MAP.items()}
RENDER_DICT[EMPTY] = " "
RENDER_DICT[START] = " "


def spec_from_str(s: str, valmap: Dict[str, int] = STR_MAP) -> Spec:
    if s.endswith("\\"):
        s = s[:-1]
    rows = s.split("\\")
    rowlens = np.array([len(row) for row in rows])
    assert np.all(rowlens == rowlens[0])
    w, h = len(rows[0]), len(rows)

    gs = Spec(w, h, string=s)
    for i in range(h):
        for j in range(w):
            gs[j, i] = valmap[rows[i][j]]
    return gs


class Spec(object):
    def __init__(self, w, h, string=""):
        self.__data = np.zeros((w, h), dtype=np.int32)
        self.__w = w
        self.__h = h
        self.string = string

    def __setitem__(self, key, val):
        self.__data[key] = val

    def __getitem__(self, key):
        if self.out_of_bounds(key):
            raise NotImplementedError("Out of bounds:" + str(key))
        return self.__data[tuple(key)]

    def out_of_bounds(self, wh):
        """ Return true if x, y is out of bounds """
        w, h = wh
        if w < 0 or w >= self.__w:
            return True
        if h < 0 or h >= self.__h:
            return True
        return False

    def get_neighbors(self, k, xy=False):
        """ Return values of up, down, left, and right tiles """
        if not xy:
            k = self.idx_to_xy(k)
        offsets = [
            np.array([0, -1]),
            np.array([0, 1]),
            np.array([-1, 0]),
            np.array([1, 0]),
        ]
        neighbors = [
            self[k + offset] if (not self.out_of_bounds(k + offset)) else OUT_OF_BOUNDS
            for offset in offsets
        ]
        return neighbors

    def get_value(self, k, xy=False):
        """ Return values of up, down, left, and right tiles """
        if not xy:
            k = self.idx_to_xy(k)
        return self[k]

    def find(self, value):
        return np.array(np.where(self.spec == value)).T

    @property
    def spec(self):
        return self.__data

    @property
    def width(self):
        return self.__w

    def __len__(self):
        return self.__w * self.__h

    @property
    def height(self):
        return self.__h

    def idx_to_xy(self, idx):
        if hasattr(idx, "__len__"):  # array
            x = idx % self.__w
            y = np.floor(idx / self.__w).astype(np.int32)
            xy = np.c_[x, y]
            return xy
        else:
            return np.array([idx % self.__w, int(np.floor(idx / self.__w))])

    def xy_to_idx(self, key):
        shape = np.array(key).shape
        if len(shape) == 1:
            return key[0] + key[1] * self.__w
        elif len(shape) == 2:
            return key[:, 0] + key[:, 1] * self.__w
        else:
            raise NotImplementedError()

    def __hash__(self):
        data = (self.__w, self.__h) + tuple(self.__data.reshape([-1]).tolist())
        return hash(data)
