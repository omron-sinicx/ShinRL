"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
import os
import pathlib
import pickle
from typing import Dict, List, Union

import haiku as hk
import optax

from .jittable import DictJittable

PathLike = Union[str, os.PathLike]
PRMS = Union[hk.Params, Union[optax.OptState, List[optax.OptState]]]


class ParamsDict(DictJittable, Dict[str, PRMS]):
    """
    Class to store network's parameters and optimizer's states.
    This class can be treated as a dictionary {key: PRMS}.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def set(self, key: str, param: PRMS) -> None:
        self[key] = param

    def save(self, dir_path: PathLike) -> None:
        """Save each parameters to [dir_path]/[key].pkl

        Args:
            dir_path (PathLike): directory path to save
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for key, param in self.items():
            pkl_path = os.path.join(dir_path, key + ".pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(param, f)

    def load(self, dir_path: PathLike) -> None:
        """Load all parameters from [dir_path]

        Args:
            dir_path (PathLike)
        """
        dir_path = pathlib.Path(dir_path)
        for pkl_path in dir_path.rglob("*.pkl"):
            key = pkl_path.name.split(".pkl")[0]
            with open(str(pkl_path), "rb") as f:
                param = pickle.load(f)
            self.set(key, param)
