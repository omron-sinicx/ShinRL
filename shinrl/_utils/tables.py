"""
Author: Toshinori Kitamura
Affiliation: NAIST
"""
import os
import pickle
from typing import Callable, Dict, Union

import jax.numpy as jnp
from chex import Array

from .jittable import DictJittable

PathLike = Union[str, os.PathLike]
TB = Union[Callable[[], Array], Array]


class TbDict(DictJittable, Dict[str, TB]):
    """
    Class to store tables.
    This class can be treated as a dictionary {key: table}.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getitem__(self, key: str) -> Array:
        tb = dict.__getitem__(self, key)
        if isinstance(tb, Array):
            return tb
        elif callable(tb):
            return tb()

    def set(self, key: str, tb: TB) -> None:
        """Set an table to self[key].

        Args:
            key (str): key of the table
            tb (TB): Array or a map function.
                Map function is useful to automatically update tables.
        """
        if isinstance(tb, Array):
            self.__dict__[key] = tb.astype(float)
        elif callable(tb):
            self.__dict__[key] = tb
        else:
            raise ValueError("tb needs to be array or callable.")

    def make_frozen(self) -> Dict[str, Array]:
        """make a dict of np.ndarray

        Returns:
            Dict[str, Array]: tables
        """
        return {key: self[key] for key in self.keys()}

    def save(self, pkl_path: PathLike) -> None:
        """save the frozen table dictionary to file_path

        Args:
            pkl_path (PathLike): file to save
        """
        with open(pkl_path, "wb") as f:
            pickle.dump(self.make_frozen(), f)

    def load(self, pkl_path: PathLike) -> None:
        with open(pkl_path, "rb") as f:
            tb_dict = pickle.load(f)
        for key in tb_dict.keys():
            if key in self:
                tb = dict.__getitem__(self, key)
                # set if tb is not a map function
                if isinstance(tb, Array):
                    self.set(key, jnp.array(tb_dict[key]))
            else:
                # when TbDict does not have the key
                self[key] = jnp.array(tb_dict[key])
