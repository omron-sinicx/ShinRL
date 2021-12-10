"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
import os
import pathlib
from typing import Dict, List, TypedDict, Union

import numpy as np

PathLike = Union[str, os.PathLike]


class XY(TypedDict):  # type: ignore
    x: List[int]
    y: List[float]


class Scalars(Dict[str, XY]):
    """
    Class to store scalars.
    This class can be treated as a nested dictionary {key: {"x": [step], "y": [val]}}.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def add(self, key: str, x: int, y: float) -> None:
        """Add a scalar to self[key].

        Args:
            key (str): key of the scalar
            y (float): the scalar to be added.
            x (int): the # steps of the scalar.
        """
        assert np.isscalar(y)
        if key not in self:
            xy: XY = {"x": [], "y": []}
            self[key] = xy
        self[key]["x"].append(x)
        self[key]["y"].append(y)

    def dump_xy(self, key: str, csv_path: PathLike) -> None:
        """
        Append self[key] to a csv_file.
        Delete stored scalars after dump.

        Args:
            csv_path (PathLike): csv file path to save
        """
        xy = np.array([self[key]["x"], self[key]["y"]]).T
        with open(csv_path, "a") as f:
            np.savetxt(f, xy, fmt="%f", header="x,y", delimiter=",")
        del self[key]

    def load_xy(self, key: str, csv_path: PathLike) -> None:
        """Load scalar data from a csv file and store it to self[key]

        Args:
            key (str)
            csv_path (str): csv file path to load data.
        """
        scalars = np.loadtxt(csv_path, delimiter=",").T
        x, y = scalars[0].tolist(), scalars[1].tolist()
        self[key] = {"x": x, "y": y}

    def dump(self, dir_path: PathLike) -> None:
        """
        Append each scalars to [dir_path]/[key].csv.
        Delete all scalars after dump.

        Args:
            dir_path (PathLike): directory path to save
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        keys = list(self.keys())
        for key in keys:
            csv_path = os.path.join(dir_path, key + ".csv")
            self.dump_xy(key, csv_path)

    def load(self, dir_path: PathLike) -> None:
        """Load all scalars data from a directory: [dir_path]/~.csv, ~.csv...

        Args:
            dir_path (PathLike): directory path to load data.
        """
        dir_path = pathlib.Path(dir_path)
        for file in dir_path.rglob("*.csv"):
            key = file.name.split(".csv")[0]
            self.load_xy(key, str(file))

    def recent_summary(self, step_range: int = 100000) -> Dict[str, Dict[str, float]]:
        """Return the summary of recent [step_range] steps' history.

        Args:
            step_range (int, optional): Recent step size to summarize. Defaults to 100000.

        Returns:
            Dict[str, Dict[str, float]]: Summary of the recent history.
        """
        scalars = {}
        for key in self.keys():
            steps = np.array(self[key]["x"])
            vals = np.array(self[key]["y"])
            if len(vals.shape) == 1 and len(vals) > 0:
                idx = (steps - steps[-1]) >= -step_range
                _avg, _max, _min = vals[idx].mean(), vals[idx].max(), vals[idx].min()
                scalars[key] = {"Average": _avg, "Max": _max, "Min": _min}
        return scalars
