"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
import os
import pathlib
import pickle
import shutil
from typing import Any, ClassVar, Dict, Optional, Type, Union

import numpy as np
import structlog
from cpprb import ReplayBuffer
from structlog import BoundLogger

import shinrl as srl
from shinrl._utils.scalars import Scalars

from .config import SolverConfig

PathLike = Union[str, os.PathLike]
DataDict = Dict[str, Any]


def prepare_history_dir(dir_path: PathLike, delete_existing: bool = False) -> None:
    if os.path.exists(dir_path) and delete_existing:
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    srl.add_logfile_handler(os.path.join(dir_path, "log.txt"))


class History:
    """Store all the data of a solver.
    * self.n_step: Number of elapsed steps.
    * self.n_epoch: Number of elapsed epochs.
    * self.scalars:
        Store scalar data as {key: {"x": [step], "y": [val]}}. See srl.Scalars.
    * self.data:
        Store jittable object as {key: jittable object}.
        E.g., network's parameters, optimizer's states, and Q-tables.
    * self.config: Configuration of the solver.
    * self.buffer: Replay buffer.
    """

    DefaultConfig: ClassVar[Type[SolverConfig]] = SolverConfig

    def __init__(self):
        self.init_history()
        self.logger: BoundLogger = structlog.get_logger()

    @property
    def n_step(self) -> int:
        return self.data["n_step"]

    @n_step.setter
    def n_step(self, step: int) -> None:
        self.data["n_step"] = step

    @property
    def n_epoch(self) -> int:
        return self.data["n_epoch"]

    @n_epoch.setter
    def n_epoch(self, epoch: int) -> None:
        self.data["n_epoch"] = epoch

    def init_history(self) -> None:
        self.scalars: Scalars = srl.Scalars()
        self.data: DataDict = {"n_step": 0, "n_epoch": 0}
        self.config: srl.Config = self.DefaultConfig()
        self.buffer: Optional[ReplayBuffer] = None

    def add_scalar(self, key: str, val: float) -> None:
        if key not in self.scalars:
            pass
        elif len(self.scalars[key]["x"]) > 0:
            prev_step = self.scalars[key]["x"][-1]
            if self.n_step - prev_step < self.config.add_interval:
                return None
        self.scalars.add(key, self.n_step, val)

    def set_config(self, config: Optional[SolverConfig] = None) -> None:
        if config is None:
            config = self.DefaultConfig()
        assert isinstance(config, self.DefaultConfig)
        self.config = config
        if self.config.verbose:
            self.logger.info("set_config is called.", config=self.config.asdict())

    def recent_summary(self, step_range: int = 100000) -> Dict[str, Dict[str, float]]:
        return self.scalars.recent_summary(step_range)

    def save(self, dir_path: PathLike, save_buffer: bool = False) -> None:
        """Save all histories to [dir_path]

        Args:
            dir_path (str): directory to save
            save_buffer (bool): save replay buffer if True
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # save step and epoch numbers
        np.savetxt(
            os.path.join(dir_path, "step_epoch.csv"),
            np.array([self.n_step, self.n_epoch]),
            fmt="%d",
            header="step,epoch",
        )

        # save scalars
        self.scalars.dump(os.path.join(dir_path, "scalars"))

        # save config
        self.config.save_as_yaml(os.path.join(dir_path, "config.yaml"))

        # save data
        data_path = os.path.join(dir_path, "data")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        for key, data in self.data.items():
            pkl_path = os.path.join(data_path, key + ".pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(data, f)

        # save replay buffer
        if isinstance(self.buffer, ReplayBuffer) and save_buffer:
            self.buffer.save_transitions(os.path.join(dir_path, "buffer"), safe=True)
        if self.config.verbose:
            self.logger.info("History saved.", dir_path=dir_path)

    def load(self, dir_path: PathLike) -> None:
        # load step and epoch numbers
        step_epoch = np.loadtxt(os.path.join(dir_path, "step_epoch.csv"))
        self.n_step, self.n_epoch = step_epoch[0], step_epoch[1]

        # load scalars
        self.scalars.load(os.path.join(dir_path, "scalars"))

        # load config
        self.config.load_from_yaml(os.path.join(dir_path, "config.yaml"))

        # load data
        data_path = os.path.join(dir_path, "data")
        data_path = pathlib.Path(data_path)
        for pkl_path in data_path.rglob("*.pkl"):
            key = pkl_path.name.split(".pkl")[0]
            with open(str(pkl_path), "rb") as f:
                self.data[key] = pickle.load(f)

        # load replay buffer
        buffer_path = os.path.join(dir_path, "buffer.npz")
        if isinstance(self.buffer, ReplayBuffer) and os.path.exists(buffer_path):
            self.buffer.load_transitions(buffer_path)

        if self.config.verbose:
            self.logger.info("History loaded.", dir_path=dir_path)
