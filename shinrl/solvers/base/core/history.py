"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
import os
import shutil
from typing import Callable, ClassVar, Dict, Optional, Type, Union

import numpy as np
import structlog
from chex import Array
from cpprb import ReplayBuffer
from structlog import BoundLogger

from shinrl import ParamsDict, Scalars, TbDict

from .config import SolverConfig

TB = Union[Callable[[], Array], Array]


class History:
    """Store all the data of a solver.
    * self.n_step: Number of elapsed steps.
    * self.n_epoch: Number of elapsed epochs.
    * self.scalars: Store scalar data as {key: {"x": [step], "y": [val]}}. See Scalars.
    * self.tb_dict: Store tabular data as {key: table}. See TbDict.
    * self.prms_dict: Store parameters as {key: params or opt_state}. See ParamsDict.
    * self.config: Configuration of the solver.
    * self.buffer:: Replay buffer.
    """

    DefaultConfig: ClassVar[Type[SolverConfig]] = SolverConfig

    def __init__(self):
        self.init_history()
        self.logger: BoundLogger = structlog.get_logger()

    def init_history(self) -> None:
        self.n_step: int = 0
        self.n_epoch: int = 0
        self.scalars = Scalars()
        self.tb_dict = TbDict()
        self.prms_dict = ParamsDict()
        self.config = self.DefaultConfig()
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
        self.logger.info("set_config is called.", config=self.config.asdict())

    def recent_summary(self, step_range: int = 100000) -> Dict[str, Dict[str, float]]:
        return self.scalars.recent_summary(step_range)

    def save(self, dir_path: str, save_buffer: bool = False) -> None:
        """Save all histories to [dir_path]

        Args:
            dir_path (str): directory to save
            save_buffer (bool): save replay buffer if True
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        np.savetxt(
            os.path.join(dir_path, "step_epoch.csv"),
            np.array([self.n_step, self.n_epoch]),
            fmt="%d",
            header="step,epoch",
        )
        self.scalars.dump(os.path.join(dir_path, "scalars"))
        self.config.save_as_yaml(os.path.join(dir_path, "config.yaml"))
        self.prms_dict.save(os.path.join(dir_path, "params"))
        self.tb_dict.save(os.path.join(dir_path, "tables"))
        if isinstance(self.buffer, ReplayBuffer) and save_buffer:
            self.buffer.save_transitions(os.path.join(dir_path, "buffer"), safe=True)
        self.logger.info("Histories are saved.", dir_path=dir_path)

    def load(self, dir_path: str) -> None:
        step_epoch = np.loadtxt(os.path.join(dir_path, "step_epoch.csv"))
        self.n_step, self.n_epoch = step_epoch[0], step_epoch[1]
        self.scalars.load(os.path.join(dir_path, "scalars"))
        self.config.load_from_yaml(os.path.join(dir_path, "config.yaml"))
        self.tb_dict.load(os.path.join(dir_path, "tables"))
        self.prms_dict.load(os.path.join(dir_path, "params"))
        buffer_path = os.path.join(dir_path, "buffer.npz")
        if isinstance(self.buffer, ReplayBuffer) and os.path.exists(buffer_path):
            self.buffer.load_transitions(buffer_path)
        self.logger.info("Load histories.", dir_path=dir_path)
