from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod, abstractstaticmethod
from dataclasses import dataclass
from itertools import count
from typing import Iterator, Optional, Type

import gym
import numpy as np
import structlog
import torch
from tqdm import tqdm

from shinrl.envs import ShinEnv
from shinrl.utils import Config, History


@dataclass
class BaseConfig(Config):
    """
    Args:
        seed (int): random seed for the solver
        discount (float): discount factor
        eval_trials (int): number of trials for evaluation
        eval_interval (int): interval to evaluate
        add_interval (int): interval to add a scalar to the history
        steps_per_epoch (int): number of steps per one epoch
    """

    seed: int = 0
    discount: float = 0.99
    eval_trials: int = 10
    eval_interval: int = 100
    add_interval: int = 100
    steps_per_epoch: int = 1000


class BaseSolver(ABC):
    _id: Iterator[int] = count(0)
    DefaultConfig: Type[Config] = BaseConfig

    @abstractstaticmethod
    def factory(config: Config) -> BaseSolver:
        """Create and initialize the solver instance based on env and config."""
        pass

    def __init__(self) -> None:
        self.env_id: int = -1
        self.solver_id: str = f"{type(self).__name__}-{next(self._id)}"
        self.logger: structlog.BoundLogger = structlog.get_logger(
            solver_id=self.solver_id, env_id=None
        )
        self.is_initialized: bool = False
        self._config = None
        self._env = None

    @abstractmethod
    def evaluate(self) -> None:
        """Evaluate the solver. Called every self.config.eval_interval steps."""
        pass

    @abstractmethod
    def step(self) -> None:
        """Execute the solver by one step."""
        pass

    def initialize(
        self,
        env: gym.Env,
        config: Optional[Config] = None,
    ) -> None:
        """Set the env and the config. Initialize the history.
        Args:
            env (gym.Env): Environment to solve.
            config (Config, optional): Configuration for the algorithm, e.g. discount_factor.
        """

        self.set_config(config)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        env.seed(self.config.seed)
        env.action_space.np_random.seed(self.config.seed)
        self.set_env(env)
        self.history = History(add_interval=self.config.add_interval)
        self.logger.info("Solver is initialized.")
        self.is_initialized = True

    @property
    def config(self) -> Config:
        return self._config

    def set_config(self, config: Optional[Config] = None) -> None:
        if config is None:
            self._config = self.DefaultConfig()
        else:
            assert isinstance(config, self.DefaultConfig)
            self._config = config
        self.logger.info("set_config is called.", config=self.config.asdict())

    @property
    def env(self) -> gym.Env:
        return self._env

    @property
    def is_shin_env(self) -> bool:
        return isinstance(self._env, ShinEnv)

    def set_env(self, env: gym.Env, reset: bool = True) -> None:
        """Set the environment to solve.
        Args:
            env (gym.Env): Environment to solve.
            reset (bool): Reset the env if True
        """
        self._env = env
        if self.is_shin_env:
            self.dS, self.dA, self.horizon = env.dS, env.dA, env.horizon

        if reset:
            if isinstance(self.env, gym.wrappers.Monitor):
                # With Monitor, reset() cannot be called unless the episode is over.
                if self.env.stats_recorder.steps is None:
                    self.env.obs = self.env.reset()
                else:
                    done = False
                    while not done:
                        _, _, done, _ = self.env.step(self.env.action_space.sample())
                    self.env.obs = self.env.reset()
            else:
                self.env.obs = self.env.reset()
        else:
            assert hasattr(
                env, "obs"
            ), 'env has no attribute "obs". Do env.obs = obs before calling "set_env".'
        self.env_id += 1
        self.logger = structlog.get_logger(solver_id=self.solver_id, env_id=self.env_id)
        self.logger.info("set_env is called.")

    def run(self) -> None:
        """Run the solver with the step function."""

        assert self.is_initialized, '".initialize" is not called.'
        num_steps = self.config.steps_per_epoch
        for _ in tqdm(range(num_steps), desc=f"Epoch {self.history.epoch}"):
            if self.history.step % self.config.eval_interval == 0:
                self.evaluate()
            self.step()
            self.history.step += 1
        self.history.epoch += 1
        self.logger.info(
            f"Epoch {self.history.epoch} has ended.",
            epoch_summary=self.history.recent_summary(num_steps),
        )

    def save(self, dir_path: str) -> None:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.config.save_as_yaml(os.path.join(dir_path, "config.yaml"))
        self.history.save_all(dir_path)
        self.logger.info(f"Histories are saved to {dir_path}")

    def load(self, dir_path: str, device: str = "cpu") -> None:
        config = self.config.load_from_yaml(os.path.join(dir_path, "config.yaml"))
        self.set_config(config)
        self.history.load_all(dir_path, device)
        self.logger.info(f"Load data from {dir_path}")
