import gym
import numpy as np
import pytest

from shinrl import utils
from shinrl.envs import Pendulum
from shinrl.solvers import BaseSolver


class MockSolver(BaseSolver):
    @staticmethod
    def factory(config=None):
        return MockSolver()

    def step(self):
        i = self.history.step
        self.history.add_scalar("test1", i)
        self.history.add_scalar("test2", i * 10)
        self.history.set_tb("test_array", np.array([1, 2, 3]))

    def evaluate(self):
        pass


def test_solver_id():
    env = gym.make("Pendulum-v0")
    solver1 = MockSolver.factory()
    solver1.initialize(env)
    assert solver1.solver_id == "MockSolver-0"
    assert solver1.env_id == 0
    env = gym.make("CartPole-v0")
    solver1.set_env(env)
    assert solver1.env_id == 1

    solver2 = MockSolver.factory()
    assert solver2.solver_id == "MockSolver-1"


def test_solver():
    env = Pendulum(state_disc=32, dA=5, horizon=200)
    solver = MockSolver.factory()
    config = solver.DefaultConfig(add_interval=5, steps_per_epoch=10)
    solver.initialize(env, config=config)
    solver.run()

    env = gym.make("Pendulum-v0")
    solver = MockSolver.factory()
    solver.initialize(env, config=config)
    solver.run()
    assert solver.history.step == 10
    assert solver.history.scalars["test1"] == {"x": [0, 5], "y": [0, 5]}

    solver.config.steps_per_epoch = 20
    solver.run()
    assert solver.history.step == 30
    assert solver.history.scalars["test2"] == {
        "x": [0, 5, 10, 15, 20, 25],
        "y": [0, 50, 100, 150, 200, 250],
    }


def test_save_load_solver(tmpdir):
    path = tmpdir.mkdir("tmp")

    env = gym.make("Pendulum-v0")
    solver = MockSolver.factory()
    config = solver.DefaultConfig(add_interval=5, steps_per_epoch=10)
    solver.initialize(env, config=config)
    solver.run()
    solver.save(path)
    solver.load(path)
