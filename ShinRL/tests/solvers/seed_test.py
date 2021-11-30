import numpy as np
import pytest
from cpprb import ReplayBuffer

from shinrl.envs import Pendulum
from shinrl.solvers.vi.discrete import ViSolver


@pytest.mark.parametrize(
    "approx,explore", [("tabular", "eps_greedy"), ("nn", "eps_greedy")]
)
def test_seed_shin(approx, explore):
    env = Pendulum(state_disc=32, dA=5, horizon=200)
    config = ViSolver.DefaultConfig(
        seed=0, eval_interval=1, approx=approx, explore=explore, steps_per_epoch=10
    )
    solver1 = ViSolver.factory(config)
    solver1.initialize(env, config)
    solver1.run()

    solver2 = ViSolver.factory(config)
    solver2.initialize(env, config)
    solver2.run()

    config3 = ViSolver.DefaultConfig(
        seed=1, eval_interval=1, approx=approx, explore=explore, steps_per_epoch=10
    )
    solver3 = ViSolver.factory(config3)
    solver3.initialize(env, config3)
    solver3.run()

    val1 = solver1.history.tbs["Q"]
    val2 = solver2.history.tbs["Q"]
    val3 = solver3.history.tbs["Q"]

    assert np.all(val1 == val2)
    assert not np.all(val1 == val3)
