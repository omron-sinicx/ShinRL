import gym
import pytest

from shinrl.envs import Pendulum
from shinrl.solvers.pi.discrete import PiSolver


@pytest.mark.parametrize(
    "er_coef,explore,exploit,approx",
    [
        (0.0, "oracle", "greedy", "tabular"),
        (0.1, "oracle", "softmax", "tabular"),
        (0.0, "eps_greedy", "greedy", "tabular"),
        (0.1, "eps_greedy", "greedy", "tabular"),
        (0.0, "oracle", "greedy", "nn"),
        (0.1, "oracle", "greedy", "nn"),
        (0.0, "eps_greedy", "greedy", "nn"),
        (0.1, "eps_greedy", "greedy", "nn"),
    ],
)
def test_pi_shin(er_coef, explore, exploit, approx):
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5)
    pend_env.reset()
    config = PiSolver.DefaultConfig(
        er_coef=er_coef,
        explore=explore,
        exploit=exploit,
        approx=approx,
        eval_interval=3,
        add_interval=1,
        steps_per_epoch=10,
    )
    solver = PiSolver.factory(config)
    solver.initialize(pend_env, config=config)
    solver.run()
    assert solver.config == config
    assert solver.history.step == 10


@pytest.mark.parametrize(
    "er_coef,explore,exploit,approx",
    [
        (0.0, "eps_greedy", "greedy", "nn"),
        (0.1, "eps_greedy", "greedy", "nn"),
    ],
)
def test_pi_gym(er_coef, explore, exploit, approx):
    env = gym.make("CartPole-v0")
    config = PiSolver.DefaultConfig(
        er_coef=er_coef,
        explore=explore,
        exploit=exploit,
        approx=approx,
        eval_interval=3,
        add_interval=1,
        steps_per_epoch=10,
    )
    solver = PiSolver.factory(config)
    solver.initialize(env, config=config)
    solver.run()
    assert solver.config == config
    assert solver.history.step == 10
