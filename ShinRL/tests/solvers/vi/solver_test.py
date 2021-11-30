import gym
import pytest

from shinrl.envs import Pendulum
from shinrl.solvers.vi.discrete import ViSolver


@pytest.mark.parametrize(
    "kl,er,explore,exploit,approx,double_q",
    [
        (0.0, 0.0, "oracle", "greedy", "tabular", False),
        (0.1, 0.0, "oracle", "softmax", "tabular", False),
        (0.0, 0.0, "oracle", "greedy", "nn", False),
        (0.1, 0.1, "oracle", "greedy", "nn", False),
        (0.1, 0.1, "eps_greedy", "greedy", "tabular", False),
        (0.027, 0.003, "eps_greedy", "greedy", "nn", False),
        (0.0, 0.0, "eps_greedy", "greedy", "nn", False),
        (0.0, 0.0, "oracle", "greedy", "nn", True),
        (0.0, 0.0, "eps_greedy", "greedy", "nn", True),
    ],
)
def test_vi_shin(kl, er, explore, exploit, approx, double_q):
    pend_env = Pendulum(state_disc=5, dA=3, horizon=5)
    pend_env.reset()
    config = ViSolver.DefaultConfig(
        kl_coef=kl,
        er_coef=er,
        explore=explore,
        exploit=exploit,
        approx=approx,
        eval_interval=3,
        add_interval=1,
        steps_per_epoch=10,
        use_double_q=double_q,
    )
    solver = ViSolver.factory(config)
    solver.initialize(pend_env, config=config)
    solver.run()
    assert solver.config == config
    assert solver.history.step == 10


@pytest.mark.parametrize(
    "kl,er,explore,exploit,approx",
    [
        (0.0, 0.0, "eps_greedy", "greedy", "nn"),
        (0.1, 0.1, "eps_greedy", "softmax", "nn"),
    ],
)
def test_vi_gym(kl, er, explore, exploit, approx):
    env = gym.make("CartPole-v0")
    config = ViSolver.DefaultConfig(
        kl_coef=kl,
        er_coef=er,
        explore=explore,
        exploit=exploit,
        approx=approx,
        eval_interval=3,
        add_interval=1,
        steps_per_epoch=10,
    )
    solver = ViSolver.factory(config)
    solver.initialize(env, config=config)
    solver.run()
    assert solver.config == config
    assert solver.history.step == 10
