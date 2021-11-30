import numpy as np
import pytest

from shinrl.envs import Pendulum


@pytest.fixture
def setUp():
    env = Pendulum(state_disc=32, dA=5, horizon=200)
    env.reset()
    policy = np.random.rand(env.dS, env.dA)
    policy /= policy.sum(axis=1, keepdims=True)
    base_policy = np.random.rand(env.dS, env.dA)
    base_policy /= base_policy.sum(axis=1, keepdims=True)
    yield env, policy, base_policy


def test_all_observations():
    env = Pendulum(state_disc=5, dA=3)
    env.reset()
    obss = env.all_observations
    assert obss.shape == (env.dS, env.observation_space.shape[0])


def test_all_actions():
    env = Pendulum(state_disc=5, dA=3, action_mode="continuous")
    env.reset()
    actions = env.all_actions
    assert actions.shape == (env.dA, 1)


def calc_visit(env, policy):
    return env.calc_visit(policy)


def calc_return(env, policy):
    return env.calc_return(policy)


def calc_q(env, policy):
    return env.calc_q(policy)


def calc_optimal_q(env):
    return env.calc_optimal_q()


def calc_er_action_values(env, policy, base_policy):
    return env.calc_q(policy, base_policy=base_policy, er_coef=0.1, kl_coef=0.1)


def test_calc_visit(setUp, benchmark):
    env, policy, _ = setUp
    benchmark.pedantic(calc_visit, kwargs={"env": env, "policy": policy})


def test_calc_return(setUp, benchmark):
    env, policy, _ = setUp
    benchmark.pedantic(calc_return, kwargs={"env": env, "policy": policy})


def test_calc_q(setUp, benchmark):
    env, policy, _ = setUp
    benchmark.pedantic(calc_q, kwargs={"env": env, "policy": policy})


def test_calc_optimal_q(setUp, benchmark):
    env, _, _ = setUp
    benchmark.pedantic(calc_optimal_q, kwargs={"env": env})


def test_er_calc_q(setUp, benchmark):
    env, policy, base_policy = setUp
    benchmark.pedantic(
        calc_er_action_values,
        kwargs={"env": env, "policy": policy, "base_policy": base_policy},
    )
