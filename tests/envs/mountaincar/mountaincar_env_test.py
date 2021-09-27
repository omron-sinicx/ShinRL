import distrax
import jax.numpy as jnp
import pytest

from shinrl import MountainCar


@pytest.fixture
def setUp():
    config = MountainCar.DefaultConfig(dA=5)
    return config


def test_step_reset():
    env = MountainCar()
    env.reset()
    for _ in range(env.config.horizon - 1):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        assert not done
    obs, rew, done, info = env.step(a)
    assert info["TimeLimit.truncated"]


def test_continuous_step_reset():
    config = MountainCar.DefaultConfig(act_mode="continuous")
    env = MountainCar(config)
    env.reset()
    for _ in range(env.config.horizon - 1):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        assert not done
    obs, rew, done, info = env.step(a)
    assert info["TimeLimit.truncated"]


def test_q():
    env = MountainCar()
    pol = jnp.ones((env.mdp.dS, env.mdp.dA)) / env.mdp.dA
    ret = env.calc_return(pol)
    assert ret < -60
    q = env.calc_optimal_q()
    pol = distrax.Greedy(q).probs
    assert q.max() > -10
    assert env.calc_return(pol) > -30
