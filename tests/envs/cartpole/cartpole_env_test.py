import distrax
import jax.numpy as jnp
import pytest

from shinrl import CartPole


@pytest.fixture
def setUp():
    config = CartPole.DefaultConfig(dA=5)
    return config


def test_step_reset():
    env = CartPole()
    env.reset()
    for _ in range(env.config.horizon - 1):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        assert not done
    obs, rew, done, info = env.step(a)
    assert info["TimeLimit.truncated"]


def test_continuous_step_reset():
    config = CartPole.DefaultConfig(act_mode="continuous")
    env = CartPole(config)
    env.reset()
    for _ in range(env.config.horizon - 1):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        assert not done
    obs, rew, done, info = env.step(a)
    assert info["TimeLimit.truncated"]


def test_q():
    env = CartPole()
    pol = jnp.ones((env.mdp.dS, env.mdp.dA)) / env.mdp.dA
    ret = env.calc_return(pol)
    assert ret < 30
    q = env.calc_optimal_q()
    pol = distrax.Greedy(q).probs
    assert q.max() > 80
    assert env.calc_return(pol) > 199
