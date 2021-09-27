import distrax
import jax.numpy as jnp

from shinrl import Pendulum


def test_step_reset():
    env = Pendulum()
    env.reset()
    for _ in range(env.config.horizon - 1):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        assert not done
    obs, rew, done, info = env.step(a)
    assert info["TimeLimit.truncated"]


def test_continuous_step_reset():
    config = Pendulum.DefaultConfig(act_mode="continuous")
    env = Pendulum(config)
    env.reset()
    for _ in range(env.config.horizon - 1):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        assert not done
    obs, rew, done, info = env.step(a)
    assert info["TimeLimit.truncated"]


def test_q():
    env = Pendulum()
    pol = jnp.ones((env.mdp.dS, env.mdp.dA)) / env.mdp.dA
    ret = env.calc_return(pol)
    assert ret < -1000
    q = env.calc_optimal_q()
    pol = distrax.Greedy(q).probs
    assert q.max() > -50
    assert env.calc_return(pol) > -50
