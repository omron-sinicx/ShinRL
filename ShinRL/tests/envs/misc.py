import numpy as np


def check_discrete_env(env):
    env.reset()
    for i in range(env.horizon - 1):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        assert not done
    obs, rew, done, info = env.step(a)
    assert done

    # check transition_matrix
    trans_matrix = env.transition_matrix
    assert trans_matrix.shape == (env.dS * env.dA, env.dS)
    t = trans_matrix.sum(axis=1)
    assert np.all(t == 1.0)

    # check reward matrix
    assert env.reward_matrix.shape == (env.dS, env.dA)


def check_continuous_env(env):
    env.reset()
    for i in range(env.horizon - 1):
        a = env.action_space.sample()
        a = env.discretize_action(a)
        obs, rew, done, info = env.step(a)
        assert not done
    obs, rew, done, info = env.step(a)
    assert done

    # check transition_matrix
    trans_matrix = env.transition_matrix
    assert trans_matrix.shape == (env.dS * env.dA, env.dS)
    t = trans_matrix.sum(axis=1)
    assert np.all(t == 1.0)

    # check reward matrix
    assert env.reward_matrix.shape == (env.dS, env.dA)
