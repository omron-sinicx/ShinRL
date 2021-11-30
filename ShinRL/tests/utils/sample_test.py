import gym
import numpy as np
import pytest
import torch
from scipy import special
from torch import nn

from shinrl import utils
from shinrl.envs import Pendulum


def test_samples_dsc_act():
    samples = utils.Samples(
        obs=np.random.rand(3, 2, 1),
        next_obs=np.random.rand(3, 2, 1),
        rew=np.random.rand(3),
        done=np.array([1, 1, 0]),
        log_prob=np.random.rand(3),
        act=np.array([1, 2, 3]),
        timeout=np.array([1, 0, 1]),
    )

    tnsr_samples = samples.np_to_tnsr()
    np_samples = tnsr_samples.tnsr_to_np()

    assert tnsr_samples.obs.shape == (3, 2)
    assert tnsr_samples.obs.dtype == torch.float32

    assert np.all(np_samples.obs == samples.obs)
    assert np_samples.obs.shape == (3, 2)
    assert np_samples.obs.dtype == np.float32


def test_samples_cont_act():
    samples = utils.Samples(
        obs=np.random.rand(3, 2, 1),
        next_obs=np.random.rand(3, 2, 1),
        rew=np.random.rand(3),
        done=np.array([1, 1, 0]),
        log_prob=np.random.rand(3),
        act=np.random.rand(3, 2),
        timeout=np.array([1, 0, 1]),
    )

    tnsr_samples = samples.np_to_tnsr()
    np_samples = tnsr_samples.tnsr_to_np()

    assert tnsr_samples.act.dtype == torch.float32
    assert np.all(np_samples.act == samples.act)
    assert np_samples.act.dtype == np.float32


def test_samples_iter():
    samples = utils.Samples(
        obs=np.random.rand(3, 2, 1),
        next_obs=np.random.rand(3, 2, 1),
        rew=np.random.rand(3),
        done=np.array([1, 1, 0]),
        log_prob=np.random.rand(3),
        act=np.array([0, 1, 2]),
        timeout=np.array([1, 0, 1]),
    )

    for i, sample in enumerate(samples):
        assert sample.act == i


# ----- tabular -----


@pytest.fixture
def setUpTb():
    env = Pendulum(state_disc=5, dA=3)
    env.obs = env.reset()
    policy = np.random.rand(env.dS, env.dA)
    policy = policy / policy.sum(1, keepdims=True)

    cont_env = Pendulum(state_disc=5, dA=3, action_mode="continuous")
    cont_env.obs = cont_env.reset()
    cont_policy = np.random.rand(cont_env.dS, cont_env.dA)
    cont_policy = cont_policy / cont_policy.sum(1, keepdims=True)
    yield env, policy, cont_env, cont_policy


def test_collect_samples(setUpTb):
    env, policy, cont_env, cont_policy = setUpTb

    # discrete
    buf = utils.make_replay_buffer(env, 100)
    samples = utils.collect_samples(
        env, utils.get_tb_act, 10, buffer=buf, get_act_args={"policy": policy}
    )
    assert len(samples.obs) == 10
    assert samples.act.dtype == np.int32
    assert buf.get_stored_size() == 10

    # continuous
    buf = utils.make_replay_buffer(cont_env, 100)
    samples = utils.collect_samples(
        cont_env, utils.get_tb_act, 10, buffer=buf, get_act_args={"policy": cont_policy}
    )
    assert samples.obs.dtype == np.float32
    assert buf.get_stored_size() == 10


def test_collect_samples_episodic(setUpTb):
    env, policy, cont_env, cont_policy = setUpTb

    # discrete
    samples = utils.collect_samples(
        env, utils.get_tb_act, 10, num_episodes=5, get_act_args={"policy": policy}
    )
    assert np.sum(samples.done) == 5
    assert samples.act.dtype == np.int32

    # continuous
    samples = utils.collect_samples(
        cont_env,
        utils.get_tb_act,
        10,
        num_episodes=5,
        get_act_args={"policy": cont_policy},
    )
    assert np.sum(samples.done) == 5
    assert samples.obs.dtype == np.float32


# ----- gym -----


def fc_net(env):
    modules = []
    obs_shape = env.observation_space.shape[0]
    n_acts = env.action_space.n
    modules.append(nn.Linear(obs_shape, 256))
    modules += [nn.ReLU(), nn.Linear(256, n_acts)]
    return nn.Sequential(*modules)


def mock_policy(value):
    policy = special.softmax(value, axis=-1).astype(np.float64)
    policy /= policy.sum(axis=-1, keepdims=True)
    return policy.astype(np.float32)


@pytest.fixture
def setUpGym():
    env = gym.make("CartPole-v0")
    env.obs = env.reset()
    net = fc_net(env)
    yield env, net


def get_action(env, net):
    if not hasattr(env, "obs"):
        env.obs = env.reset()
    obs = torch.as_tensor(env.obs, dtype=torch.float32).unsqueeze(0)
    probs = net(obs).detach().cpu().numpy()  # 1 x dA
    probs = mock_policy(probs).reshape(-1)
    log_probs = np.log(probs)
    action = np.random.choice(np.arange(0, env.action_space.n), p=probs)
    log_prob = log_probs[action]
    return action, log_prob


def test_collect_samples(setUpGym):
    env, net = setUpGym
    samples = utils.collect_samples(env, get_action, 10, get_act_args={"net": net})
    assert len(samples.obs) == 10
    assert samples.act.dtype == np.int32


def test_collect_samples_episodic(setUpGym):
    env, net = setUpGym
    buf = utils.make_replay_buffer(env, 100)
    samples = utils.collect_samples(
        env, get_action, 10, num_episodes=5, get_act_args={"net": net}, buffer=buf
    )
    assert np.sum(samples.done) == 5
    assert samples.act.dtype == np.int32
