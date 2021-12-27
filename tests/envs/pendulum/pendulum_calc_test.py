import chex
import numpy.testing as npt
import pytest

from shinrl import Pendulum


@pytest.fixture
def setUp():
    config = Pendulum.DefaultConfig(dA=5)
    return config


def test_to_discrete_act(setUp):
    from shinrl.envs.pendulum.calc import to_discrete_act

    config = setUp
    act = to_discrete_act(config, -0.4)

    # jit testing
    config10 = Pendulum.DefaultConfig(dA=50)
    act = to_discrete_act(config10, -0.4)


def test_to_continuous_act(setUp):
    from shinrl.envs.pendulum.calc import to_continuous_act

    config = setUp
    act = to_continuous_act(config, 2)


def test_state_to_th_vel(setUp):
    from shinrl.envs.pendulum.calc import state_to_th_vel

    config = setUp
    th, vel = state_to_th_vel(config, 1)
    npt.assert_allclose(th, -2.938909)
    npt.assert_allclose(vel, -8)


def test_th_vel_to_state(setUp):
    from shinrl.envs.pendulum.calc import th_vel_to_state

    config = setUp
    state = th_vel_to_state(config, -2.938909, -8)
    assert state == 1


def test_transition(setUp):
    from shinrl.envs.pendulum.calc import transition

    config = setUp
    next_state, probs = transition(config, 1, 2)
    chex.assert_shape(next_state, (1,))
    chex.assert_shape(probs, (1,))


def test_reward(setUp):
    from shinrl.envs.pendulum.calc import reward

    config = setUp
    rew = reward(config, 1, 2)
    npt.assert_allclose(rew, -15.0373, rtol=1e-3)


def test_observation():
    from shinrl.envs.pendulum.calc import observation_tuple

    config = Pendulum.DefaultConfig(obs_mode="tuple")
    obs = observation_tuple(config, 1)
    chex.assert_shape(obs, (3,))

    from shinrl.envs.pendulum.calc import observation_image

    config = Pendulum.DefaultConfig(obs_mode="image")
    obs = observation_image(config, 1)
    chex.assert_shape(obs, (28, 28, 1))
