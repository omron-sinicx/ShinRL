import chex
import numpy.testing as npt
import pytest

from shinrl import Pendulum


@pytest.fixture
def setUp():
    config = Pendulum.DefaultConfig(dA=5)
    return config


def test_torque_to_act(setUp):
    from shinrl.envs.pendulum.core.calc import torque_to_act

    config = setUp
    act = torque_to_act(config, -0.4)
    assert act == 2

    # jit testing
    config10 = Pendulum.DefaultConfig(dA=50)
    act = torque_to_act(config10, -0.4)
    assert act == 20


def test_act_to_torque(setUp):
    from shinrl.envs.pendulum.core.calc import act_to_torque

    config = setUp
    act = act_to_torque(config, 2)
    npt.assert_allclose(act, -0.4)


def test_state_to_th_vel(setUp):
    from shinrl.envs.pendulum.core.calc import state_to_th_vel

    config = setUp
    th, vel = state_to_th_vel(config, 1)
    npt.assert_allclose(th, -2.938909)
    npt.assert_allclose(vel, -8)


def test_th_vel_to_state(setUp):
    from shinrl.envs.pendulum.core.calc import th_vel_to_state

    config = setUp
    state = th_vel_to_state(config, -2.938909, -8)
    assert state == 1


def test_transition(setUp):
    from shinrl.envs.pendulum.core.calc import transition

    config = setUp
    next_state, probs = transition(config, 1, 2)
    chex.assert_shape(next_state, (1,))
    chex.assert_shape(probs, (1,))


def test_reward(setUp):
    from shinrl.envs.pendulum.core.calc import reward

    config = setUp
    rew = reward(config, 1, 2)
    npt.assert_allclose(rew, -15.0373, rtol=1e-3)


def test_observation():
    from shinrl.envs.pendulum.core.calc import observation_tuple

    config = Pendulum.DefaultConfig(obs_mode="tuple")
    obs = observation_tuple(config, 1)
    chex.assert_shape(obs, (3,))

    from shinrl.envs.pendulum.core.calc import observation_image

    config = Pendulum.DefaultConfig(obs_mode="image")
    obs = observation_image(config, 1)
    chex.assert_shape(obs, (28, 28, 1))
