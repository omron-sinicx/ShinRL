import chex
import numpy.testing as npt
import pytest

from shinrl import MountainCar


@pytest.fixture
def setUp():
    config = MountainCar.DefaultConfig(dA=5)
    return config


def test_force_to_act(setUp):
    from shinrl.envs.mountaincar.core.calc import force_to_act

    config = setUp
    act = force_to_act(config, -0.4)

    # jit testing
    config10 = MountainCar.DefaultConfig(dA=50)
    act = force_to_act(config10, -0.4)


def test_act_to_force(setUp):
    from shinrl.envs.mountaincar.core.calc import act_to_force

    config = setUp
    act = act_to_force(config, 2)


def test_state_to_pos_vel(setUp):
    from shinrl.envs.mountaincar.core.calc import state_to_pos_vel

    config = setUp
    pos, vel = state_to_pos_vel(config, 1)


def test_pos_vel_to_state(setUp):
    from shinrl.envs.mountaincar.core.calc import pos_vel_to_state

    config = setUp
    state = pos_vel_to_state(config, -2.938909, -8)


def test_transition(setUp):
    from shinrl.envs.mountaincar.core.calc import transition

    config = setUp
    next_state, probs = transition(config, 1, 2)
    chex.assert_shape(next_state, (1,))
    chex.assert_shape(probs, (1,))


def test_reward(setUp):
    from shinrl.envs.mountaincar.core.calc import reward

    config = setUp
    rew = reward(config, 1, 2)
    npt.assert_allclose(rew, -1.0, rtol=1e-3)


def test_observation():
    from shinrl.envs.mountaincar.core.calc import observation_tuple

    config = MountainCar.DefaultConfig(obs_mode="tuple")
    obs = observation_tuple(config, 1)
    chex.assert_shape(obs, (2,))

    from shinrl.envs.mountaincar.core.calc import observation_image

    config = MountainCar.DefaultConfig(obs_mode="image")
    obs = observation_image(config, 1)
    chex.assert_shape(obs, (28, 28, 1))
