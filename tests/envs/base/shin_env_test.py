import chex
import gym
import jax.numpy as jnp

from shinrl import EnvConfig, ShinEnv


@chex.dataclass
class MockConfig(EnvConfig):
    dS: int = 10
    dA: int = 5
    discount: float = 0.99
    horizon: int = 10


class MockEnv(ShinEnv):
    def __init__(self, config):
        super().__init__(config)

    @property
    def dS(self) -> int:
        return self.config.dS

    @property
    def dA(self) -> int:
        return self.config.dA

    @property
    def observation_space(self):
        return gym.spaces.Box(low=jnp.array([0, 0]), high=jnp.array([15, 15]))

    @property
    def action_space(self):
        return gym.spaces.Discrete(5)

    def _init_probs(self):
        return jnp.array([0.2, 0.8, 0, 0, 0, 0, 0, 0, 0, 0])

    def _make_transition_fn(self):
        def transition(state, action):
            next_state = jnp.array([state, (state + action) % 10], dtype=int)
            prob = jnp.array([0.2, 0.8], dtype=float)
            return next_state, prob

        return transition

    def _make_reward_fn(self):
        return lambda state, action: jnp.array(state + action, dtype=float)

    def _make_observation_fn(self):
        return lambda state: jnp.array([state, state + 5], dtype=float)


def test_reset_step():
    config = MockConfig()
    env = MockEnv(config)
    env.reset()
    for _ in range(config.horizon - 1):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        assert not done
    obs, rew, done, info = env.step(a)
    assert info["TimeLimit.truncated"]
