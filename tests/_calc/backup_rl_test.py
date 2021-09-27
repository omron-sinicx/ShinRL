import chex
import jax
import jax.numpy as jnp

import shinrl as srl

batch = 10
rew = jax.random.uniform(jax.random.PRNGKey(0), shape=(batch, 1))
done = jnp.zeros((batch, 1), dtype=bool)
act = jax.random.randint(jax.random.PRNGKey(0), (batch, 1), 0, 4, dtype=int)
next_q = jax.random.uniform(jax.random.PRNGKey(0), shape=(batch, 5))
discount = 0.99
next_pol = jax.random.uniform(jax.random.PRNGKey(0), shape=(batch, 5))
next_pol /= next_pol.sum(axis=1, keepdims=True)


def test_expected_backup_rl():
    new_q = srl.expected_backup_rl(next_q, next_pol, rew, done, discount)
    chex.assert_shape(new_q, (batch, 1))


def test_soft_expected_backup_rl():
    new_q = srl.soft_expected_backup_rl(
        next_q, next_pol, jnp.log(next_pol), rew, done, discount, 1.0
    )
    chex.assert_shape(new_q, (batch, 1))


def test_optimal_backup_rl():
    new_q = srl.optimal_backup_rl(next_q, rew, done, discount)
    chex.assert_shape(new_q, (batch, 1))


def test_double_backup_rl():
    new_q = srl.double_backup_rl(next_q, next_q, rew, done, discount)
    chex.assert_shape(new_q, (batch, 1))


def test_muchausen_backup_rl():
    new_q = srl.munchausen_backup_rl(next_q, next_q, rew, done, act, discount, 1.0, 1.0)
    chex.assert_shape(new_q, (batch, 1))
