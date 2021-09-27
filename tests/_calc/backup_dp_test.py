import chex
import jax
import jax.numpy as jnp

import shinrl as srl


def tran_fn(state, action):
    next_state = jnp.array([state, (state + action) % 10], dtype=int)
    prob = jnp.array([0.2, 0.8], dtype=float)
    return next_state, prob


def rew_fn(state, action):
    return jnp.array(state + action, dtype=float)


def obs_fn(state):
    return jnp.array([state, state + 5], dtype=float)


dS, dA, obs_shape, discount = 10, 5, (2,), 0.99
obs_mat = srl.MDP.make_obs_mat(obs_fn, dS, obs_shape)
tran_mat = srl.MDP.make_tran_mat(tran_fn, dS, dA)
rew_mat = srl.MDP.make_rew_mat(rew_fn, dS, dA)
init_probs = jnp.array([0.2, 0.8, 0, 0, 0, 0, 0, 0, 0, 0])
horizon = 10

q = jnp.zeros((dS, dA))
pol = jax.random.uniform(jax.random.PRNGKey(0), shape=(dS, dA))
pol /= pol.sum(axis=1, keepdims=True)


def test_expected_backup_dp():
    new_q = srl.expected_backup_dp(q, pol, rew_mat, tran_mat, discount)
    chex.assert_equal_shape((q, new_q))


def test_soft_expected_backup_dp():
    new_q = srl.soft_expected_backup_dp(
        q, pol, jnp.log(pol), rew_mat, tran_mat, discount, 1.0
    )
    chex.assert_equal_shape((q, new_q))


def test_optimal_backup_dp():
    new_q = srl.optimal_backup_dp(q, rew_mat, tran_mat, discount)
    chex.assert_equal_shape((q, new_q))


def test_double_backup_dp():
    new_q = srl.double_backup_dp(q, q, rew_mat, tran_mat, discount)
    chex.assert_equal_shape((q, new_q))


def test_munchausen_backup_dp():
    new_q = srl.munchausen_backup_dp(q, rew_mat, tran_mat, discount, 1.0, 1.0)
    chex.assert_equal_shape((q, new_q))


def test_calc_q():
    new_q = srl.calc_q(q, rew_mat, tran_mat, discount, horizon)
    chex.assert_equal_shape((q, new_q))


def test_calc_return():
    ret = srl.calc_return(q, rew_mat, tran_mat, init_probs, horizon)


def test_calc_optimal_q():
    new_q = srl.calc_optimal_q(rew_mat, tran_mat, discount, horizon)
    chex.assert_equal_shape((q, new_q))


def test_calc_visit():
    ret = srl.calc_visit(pol, rew_mat, tran_mat, init_probs, discount, horizon)
