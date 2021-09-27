import chex
import jax
import jax.numpy as jnp

import shinrl as srl

dS = 10
dA = 5
obs_shape = (2,)
init_probs = jnp.array([0.2, 0.8, 0, 0, 0, 0, 0, 0, 0, 0])
discount = 0.99


def tran_fn(state, action):
    next_state = jnp.array([state, (state + action) % 10], dtype=int)
    prob = jnp.array([0.2, 0.8], dtype=float)
    return next_state, prob


def rew_fn(state, action):
    return jnp.array(state + action, dtype=float)


def obs_fn(state):
    return jnp.array([state, state + 5], dtype=float)


def test_mdp():
    obs_mat = srl.MDP.make_obs_mat(obs_fn, dS, obs_shape)
    chex.assert_shape(obs_mat, (dS, *obs_shape))

    tran_mat = srl.MDP.make_tran_mat(tran_fn, dS, dA)
    key = jax.random.PRNGKey(0)
    dmat = jax.random.uniform(key, shape=(dS * dA, 11))
    res = srl.sp_mul(tran_mat, dmat, (dS * dA, dS))
    chex.assert_shape(res, (dS * dA, 11))

    rew_mat = srl.MDP.make_rew_mat(rew_fn, dS, dA)
    chex.assert_shape(rew_mat, (dS, dA))

    mdp = srl.MDP(
        dS=dS,
        dA=dA,
        obs_shape=(2,),
        obs_mat=obs_mat,
        rew_mat=rew_mat,
        tran_mat=tran_mat,
        init_probs=init_probs,
        discount=discount,
    )

    srl.MDP.is_valid_mdp(mdp)
