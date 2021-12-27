"""
Bellman-backup functions for dynamic programming. 
Use oracle transition-matrix to compute the expectation.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
import functools
from typing import Optional

import chex
import jax
import jax.numpy as jnp
from chex import Array

from .sparse import SparseMat, sp_mul, sp_mul_t

# ----- One-step operators -----


@jax.jit
def optimal_backup_dp(
    q: Array, rew_mat: Array, tran_mat: SparseMat, discount: float
) -> Array:
    """Do optimal bellman-backup :math:`r + \gamma P \max_{a} q`.

    Args:
        q (Array): dS x dA q-value table.
        rew_mat (Array): dS x dA reward table.
        tran_mat ((dSxdA) x dS SparseMat): Transition matrix.
        discount (float): Discount factor.

    Returns:
        q (Array): dS x dA q-value table.
    """
    chex.assert_rank([q, rew_mat], 2)
    dS, dA = q.shape
    v = q.max(axis=-1, keepdims=True)  # S x 1
    q = rew_mat + discount * sp_mul(tran_mat, v, (dS * dA, dS)).reshape(dS, dA)
    return q


@jax.jit
def expected_backup_dp(
    q: Array, pol: Array, rew_mat: Array, tran_mat: SparseMat, discount: float
) -> Array:
    """Do expected bellman-backup :math:`r + \gamma P \langle \pi, q\rangle`.

    Args:
        q (Array): dS x dA q-value table.
        pol (Array): dS x dA policy table.
        rew_mat (Array): dS x dA reward table.
        tran_mat ((dSxdA) x dS SparseMat): Transition matrix.
        discount (float): Discount factor.

    Returns:
        q (Array): dS x dA q-value table.
    """
    chex.assert_rank([q, pol, rew_mat], 2)
    dS, dA = q.shape
    v = jnp.sum(pol * q, axis=-1, keepdims=True)  # S x 1
    q = rew_mat + discount * sp_mul(tran_mat, v, (dS * dA, dS)).reshape(dS, dA)
    return q


@jax.jit
def soft_expected_backup_dp(
    q: Array,
    pol: Array,
    log_pol: Array,
    rew_mat: Array,
    tran_mat: SparseMat,
    discount: float,
    er_coef: float,
) -> Array:
    """Do soft expected bellman-backup :math:`r + \gamma P \langle \pi, q - \tau * \log{\pi}\rangle`.

    Args:
        q (Array): dS x dA q-value table.
        pol (Array): dS x dA policy table.
        log_pol (Array): dS x dA log-policy table.
        rew_mat (Array): dS x dA reward table.
        tran_mat ((dSxdA) x dS SparseMat): Transition matrix.
        discount (float): Discount factor.
        er_coef (float): Entropy coefficient.

    Returns:
        q (Array): dS x dA q-value table.
    """
    chex.assert_rank([q, pol, rew_mat], 2)
    dS, dA = q.shape
    v = jnp.sum(pol * (q - er_coef * log_pol), axis=-1, keepdims=True)  # S x 1
    q = rew_mat + discount * sp_mul(tran_mat, v, (dS * dA, dS)).reshape(dS, dA)
    return q


@jax.jit
def double_backup_dp(
    q: Array, double_q: Array, rew_mat: Array, tran_mat: SparseMat, discount: float
) -> Array:
    """Do double bellman-backup
    :math:`r + \gamma P q[a']` where :math:`a' = \argmax{q'}`.
    Paper: https://arxiv.org/abs/1509.06461

    Args:
        q (Array): dS x dA q-value table.
        double_q (Array): dS x dA double q-value table.
        rew_mat (Array): dS x dA reward table.
        tran_mat ((dSxdA) x dS SparseMat): Transition matrix.
        discount (float): Discount factor.

    Returns:
        q (Array): dS x dA q-value table.
    """
    chex.assert_rank([q, rew_mat], 2)
    dS, dA = q.shape
    argmax_a = double_q.argmax(axis=-1)[:, None]  # S x 1
    v = jnp.take_along_axis(q, argmax_a, axis=-1)  # S x 1
    q = rew_mat + discount * sp_mul(tran_mat, v, (dS * dA, dS)).reshape(dS, dA)
    return q


@jax.jit
def munchausen_backup_dp(
    q: Array,
    rew_mat: Array,
    tran_mat: SparseMat,
    discount: float,
    kl_coef: float,
    er_coef: float,
    logpol_clip: Optional[float] = -1e8,
) -> Array:
    """Do munchausen bellman-backup
    Paper: https://arxiv.org/abs/2007.14430

    Args:
        q (Array): dS x dA q-value table.
        rew_mat (Array): dS x dA reward table.
        tran_mat ((dSxdA) x dS SparseMat): Transition matrix.
        discount (float): Discount factor.
        kl_coef (float): KL coefficient.
        er_coef (float): Entropy coefficient.
        logpol_clip (float): Clipping value of log-policy.

    Returns:
        q (Array): dS x dA q-value table.
    """
    chex.assert_rank([q, rew_mat], 2)
    tau = kl_coef + er_coef
    alpha = kl_coef / tau
    dS, dA = q.shape
    log_pol = jax.nn.log_softmax(q / tau, axis=-1)
    munchausen = alpha * jnp.clip(tau * log_pol, a_min=logpol_clip)
    pol = jax.nn.softmax(q / tau, axis=-1)
    v = (pol * (q - tau * log_pol)).sum(axis=-1, keepdims=True)
    v = sp_mul(tran_mat, v, (dS * dA, dS)).reshape(dS, dA)
    q = munchausen + rew_mat + discount * v
    return q


# ----- Finite horizon operators -----


@functools.partial(jax.jit, static_argnames=("horizon",))
def calc_q(
    pol: Array, rew_mat: Array, tran_mat: SparseMat, discount: float, horizon: int
) -> Array:
    """Compute the oracle q table of a policy.

    Args:
        pol (Array): dS x dA policy table.
        rew_mat (Array): dS x dA reward table.
        tran_mat ((dSxdA) x dS SparseMat): Transition matrix.
        discount (float): Discount factor.
        horizon (int): Environment's horizon.

    Returns:
        q (Array): dS x dA q-value table.
    """
    q = jnp.zeros_like(pol, dtype=float)
    body_fun = lambda i, _q: expected_backup_dp(_q, pol, rew_mat, tran_mat, discount)
    q = jax.lax.fori_loop(0, horizon, body_fun, q)
    return q


@functools.partial(jax.jit, static_argnames=("horizon",))
def calc_return(
    pol: Array, rew_mat: Array, tran_mat: SparseMat, init_probs: Array, horizon: int
) -> Array:
    """Compute undiscounted return of a policy.

    Args:
        pol (Array): dS x dA policy table.
        rew_mat (Array): dS x dA reward table.
        tran_mat ((dSxdA) x dS SparseMat): Transition matrix.
        init_probs (dS Array): Probability of initial states.
        horizon (int): Environment's horizon.

    Returns:
        ret (float)
    """
    q = calc_q(pol, rew_mat, tran_mat, 1.0, horizon)
    v = jnp.sum(pol * q, axis=-1)  # S
    ret = jnp.sum(init_probs * v)
    return ret


@functools.partial(jax.jit, static_argnames=("horizon",))
def calc_optimal_q(
    rew_mat: Array, tran_mat: SparseMat, discount: float, horizon: int
) -> Array:
    """Compute the optimal q table.

    Args:
        rew_mat (Array): dS x dA reward table.
        tran_mat ((dSxdA) x dS SparseMat): Transition matrix.
        discount (float): Discount factor.
        horizon (int): Environment's horizon.

    Returns:
        q (Array): dS x dA q-value table.
    """
    q = jnp.zeros_like(rew_mat, dtype=float)
    body_fun = lambda i, _q: optimal_backup_dp(_q, rew_mat, tran_mat, discount)
    q = jax.lax.fori_loop(0, horizon, body_fun, q)
    return q


@functools.partial(jax.jit, static_argnames=("horizon",))
def calc_visit(
    pol: Array,
    rew_mat: Array,
    tran_mat: SparseMat,
    init_probs: Array,
    discount: float,
    horizon: int,
) -> Array:
    """Compute the discounted state and action frequency of a policy :math:`(1-\gamma)\sum^{H}_{t=0}\gamma^t P(s_t=s, a_t=a|\pi)`

    Args:
        pol (Array): dS x dA policy table.
        rew_mat (Array): dS x dA reward table.
        tran_mat ((dSxdA) x dS SparseMat): Transition matrix.
        init_probs (dS Array): Probability of initial states.
        discount (float): Discount factor.
        horizon (int): Environment's horizon.

    Returns:
        visit: TxSxA table
    """
    chex.assert_rank([pol, rew_mat], 2)
    dS, dA = pol.shape
    norm_factor = 0.0

    def body_fun(t, n_v):
        norm_factor, s_visit = n_v
        cur_discount = discount ** t
        norm_factor = norm_factor + cur_discount
        sa_visit = s_visit * pol  # SxA
        s_visit = sp_mul_t(
            sa_visit.reshape(1, dS * dA), tran_mat, (dS * dA, dS)
        ).reshape(dS, 1)
        return (norm_factor, s_visit)

    s_visit = init_probs.reshape(dS, 1)  # Sx1
    norm_factor, s_visit = jax.lax.fori_loop(0, horizon, body_fun, (0.0, s_visit))
    visit = (s_visit * pol) / norm_factor
    return visit  # SxA
