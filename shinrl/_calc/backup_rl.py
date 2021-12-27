"""
Bellman-backup functions for reinforcement learning. 
Use collected samples to approximate the expectation.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from typing import Optional

import chex
import jax
import jax.numpy as jnp
from chex import Array


@jax.jit
def optimal_backup_rl(next_q: Array, rew: Array, done: Array, discount: float) -> Array:
    """Do optimal bellman-backup :math:`r + \gamma P \max_{a} q`.

    Args:
        next_q (Array): ? x dA q-values.
        rew (Array): ? x 1 rewards.
        done (Array): ? x 1 done flags.
        discount (float): Discount factor.

    Returns:
        q (Array): ? x 1 q-values.
    """
    chex.assert_rank(next_q, 2)
    next_v = next_q.max(axis=-1, keepdims=True)  # ? x 1
    q = rew + discount * next_v * (~done)
    return q


@jax.jit
def expected_backup_rl(
    next_q: Array, next_pol: Array, rew: Array, done: Array, discount: float
) -> Array:
    """Do expected bellman-backup :math:`r + \gamma P \langle \pi, q\rangle`.

    Args:
        next_q (Array): ? x dA q-values.
        next_pol (Array): ? x dA policy.
        rew (Array): ? x 1 rewards.
        done (Array): ? x 1 done flags.
        discount (float): Discount factor.

    Returns:
        q (Array): ? x 1 q-values.
    """
    chex.assert_rank([next_q, next_pol], 2)
    next_v = (next_pol * next_q).sum(axis=-1, keepdims=True)  # ? x 1
    q = rew + discount * next_v * (~done)
    return q


@jax.jit
def soft_expected_backup_rl(
    next_q: Array,
    next_pol: Array,
    next_log_pol: Array,
    rew: Array,
    done: Array,
    discount: float,
    er_coef: float,
) -> Array:
    """Do soft expected bellman-backup :math:`r + \gamma P \langle \pi, q - \tau * \log{\pi}\rangle`.

    Args:
        next_q (Array): ? x dA q-values.
        next_pol (Array): ? x dA policy.
        next_log_pol (Array): ? x dA log-policy.
        rew (Array): ? x 1 rewards.
        done (Array): ? x 1 done flags.
        discount (float): Discount factor.
        er_coef (float): Entropy coefficient.

    Returns:
        q (Array): ? x 1 q-values.
    """
    chex.assert_rank([next_q, next_pol], 2)
    next_v = next_pol * (next_q - er_coef * next_log_pol)
    next_v = next_v.sum(axis=-1, keepdims=True)  # ? x 1
    q = rew + discount * next_v * (~done)
    return q


@jax.jit
def double_backup_rl(
    next_q: Array, next_double_q: Array, rew: Array, done: Array, discount: float
) -> Array:
    """Do double bellman-backup
    :math:`r + \gamma P q[a']` where :math:`a' = \argmax{q'}`.
    Paper: https://arxiv.org/abs/1509.06461

    Args:
        next_q (Array): ? x dA q-values.
        next_double_q (Array): ? x dA double q-values.
        rew (Array): ? x 1 rewards.
        done (Array): ? x 1 done flags.
        discount (float): Discount factor.

    Returns:
        q (Array): dS x dA q-value table.
    """
    chex.assert_rank([next_q, next_double_q], 2)
    next_argmax_a = next_double_q.argmax(axis=-1)[:, None]
    next_v = jnp.take_along_axis(next_q, next_argmax_a, axis=-1)
    q = rew + discount * next_v * (~done)
    return q


@jax.jit
def munchausen_backup_rl(
    next_q: Array,
    q: Array,
    rew: Array,
    done: Array,
    act: Array,
    discount: float,
    kl_coef: float,
    er_coef: float,
    logpol_clip: Optional[float] = -1e8,
) -> Array:
    """Do munchausen bellman-backup
    Paper: https://arxiv.org/abs/2007.14430

    Args:
        next_q (Array): ? x dA q-values.
        next_double_q (Array): ? x dA double q-values.
        rew (Array): ? x 1 rewards.
        done (Array): ? x 1 done flags.
        act (Array): ? x 1 actions.
        discount (float): Discount factor.
        kl_coef (float): KL coefficient.
        er_coef (float): Entropy coefficient.
        logpol_clip (float): Clipping value of log-policy.

    Returns:
        q (Array): dS x dA q-value table.
    """
    chex.assert_rank([next_q, q], 2)
    tau = kl_coef + er_coef
    alpha = kl_coef / tau
    log_pol = jax.nn.log_softmax(q / tau, axis=-1)
    log_pol = jnp.take_along_axis(log_pol, act, axis=-1)  # (?, 1)
    munchausen = alpha * jnp.clip(tau * log_pol, a_min=logpol_clip)
    next_pol = jax.nn.softmax(next_q / tau, axis=-1)
    next_log_pol = jax.nn.log_softmax(next_q / tau, axis=-1)
    next_v = (next_pol * (next_q - tau * next_log_pol)).sum(axis=-1, keepdims=True)
    q = munchausen + rew + discount * next_v * (~done)
    return q
