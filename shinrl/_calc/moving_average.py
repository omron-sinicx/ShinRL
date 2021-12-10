""" JAX functions to Calculate moving average.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from __future__ import annotations

import jax
from chex import Array
from jax import lax


@jax.jit
def calc_ma(lr: float, idx1: Array, idx2: Array, tb: Array, tb_targ: Array) -> Array:
    """Calculate moving average.
    The semantics of calc_ma are given by:
        def calc_ma(lr, idx1, idx2, tb, tb_targ):
            for s, a, targ in zip(idx1, idx2, tb_targ):
                tb[s, a] = (1 - lr) * tb[s, a] + lr * targ
            return tb
    Args:
        lr (float): Learning rate
        idx1 (Array): (?, ) or (?, 1) array
        idx2 (Array): (?, ) or (?, 1) array
        tb (Array): (?, ?) initial array
        tb_targ (Array): (?, ) or (?, 1) target array

    Returns:
        tb (Array): (?, ) array
    """
    assert len(tb.shape) == 2  # dSxdA
    idx1 = idx1.squeeze(axis=1) if len(idx1) == 2 else idx1
    idx2 = idx2.squeeze(axis=1) if len(idx2) == 2 else idx2
    tb_targ = tb_targ.squeeze(axis=1) if len(tb_targ) == 2 else tb_targ

    def body_fn(i, tb):
        i1, i2, t = idx1[i], idx2[i], tb_targ[i]
        targ = (1 - lr) * tb[i1, i2] + lr * t
        return tb.at[i1, i2].set(targ)

    tb = lax.fori_loop(0, len(idx1), body_fn, tb)
    return tb
