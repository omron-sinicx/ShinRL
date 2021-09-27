"""
Author: Toshinori Kitamura
Affiliation: NAIST
"""
import jax
import jax.numpy as jnp
import numpy.testing as npt

import shinrl as srl


def test_sparse():
    """
    smat = [
        [0, 2, 0],
        [1, 0, 5],
    ]  # 2x3
    dmat = [
        [1],
        [3],
        [5]
    ]  # 3x1
    """
    data = jnp.array([2.0, 1.0, 5.0])
    row = jnp.array([0, 1, 1])
    col = jnp.array([1, 0, 2])
    dmat = jnp.array([1, 3, 5]).reshape(3, 1)
    smat = srl.SparseMat(data=data, row=row, col=col, shape=(2, 3))
    sp_mul = jax.jit(srl.sp_mul, static_argnames=("shape"))
    res = sp_mul(smat, dmat, smat.shape)
    npt.assert_allclose(res, jnp.array([6, 26]).reshape(2, 1))


def test_sparse_t():
    """
    dmat = [
        [1, 3, 5]
        [0, 2, 4]
    ]  # 2x3
    smat = [
        [0, 1],
        [2, 3],
        [0, 5],
    ]  # 3x2
    """
    dmat = jnp.array([[1.0, 3.0, 5.0], [0.0, 2.0, 4.0]])
    data = jnp.array([1.0, 2.0, 3.0, 5.0])
    row = jnp.array([0, 1, 1, 2])
    col = jnp.array([1, 0, 1, 1])
    shape = (3, 2)
    smat = srl.SparseMat(data=data, row=row, col=col, shape=shape)
    sp_mul_t = jax.jit(srl.sp_mul_t, static_argnames=("shape"))
    res = sp_mul_t(dmat, smat, smat.shape)
    npt.assert_allclose(res, jnp.array([[6, 35], [4, 26]]))
