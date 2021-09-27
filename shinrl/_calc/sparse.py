""" JAX functions for sparse multiplication. 
Author: Toshinori Kitamura
Affiliation: NAIST
"""
from __future__ import annotations

import functools
from typing import NamedTuple, Tuple

import jax
from chex import Array
from jax.ops import segment_sum


class SparseMat(NamedTuple):
    """Object to store a sparse matrix.
    Args:
        data (Array): Data of the sparse matrix.
        row (1-D Array): Rows of the sparse matrix's indices.
        col (1-D Array): Columns of the sparse matrix's indices.
        shape (Tuple[int, int]): Shape of the sparse matrix.
    """

    data: Array
    row: Array
    col: Array
    shape: Tuple[int, int]


@functools.partial(jax.jit, static_argnames=("shape",))
def sp_mul(smat: SparseMat, dmat: Array, shape: Tuple[int, int]) -> Array:
    """Compute smat (NxM) * dmat (MxK).
    must be passed.

    Args:
        smat (SparseMat): NxM sparse matrix.
        dmat (Array): MxK dense matrix.
        shape (Tuple[int, int]): Shape of the sparse matrix (N, M).

    Returns:
        NxK dense matrix
    """
    assert dmat.ndim == 2
    in_ = dmat.take(smat.col, axis=0)
    prod = in_ * smat.data[:, None]
    res = segment_sum(prod, smat.row, shape[0])
    return res


@functools.partial(jax.jit, static_argnames=("shape",))
def sp_mul_t(dmat: Array, smat: SparseMat, shape: Tuple[int, int]) -> Array:
    """Compute dmat (NxM) * smat (MxK).

    Args:
        dmat (Array): NxM dense matrix.
        smat (SparseMat): MxK sparse matrix.
        shape (Tuple[int, int]): Shape of the sparse matrix (M, K).

    Returns:
        NxK dense matrix
    """
    assert dmat.ndim == 2
    in_ = dmat.take(smat.row, axis=1)
    prod = in_ * smat.data[None, :]
    res = segment_sum(prod.T, smat.col, shape[1]).T
    return res
