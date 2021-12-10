""" JAX functions for drawing images on an array. 
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from chex import Array


@functools.partial(jax.jit, static_argnums=(0,))
def line_aa(
    n_pixels: int, c0: int, r0: int, c1: int, r1: int
) -> Tuple[Array, Array, Array]:
    """Generate anti-aliased line pixel coordinates.

    Args:
        n_pixels (int): Number of pixels to fill
        c0, r0 (int): Starting position (column, row)
        c1, r1 (int): End position (column, row)

    Returns:
        cc, rr, val (Array):
            Indices of pixels (cc, rr) and intensity values (val). img[cc, rr] = val.
    """

    flag1 = jnp.abs(c1 - c0) < jnp.abs(r1 - r0)
    r0, c0, r1, c1 = jax.lax.cond(
        flag1, lambda _: (c0, r0, c1, r1), lambda _: (r0, c0, r1, c1), None
    )
    flag2 = c0 > c1
    r0, c0, r1, c1 = jax.lax.cond(
        flag2, lambda _: (r1, c1, r0, c0), lambda _: (r0, c0, r1, c1), None
    )

    x = jnp.linspace(c0, c1, n_pixels)
    y = x * (r1 - r0) / (c1 - c0) + (c1 * r0 - c0 * r1) / (c1 - c0)

    valbot = jnp.floor(y) - y + 1
    valtop = y - jnp.floor(y)

    xx = jnp.concatenate((jnp.floor(y), jnp.floor(y) + 1)).astype(jnp.uint32)
    yy = jnp.concatenate((x, x)).astype(jnp.uint32)
    val = jnp.concatenate((valbot, valtop))

    xx, yy = jax.lax.cond(flag1, lambda _: (yy, xx), lambda _: (xx, yy), None)
    return xx, yy, val


@jax.jit
def draw_circle(image, c0, r0, R):
    """Draw a circle to an image.

    Args:
        image (Array): (?, ?) shaped 2D-array.
        c0, r0 (int): Center position of the circle (column, row)
        R (int): radius of the circle

    Returns:
        image (Array): (?, ?) shaped 2D array
    """
    size_x, size_y = image.shape
    xx, yy = jnp.mgrid[:size_x, :size_y]
    circle = (xx - c0) ** 2 + (yy - r0) ** 2
    image = circle < R ** 2
    return image
