""" JAX functions for epsilon-greedy policy.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def calc_eps(n_step: int, eps_decay: int, eps_warmup: int, eps_end: float) -> float:
    """Calculate the epsilon value for epsilon-greedy policy.
    Args:
        n_step (int): the current step
        eps_decay (int): the period over which epsilon is decayed.
        eps_warmup (int): the number of steps taken before epsilon is decayed.
        eps_end (float): the final value to which to decay the epsilon parameter.

    Returns:
        float: the epsilon value computed according to the schedule.
    """

    steps_left = eps_decay + eps_warmup - n_step
    bonus = (1.0 - eps_end) * steps_left / eps_decay
    bonus = jnp.clip(bonus, 0.0, 1.0 - eps_end)
    return eps_end + bonus
