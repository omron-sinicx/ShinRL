"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp
from chex import Array

from .sparse import SparseMat


class MDP(NamedTuple):
    """Store all the information of a Markov Decision Process (MDP).

    Args:
        dS (int): Number of states.
        dA (int): Number of actions.
        obs_shape (Tuple[int, ...]): Observation shape.
        obs_mat (dS x (obs_shape) Array): Observation of all the states.
        rew_mat (dS x dA Array): Reward matrix.
        tran_mat ((dSxdA) x dS SparseMat): Tranition matrix.
        init_probs (dS Array): Probability of initial states.
        discount (float): Discount factor.
    """

    dS: int
    dA: int
    obs_shape: Tuple[int, ...]
    obs_mat: Array
    rew_mat: Array
    tran_mat: SparseMat
    init_probs: Array
    discount: float

    @staticmethod
    def is_valid_mdp(mdp: MDP) -> bool:
        assert mdp.init_probs.sum() == 1.0
        chex.assert_shape(mdp.init_probs, (mdp.dS,))
        chex.assert_shape(mdp.obs_mat, (mdp.dS, *mdp.obs_shape))
        chex.assert_shape(mdp.rew_mat, (mdp.dS, mdp.dA))
        chex.assert_type(
            [mdp.obs_mat, mdp.rew_mat, mdp.tran_mat.data, mdp.init_probs], float
        )
        assert mdp.tran_mat.shape == (mdp.dS * mdp.dA, mdp.dS)
        assert 0.0 < mdp.discount < 1.0
        return True

    @staticmethod
    def make_obs_mat(
        obs_fn: Callable[[int], Array], dS: int, obs_shape: Tuple[int, ...]
    ) -> Array:
        """Make a observation matrix from obs_fn.

        Args:
            obs_fn (Callable[[int], Array]): Observation function. Taking a state and return its observation.
            dS (int): Number of states.
            obs_shape (Tuple[int, ...]): Observation shape.

        Returns:
            A dS x (obs_shape) array representing observation of all the states.
        """
        obs_fn = jax.vmap(obs_fn)
        all_states = jnp.arange(dS)
        all_obs = obs_fn(all_states)  # dS x obs_dim
        chex.assert_shape(all_obs, [dS, *obs_shape])
        return all_obs

    @staticmethod
    def make_rew_mat(rew_fn: Callable[[int, int], float], dS: int, dA: int) -> Array:
        """Make a reward matrix from rew_fn.

        Args:
            rew_fn (Callable[[int, int], float]): Taking a state and an action, and return its reward.
            dS (int): Number of states.
            dA (int): Number of actions.

        Returns:
            A dS x dA array where the entry reward_matrix[s, a]
            reward given to an agent when transitioning into state ns after taking
            action a from state s.
        """

        def _reward(sa):
            s, a = sa[0], sa[1]
            rew = rew_fn(s, a)
            return rew

        reward = jax.vmap(_reward)
        all_states, all_acts = jnp.arange(dS), jnp.arange(dA)
        SA = jnp.array(jnp.meshgrid(all_states, all_acts)).T.reshape(dS * dA, -1)
        rew_mat = reward(SA).reshape(dS, dA)
        chex.assert_shape(rew_mat, [dS, dA])
        return rew_mat

    @staticmethod
    def make_tran_mat(
        tran_fn: Callable[[int, int], Tuple[Array, Array]], dS: int, dA: int
    ) -> SparseMat:
        """Make a transition matrix from tran_fn.

        Args:
            tran_fn (Callable[[int, int], Tuple[Array, Array]]):
                With a state and an action, returns possible states and their probabilities.
            dS (int): Number of states.
            dA (int): Number of actions.

        Returns:
            A (dS x dA) x dS sparse array where the entry transition_matrix[sa, ns]
            corresponds to the probability of transitioning into state ns after taking
            action a from state s.
        """

        def _transition(sa):
            s, a = sa[0], sa[1]
            next_states, probs = tran_fn(s, a)
            return next_states, probs

        trans = jax.vmap(_transition)
        all_states, all_acts = jnp.arange(dS), jnp.arange(dA)
        SA = jnp.array(jnp.meshgrid(all_states, all_acts)).T.reshape(dS * dA, -1)
        col, data = trans(SA)
        row = jnp.repeat(jnp.arange(dS * dA), col.shape[-1])
        col, row, data = col.reshape(-1), row.reshape(-1), data.reshape(-1)
        return SparseMat(data=data, row=row, col=col, shape=(dS * dA, dS))
