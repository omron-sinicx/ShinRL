"""MixIns to prepare tables, i.e., isinstance(env, ShinEnv) == True.

* TbInitMixIn: Initialize tables for oracle analysis.
Author: Toshinori Kitamura
Affiliation: NAIST
"""
from __future__ import annotations

from typing import Optional

import distrax
import gym
import jax
import jax.numpy as jnp
from chex import Array

import shinrl as srl

from .config import ViConfig


@jax.jit
def logits_to_greedy(logits: Array) -> Array:
    return distrax.Greedy(logits).probs


@jax.jit
def logits_to_eps_greedy(logits: Array, n_step: int, config: ViConfig) -> Array:
    eps = srl.calc_eps(n_step, config.eps_decay, config.eps_warmup, config.eps_end)
    return distrax.EpsilonGreedy(logits, eps).probs


@jax.jit
def logits_to_softmax(logits: Array, config: ViConfig) -> Array:
    return distrax.Softmax(logits, config.max_tmp).probs


class TbInitMixIn:
    """MixIn to prepare Q tables and policy tables."""

    def initialize(self, env: gym.Env, config: Optional[ViConfig] = None) -> None:
        super().initialize(env, config)
        assert self.is_shin_env

        if self.config.approx == config.APPROX.nn:
            # Set function to map Q-network to the Q table.
            # Called when the Q table is accessed.
            pred_all = lambda params: self.q_net.apply(params, self.env.mdp.obs_mat)
            pred_all = jax.jit(pred_all)
            self.tb_dict.set("Q", lambda: pred_all(self.prms_dict["QNet"]))
        else:
            # Set all-zero table if not use NN.
            self.tb_dict.set("Q", jnp.zeros((self.dS, self.dA)))

        # Set functions to map self.tb_dict["Q"] to the policy tables.
        # Called when the policy table is accessed.
        fncs = {
            "oracle": lambda: logits_to_greedy(self.tb_dict["Q"]),
            "greedy": lambda: logits_to_greedy(self.tb_dict["Q"]),
            "eps_greedy": lambda: logits_to_eps_greedy(
                self.tb_dict["Q"],
                self.n_step,
                self.config,
            ),
            "softmax": lambda: logits_to_softmax(self.tb_dict["Q"], self.config),
        }
        self.tb_dict.set("ExplorePolicy", fncs[self.config.explore.name])
        self.tb_dict.set("ExploitPolicy", fncs[self.config.exploit.name])
