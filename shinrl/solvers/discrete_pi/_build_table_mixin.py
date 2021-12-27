""" MixIn to prepare tables in self.data.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from __future__ import annotations

from typing import Optional

import distrax
import gym
import jax
import jax.numpy as jnp
from chex import Array

import shinrl as srl

from .config import PiConfig


class BuildTableMixIn:
    def initialize(self, env: gym.Env, config: Optional[PiConfig] = None) -> None:
        super().initialize(env, config)

        # initialize tables
        self.data["Q"] = jnp.zeros((self.dS, self.dA))
        self.data["Logits"] = jnp.ones((self.dS, self.dA))
        self.data["ExplorePolicy"] = jnp.ones((self.dS, self.dA)) / self.dA
        self.data["EvaluatePolicy"] = jnp.ones((self.dS, self.dA)) / self.dA

        if self.config.approx == config.APPROX.nn:
            self.pred_all_q, self.pred_all_logits = self._build_pred_all()
        self.update_tb_data()

    def update_tb_data(self):
        # Update Q & Logits values by q_net
        if self.config.approx == self.config.APPROX.nn:
            self.data["Q"] = self.pred_all_q(self.data["QNetParams"])
            self.data["Logits"] = self.pred_all_logits(self.data["PolNetParams"])

        # Update policy tables
        logits, config = self.data["Logits"], self.config
        logits_to_pol = {
            "oracle": lambda: logits_to_greedy(logits),
            "greedy": lambda: logits_to_greedy(logits),
            "eps_greedy": lambda: logits_to_eps_greedy(logits, self.n_step, config),
            "softmax": lambda: logits_to_softmax(logits, config),
        }
        self.data["ExplorePolicy"] = logits_to_pol[config.explore.name]()
        self.data["EvaluatePolicy"] = logits_to_pol[config.exploit.name]()

    def _build_pred_all(self):
        obs = self.env.mdp.obs_mat
        pred_all_q = jax.jit(lambda params: self.q_net.apply(params, obs))
        pred_all_logits = jax.jit(lambda params: self.pol_net.apply(params, obs))
        return pred_all_q, pred_all_logits


@jax.jit
def logits_to_greedy(logits: Array) -> Array:
    return distrax.Greedy(logits).probs


@jax.jit
def logits_to_eps_greedy(logits: Array, n_step: int, config: PiConfig) -> Array:
    eps = srl.calc_eps(n_step, config.eps_decay, config.eps_warmup, config.eps_end)
    return distrax.EpsilonGreedy(logits, eps).probs


@jax.jit
def logits_to_softmax(logits: Array, config: PiConfig) -> Array:
    return distrax.Softmax(logits, config.softmax_tmp).probs
