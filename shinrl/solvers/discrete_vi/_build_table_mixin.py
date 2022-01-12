"""MixIn to initialize & update tables in self.data
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""

from typing import Optional

import distrax
import gym
import jax
import jax.numpy as jnp
from chex import Array

import shinrl as srl

from .config import ViConfig


class BuildTableMixIn:
    def initialize(self, env: gym.Env, config: Optional[ViConfig] = None) -> None:
        super().initialize(env, config)

        # initialize tables
        self.data["Q"] = jnp.zeros((self.dS, self.dA))
        self.data["ExplorePolicy"] = jnp.ones((self.dS, self.dA)) / self.dA
        self.data["EvaluatePolicy"] = jnp.ones((self.dS, self.dA)) / self.dA

        if self.config.approx == self.config.APPROX.nn:
            self.pred_all = self._build_pred_all()
        self.update_tb_data()

    def update_tb_data(self):
        # Update Q values by q_net
        if self.config.approx == self.config.APPROX.nn:
            self.data["Q"] = self.pred_all(self.data["QNetParams"])

        # Update policy tables
        q, config = self.data["Q"], self.config
        q_to_pol = {
            "oracle": lambda: logits_to_greedy(q),
            "greedy": lambda: logits_to_greedy(q),
            "eps_greedy": lambda: logits_to_eps_greedy(q, self.n_step, config),
            "softmax": lambda: logits_to_softmax(q, config),
        }
        self.data["ExplorePolicy"] = q_to_pol[config.explore.name]()
        self.data["EvaluatePolicy"] = q_to_pol[config.evaluate.name]()

    def _build_pred_all(self):
        pred_all = lambda params: self.q_net.apply(params, self.env.mdp.obs_mat)
        return jax.jit(pred_all)


@jax.jit
def logits_to_greedy(logits: Array) -> Array:
    return distrax.Greedy(logits).probs


@jax.jit
def logits_to_eps_greedy(logits: Array, n_step: int, config: ViConfig) -> Array:
    eps = srl.calc_eps(n_step, config.eps_decay, config.eps_warmup, config.eps_end)
    return distrax.EpsilonGreedy(logits, eps).probs


@jax.jit
def logits_to_softmax(logits: Array, config: ViConfig) -> Array:
    return distrax.Softmax(logits, config.softmax_tmp).probs
