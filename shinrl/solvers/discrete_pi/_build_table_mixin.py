""" MixIn to prepare tables in self.data.
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

from .config import PiConfig


class BuildTableMixIn:
    def initialize(self, env: gym.Env, config: Optional[PiConfig] = None) -> None:
        super().initialize(env, config)

        # initialize tables
        self.data["Q"] = jnp.zeros((self.dS, self.dA))
        self.data["LogPolicy"] = jnp.ones((self.dS, self.dA))
        self.data["ExplorePolicy"] = jnp.ones((self.dS, self.dA)) / self.dA
        self.data["EvaluatePolicy"] = jnp.ones((self.dS, self.dA)) / self.dA

        if self.config.approx == config.APPROX.nn:
            self.pred_all_q, self.pred_all_logits = self._build_pred_all()
        self.update_tb_data()

    def update_tb_data(self):
        # Update Q & Logits values
        if self.config.approx == self.config.APPROX.nn:
            self.data["Q"] = self.pred_all_q(self.data["QNetParams"])
            self.data["LogPolicy"] = self.pred_all_logits(self.data["LogPolNetParams"])

        # Update policy tables
        log_pol, config = self.data["LogPolicy"], self.config
        to_policy = {
            "oracle": lambda: to_greedy(log_pol),
            "greedy": lambda: to_greedy(log_pol),
            "eps_greedy": lambda: to_eps_greedy(log_pol, self.n_step, config),
            "identity": lambda: log_pol,
        }
        self.data["ExplorePolicy"] = to_policy[config.explore.name]()
        self.data["EvaluatePolicy"] = to_policy[config.exploit.name]()

    def _build_pred_all(self):
        obs = self.env.mdp.obs_mat
        pred_all_q = jax.jit(lambda params: self.q_net.apply(params, obs))

        @jax.jit
        def pred_all_policy(params):
            log_policy = self.log_pol_net.apply(params, obs)
            policy = jax.nn.softmax(log_policy, axis=-1)
            return policy

        return pred_all_q, pred_all_policy


@jax.jit
def to_greedy(logits: Array) -> Array:
    return distrax.Greedy(logits).probs


@jax.jit
def to_eps_greedy(logits: Array, n_step: int, config: PiConfig) -> Array:
    eps = srl.calc_eps(n_step, config.eps_decay, config.eps_warmup, config.eps_end)
    return distrax.EpsilonGreedy(logits, eps).probs


@jax.jit
def to_identity(logits: Array) -> Array:
    return jax.nn.softmax(logits, axis=-1)
