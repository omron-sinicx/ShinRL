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

from .config import DdpgConfig


class BuildTableMixIn:
    def initialize(self, env: gym.Env, config: Optional[DdpgConfig] = None) -> None:
        super().initialize(env, config)

        # initialize tables
        self.data["Q"] = jnp.zeros((self.dS, self.dA))
        self.data["ExplorePolicy"] = jnp.ones((self.dS, self.dA)) / self.dA
        self.data["EvaluatePolicy"] = jnp.ones((self.dS, self.dA)) / self.dA

        # build pred_all functions
        obs = self.env.mdp.obs_mat  # dS x obs_shape
        self._sa_obs_mat = jnp.repeat(obs, self.dA, axis=0)  # (dSxdA) x obs_shape
        act = self.env.mdp.act_mat  # dA x act_shape
        self._sa_act_mat = jnp.tile(act, (self.dS, 1))  # (dSxdA) x act_shape
        self.pred_all_q, self.pred_all_mean = self._build_pred_all()
        self.update_tb_data()

    def update_tb_data(self):
        # Update Q
        self.data["Q"] = self.pred_all_q(self.data["QNetParams"])  # dS x dA

        # Update policy tables
        act_mat, config = self._sa_act_mat, self.config
        mean = self.pred_all_mean(self.data["PolNetParams"])  # dS x dA x act_shape[0]
        mean_to_pol = {
            "oracle": lambda: to_greedy_probs(mean, act_mat),
            "greedy": lambda: to_greedy_probs(mean, act_mat),
            "normal": lambda: to_normal_probs(mean, config, act_mat),
        }
        self.data["ExplorePolicy"] = mean_to_pol[config.explore.name]()
        self.data["EvaluatePolicy"] = mean_to_pol[config.exploit.name]()

    def _build_pred_all(self):
        dS, dA = self.dS, self.dA
        obs, act = self._sa_obs_mat, self._sa_act_mat
        act_shape = self.env.action_space.shape
        pred_all_mean = jax.jit(
            lambda params: self.pol_net.apply(params, obs).reshape(dS, dA, *act_shape)
        )
        pred_all_q = jax.jit(
            lambda params: self.q_net.apply(params, obs, act).reshape(dS, dA)
        )
        return pred_all_q, pred_all_mean


@jax.jit
def to_greedy_probs(mean: Array, act_mat: Array) -> Array:
    dS, dA = mean.shape[0], mean.shape[1]
    mean = mean.reshape(dS * dA, -1)
    dist = distrax.Normal(mean, 1.0)  # ? x act_shape
    log_prob = dist.log_prob(act_mat).reshape(dS, dA)
    return distrax.Greedy(log_prob).probs


@jax.jit
def to_normal_probs(mean: Array, config: DdpgConfig, act_mat: Array) -> Array:
    dS, dA = mean.shape[0], mean.shape[1]
    mean = mean.reshape(dS * dA, -1)
    dist = distrax.Normal(mean, config.normal_scale)  # ? x act_shape
    log_prob = dist.log_prob(act_mat).reshape(dS, dA)
    return distrax.Softmax(log_prob).probs
