"""MixIns to compute new params for nn-based algorithms.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from typing import Optional

import chex
import gym
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from chex import Array

import shinrl as srl

from .config import ViConfig


class BuildCalcParamsDpMixIn:
    def initialize(self, env: gym.Env, config: Optional[ViConfig] = None) -> None:
        super().initialize(env, config)
        self.calc_params = self._build_calc_params()

    def _build_calc_params(self):
        loss_fn = getattr(srl, self.config.loss_fn.name)

        def calc_loss(q_prm: hk.Params, q_targ: Array, obs: Array):
            pred = self.q_net.apply(q_prm, obs)
            chex.assert_equal_shape((pred, q_targ))
            return loss_fn(pred, q_targ)

        def calc_params(data: srl.DataDict) -> Array:
            q_targ = self.target_deep_dp(data)
            q_prm, opt_state = data["QNetParams"], data["QOptState"]
            mdp = self.env.mdp
            loss, grad = jax.value_and_grad(calc_loss)(q_prm, q_targ, mdp.obs_mat)
            updates, opt_state = self.q_opt.update(grad, opt_state, q_prm)
            q_prm = optax.apply_updates(q_prm, updates)
            return loss, q_prm, opt_state

        return jax.jit(calc_params)


class BuildCalcParamsRlMixIn:
    def initialize(self, env: gym.Env, config: Optional[ViConfig] = None) -> None:
        super().initialize(env, config)
        self.calc_params = self._build_calc_params()

    def _build_calc_params(self):
        loss_fn = getattr(srl, self.config.loss_fn.name)

        def calc_loss(q_prm: hk.Params, targ: Array, obs: Array, act: Array):
            pred = self.q_net.apply(q_prm, obs)
            pred = jnp.take_along_axis(pred, act, axis=1)  # Bx1
            chex.assert_equal_shape((pred, targ))
            return loss_fn(pred, targ)

        def calc_params(data: srl.DataDict, samples: srl.Sample):
            q_targ = self.target_deep_rl(data, samples)
            act, q_prm, opt_state = samples.act, data["QNetParams"], data["QOptState"]
            loss, grad = jax.value_and_grad(calc_loss)(q_prm, q_targ, samples.obs, act)
            updates, opt_state = self.q_opt.update(grad, opt_state, q_prm)
            q_prm = optax.apply_updates(q_prm, updates)
            return loss, q_prm, opt_state

        return jax.jit(calc_params)
