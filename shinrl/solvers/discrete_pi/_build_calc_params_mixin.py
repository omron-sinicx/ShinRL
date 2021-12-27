"""MixIns to compute new params for nn-based algorithms.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from typing import Optional

import gym
import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array
from jax import value_and_grad
from optax import apply_updates

import shinrl as srl

from .config import PiConfig


class BuildCalcParamsDpMixIn:
    def initialize(self, env: gym.Env, config: Optional[PiConfig] = None) -> None:
        super().initialize(env, config)
        self.calc_params = self._build_calc_params()

    def _build_calc_params(self):
        q_loss_fn = getattr(srl, self.config.q_loss_fn.name)

        def calc_q_loss(q_prm: hk.Params, q_targ: Array, obs: Array):
            pred = self.q_net.apply(q_prm, obs)
            return q_loss_fn(pred, q_targ)

        def calc_pol_loss(pol_prm: hk.Params, targ_logits: Array, obs: Array):
            logits = self.log_pol_net.apply(pol_prm, obs)
            return self.pol_loss_fn(logits, targ_logits)

        def calc_params(data: srl.DataDict) -> Array:
            obs = self.env.mdp.obs_mat
            q_prm, pol_prm = data["QNetParams"], data["LogPolNetParams"]
            q_opt_st, pol_opt_st = data["QOptState"], data["LogPolOptState"]

            # Compute new Pol-Net params
            q = self.q_net.apply(q_prm, obs)
            logits= self.target_log_pol(q)
            pol_loss, pol_grad = value_and_grad(calc_pol_loss)(pol_prm, logits, obs)
            updates, pol_state = self.log_pol_opt.update(pol_grad, pol_opt_st, pol_prm)
            new_pol_prm = apply_updates(pol_prm, updates)
            pol_res = pol_loss, new_pol_prm, pol_state

            # Compute new Q-Net params
            q_targ = self.target_q_deep_dp(data)
            q_loss, q_grad = value_and_grad(calc_q_loss)(q_prm, q_targ, obs)
            updates, q_state = self.q_opt.update(q_grad, q_opt_st, q_prm)
            new_q_prm = apply_updates(q_prm, updates)
            q_res = q_loss, new_q_prm, q_state
            return pol_res, q_res

        return jax.jit(calc_params)


class BuildCalcParamsRlMixIn:
    def initialize(self, env: gym.Env, config: Optional[PiConfig] = None) -> None:
        super().initialize(env, config)
        self.calc_params = self._build_calc_params()

    def _build_calc_params(self):
        q_loss_fn = getattr(srl, self.config.q_loss_fn.name)

        def calc_q_loss(q_prm: hk.Params, q_targ: Array, obs: Array, act: Array):
            pred = self.q_net.apply(q_prm, obs)
            pred = jnp.take_along_axis(pred, act, axis=1)  # Bx1
            return q_loss_fn(pred, q_targ).mean()

        def calc_pol_loss(pol_prm: hk.Params, targ_logits: Array, obs: Array):
            logits = self.log_pol_net.apply(pol_prm, obs)
            return self.pol_loss_fn(logits, targ_logits)

        def calc_params(data: srl.DataDict, samples: srl.Sample) -> Array:
            obs, act = samples.obs, samples.act
            q_prm, pol_prm = data["QNetParams"], data["LogPolNetParams"]
            q_opt_st, pol_opt_st = data["QOptState"], data["LogPolOptState"]

            # Compute new Pol-Net params
            q = self.q_net.apply(q_prm, obs)
            logits = self.target_log_pol(q)
            pol_loss, pol_grad = value_and_grad(calc_pol_loss)(pol_prm, logits, obs)
            updates, pol_state = self.log_pol_opt.update(pol_grad, pol_opt_st, pol_prm)
            new_pol_prm = apply_updates(pol_prm, updates)
            pol_res = pol_loss, new_pol_prm, pol_state

            # Compute new Q-Net params
            q_targ = self.target_q_deep_rl(data, samples)
            q_loss, q_grad = value_and_grad(calc_q_loss)(q_prm, q_targ, obs, act)
            updates, q_state = self.q_opt.update(q_grad, q_opt_st, q_prm)
            new_q_prm = apply_updates(q_prm, updates)
            q_res = q_loss, new_q_prm, q_state
            return pol_res, q_res

        return jax.jit(calc_params)
