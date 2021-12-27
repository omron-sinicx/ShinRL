"""MixIns to compute new params for nn-based algorithms.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from typing import Optional

import chex
import gym
import haiku as hk
import jax
from chex import Array
from jax import value_and_grad
from optax import apply_updates

import shinrl as srl

from .config import DdpgConfig


class CalcParamsDpMixIn:
    def initialize(self, env: gym.Env, config: Optional[DdpgConfig] = None) -> None:
        super().initialize(env, config)
        self.calc_params = self._build_calc_params()

    def _build_calc_params(self):
        q_loss_fn = getattr(srl, self.config.q_loss_fn.name)

        def calc_pol_loss(pol_prm: hk.Params, q_prm: hk.Params):
            obs = self.env.mdp.obs_mat
            mean = self.pol_net.apply(pol_prm, obs)  # dS x act_shape
            pred = self.q_net.apply(q_prm, obs, mean)
            return -pred.mean()

        def calc_q_loss(q_prm: hk.Params, q_targ: Array):
            obs, act = self._sa_obs_mat, self._sa_act_mat
            pred = self.q_net.apply(q_prm, obs, act)
            pred = pred.reshape(self.dS, self.dA)  # dS x dA
            return q_loss_fn(pred, q_targ)

        def calc_params(data: srl.DataDict) -> Array:
            q_prm, pol_prm = data["QNetParams"], data["PolNetParams"]
            q_opt_st, pol_opt_st = data["QOptState"], data["PolOptState"]

            # Compute new Pol-Net params
            pol_loss, pol_grad = value_and_grad(calc_pol_loss)(pol_prm, q_prm)
            updates, pol_state = self.pol_opt.update(pol_grad, pol_opt_st, pol_prm)
            new_pol_prm = apply_updates(pol_prm, updates)
            pol_res = pol_loss, new_pol_prm, pol_state

            # Compute new Q-Net params
            q_targ = self.target_q_deep_dp(data)  # dS x dA
            q_loss, q_grad = value_and_grad(calc_q_loss)(q_prm, q_targ)
            updates, q_state = self.q_opt.update(q_grad, q_opt_st, q_prm)
            new_q_prm = apply_updates(q_prm, updates)
            q_res = q_loss, new_q_prm, q_state
            return pol_res, q_res

        return jax.jit(calc_params)


class CalcParamsRlMixIn:
    def initialize(self, env: gym.Env, config: Optional[DdpgConfig] = None) -> None:
        super().initialize(env, config)
        self.calc_params = self._build_calc_params()

    def _build_calc_params(self):
        q_loss_fn = getattr(srl, self.config.q_loss_fn.name)

        def calc_pol_loss(pol_prm: hk.Params, q_prm: hk.Params, obs: Array):
            mean = self.pol_net.apply(pol_prm, obs)  # ? x act_shape
            pred = self.q_net.apply(q_prm, obs, mean)  # ? x 1
            return -pred.mean()

        def calc_q_loss(q_prm: hk.Params, q_targ: Array, obs: Array, act: Array):
            pred = self.q_net.apply(q_prm, obs, act)  # ? x 1
            chex.assert_equal_shape((pred, q_targ))
            return q_loss_fn(pred, q_targ)

        def calc_params(data: srl.DataDict, samples: srl.Sample) -> Array:
            obs, act = samples.obs, samples.act
            q_prm, pol_prm = data["QNetParams"], data["PolNetParams"]
            q_opt_st, pol_opt_st = data["QOptState"], data["PolOptState"]

            # Compute new Pol-Net params
            pol_loss, pol_grad = value_and_grad(calc_pol_loss)(pol_prm, q_prm, obs)
            updates, pol_state = self.pol_opt.update(pol_grad, pol_opt_st, pol_prm)
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
