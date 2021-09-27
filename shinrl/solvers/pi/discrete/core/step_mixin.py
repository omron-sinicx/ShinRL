from __future__ import annotations

from copy import deepcopy
from typing import Callable, Optional

import chex
import gym
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from chex import Array
from distrax import Categorical

import shinrl as srl

from .config import PiConfig


class TabularDpStepMixIn:
    def step(self) -> None:
        # Update Policy-table
        pol_dist = self.target_pol_dist(self.tb_dict["Q"])
        self.tb_dict.set("Logits", pol_dist.logits)

        # Update Q-table
        q_targ = self.target_q_tabular_dp(self.tb_dict, pol_dist)
        self.tb_dict.set("Q", q_targ)


class TabularRlStepMixIn:
    def step(self) -> None:
        samples = self.collect_samples()
        # Update Policy-table
        pol_dist = self.target_pol_dist(self.tb_dict["Q"])
        self.tb_dict.set("Logits", pol_dist.logits)

        # Update Q-table
        q_targ = self.target_q_tabular_rl(self.tb_dict, pol_dist, samples)
        state, act = samples.state, samples.act  # B
        q_targ = srl.calc_ma(self.config.q_lr, state, act, self.tb_dict["Q"], q_targ)
        self.tb_dict.set("Q", q_targ)


def build_calc_params_dp(
    config: PiConfig,
    q_net: hk.Transformed,
    q_opt: optax.GradientTransformation,
    target_q_deep_dp: Callable[[srl.ParamsDict, Categorical], Array],
    pol_net: hk.Transformed,
    pol_opt: optax.GradientTransformation,
    target_pol_dist: Callable[[Array], Categorical],
    mdp: srl.MDP,
):
    q_loss_fn = getattr(srl, config.q_loss_fn.name)
    pol_loss_fn = getattr(srl, config.pol_loss_fn.name)

    def calc_q_loss(q_prm: hk.Params, q_targ: Array, obs: Array):
        pred = q_net.apply(q_prm, obs)
        return q_loss_fn(pred, q_targ)

    def calc_pol_loss(pol_prm: hk.Params, targ_logits: Array, obs: Array):
        logits = pol_net.apply(pol_prm, obs)
        return pol_loss_fn(logits, targ_logits)

    def calc_params(prms_dict: srl.ParamsDict) -> Array:
        obs = mdp.obs_mat

        # Compute new Pol-Net params
        q = q_net.apply(prms_dict["QNet"], obs)
        pol_dist = target_pol_dist(q)
        pol_loss, pol_grad = jax.value_and_grad(calc_pol_loss)(
            prms_dict["PolNet"], pol_dist.logits, obs
        )
        updates, pol_state = pol_opt.update(
            pol_grad, prms_dict["PolOpt"], prms_dict["PolNet"]
        )
        pol_prm = optax.apply_updates(prms_dict["PolNet"], updates)
        pol_res = pol_loss, pol_prm, pol_state

        # Compute new Q-Net params
        q_targ = target_q_deep_dp(prms_dict, pol_dist)
        q_loss, q_grad = jax.value_and_grad(calc_q_loss)(prms_dict["QNet"], q_targ, obs)
        updates, q_state = q_opt.update(q_grad, prms_dict["QOpt"], prms_dict["QNet"])
        q_prm = optax.apply_updates(prms_dict["QNet"], updates)
        q_res = q_loss, q_prm, q_state
        return pol_res, q_res

    return jax.jit(calc_params)


class DeepDpStepMixIn:
    def initialize(self, env: gym.Env, config: Optional[PiConfig] = None) -> None:
        super().initialize(env, config)
        self.calc_params = build_calc_params_dp(
            self.config,
            self.q_net,
            self.q_opt,
            self.target_q_deep_dp,
            self.pol_net,
            self.pol_opt,
            self.target_pol_dist,
            self.env.mdp,
        )

    def step(self) -> None:
        # Compute new Pol and Q Net params
        pol_res, q_res = self.calc_params(self.prms_dict)
        pol_loss, pol_prm, pol_state = pol_res
        q_loss, q_prm, q_state = q_res
        self.add_scalar("PolLoss", pol_loss.item())
        self.add_scalar("QLoss", q_loss.item())

        # Update params
        self.prms_dict.set("PolNet", pol_prm)
        self.prms_dict.set("PolOpt", pol_state)
        self.prms_dict.set("QNet", q_prm)
        self.prms_dict.set("QOpt", q_state)

        # Update target Q-Net params
        if (self.n_step + 1) % self.config.target_update_interval == 0:
            self.prms_dict.set("TargQNet", deepcopy(self.prms_dict["QNet"]))


def build_calc_params_rl(
    config: PiConfig,
    q_net: hk.Transformed,
    q_opt: optax.GradientTransformation,
    target_q_deep_rl: Callable[[srl.ParamsDict, Categorical, srl.Sample], Array],
    pol_net: hk.Transformed,
    pol_opt: optax.GradientTransformation,
    target_pol_dist: Callable[[Array], Categorical],
):
    q_loss_fn = getattr(srl, config.q_loss_fn.name)
    pol_loss_fn = getattr(srl, config.pol_loss_fn.name)

    def calc_q_loss(q_prm: hk.Params, q_targ: Array, obs: Array, act: Array):
        pred = q_net.apply(q_prm, obs)
        pred = jnp.take_along_axis(pred, act, axis=1)  # Bx1
        return q_loss_fn(pred, q_targ).mean()

    def calc_pol_loss(pol_prm: hk.Params, targ_logits: Array, obs: Array):
        logits = pol_net.apply(pol_prm, obs)
        return pol_loss_fn(logits, targ_logits)

    def calc_params(prms_dict: srl.ParamsDict, samples: srl.Sample) -> Array:
        # Compute new Pol-Net params
        q = q_net.apply(prms_dict["QNet"], samples.obs)
        pol_loss, pol_grad = jax.value_and_grad(calc_pol_loss)(
            prms_dict["PolNet"], target_pol_dist(q).logits, samples.obs
        )
        updates, pol_state = pol_opt.update(
            pol_grad, prms_dict["PolOpt"], prms_dict["PolNet"]
        )
        pol_prm = optax.apply_updates(prms_dict["PolNet"], updates)
        pol_res = pol_loss, pol_prm, pol_state

        # Compute new Q-Net params
        next_q = q_net.apply(prms_dict["QNet"], samples.next_obs)
        q_targ = target_q_deep_rl(prms_dict, target_pol_dist(next_q), samples)
        q_loss, q_grad = jax.value_and_grad(calc_q_loss)(
            prms_dict["QNet"], q_targ, samples.obs, samples.act
        )
        updates, q_state = q_opt.update(q_grad, prms_dict["QOpt"], prms_dict["QNet"])
        q_prm = optax.apply_updates(prms_dict["QNet"], updates)
        q_res = q_loss, q_prm, q_state
        return pol_res, q_res

    return jax.jit(calc_params)


class DeepRlStepMixIn:
    def initialize(self, env: gym.Env, config: Optional[PiConfig] = None) -> None:
        super().initialize(env, config)
        self.buffer = srl.make_replay_buffer(self.env, self.config.buffer_size)
        self.calc_params = build_calc_params_rl(
            self.config,
            self.q_net,
            self.q_opt,
            self.target_q_deep_rl,
            self.pol_net,
            self.pol_opt,
            self.target_pol_dist,
        )

    def step(self) -> None:
        samples = self.collect_samples(store_to_buffer=True)
        samples = srl.Sample(**self.buffer.sample(self.config.batch_size))

        # Compute new Pol and Q Net params
        pol_res, q_res = self.calc_params(self.prms_dict, samples)
        pol_loss, pol_prm, pol_state = pol_res
        q_loss, q_prm, q_state = q_res
        self.add_scalar("PolLoss", pol_loss.item())
        self.add_scalar("QLoss", q_loss.item())

        # Update params
        self.prms_dict.set("PolNet", pol_prm)
        self.prms_dict.set("PolOpt", pol_state)
        self.prms_dict.set("QNet", q_prm)
        self.prms_dict.set("QOpt", q_state)

        # Update target Q-Net
        if (self.n_step + 1) % self.config.target_update_interval == 0:
            self.prms_dict.set("TargQNet", deepcopy(self.prms_dict["QNet"]))
