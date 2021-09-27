"""MixIns to execute one-step update.
Author: Toshinori Kitamura
Affiliation: NAIST
"""
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
from optax import GradientTransformation

import shinrl as srl

from .config import ViConfig


class TabularDpStepMixIn:
    """ Step mixin for tabular dynamic programming """

    def step(self) -> None:
        # Compute Q-target
        q_targ = self.target_tabular_dp(self.tb_dict)

        # Update Q-table
        self.tb_dict.set("Q", q_targ)


class TabularRlStepMixIn:
    """ Step mixin for tabular reinforcement learning """

    def step(self) -> None:
        # Compute Q-target
        samples = self.collect_samples()
        q_targ = self.target_tabular_rl(self.tb_dict, samples)

        # Update Q-table
        state, act = samples.state, samples.act  # B
        q_targ = srl.calc_ma(self.config.lr, state, act, self.tb_dict["Q"], q_targ)
        self.tb_dict.set("Q", q_targ)


def build_calc_params_dp(
    config: ViConfig,
    q_net: hk.Transformed,
    q_opt: GradientTransformation,
    target_deep_dp_fn: Callable[[srl.ParamsDict], Array],
    mdp: srl.MDP,
):
    """Build a function to compute a new params for q_net.
    Use all observation (dSxdO) and all actions (dSxdA) for loss calculation.

    Args:
        config (ViConfig)
        q_net (hk.Transformed): Q-net forward function.
        q_opt (GradientTransformation): Q-net optimizer.
        target_deep_dp_fn (Callable[[srl.ParamsDict], Array]):
            Target function for deep_dp. See target_mixin.py.
        mdp (srl.MDP)
    """

    loss_fn = getattr(srl, config.loss_fn.name)

    def calc_loss(q_prm: hk.Params, q_targ: Array, obs: Array):
        pred = q_net.apply(q_prm, obs)
        chex.assert_equal_shape((pred, q_targ))
        return loss_fn(pred, q_targ)

    def calc_params(prms_dict: srl.ParamsDict) -> Array:
        q_targ = target_deep_dp_fn(prms_dict)
        q_prm, opt_state = prms_dict["QNet"], prms_dict["QOpt"]
        loss, grad = jax.value_and_grad(calc_loss)(q_prm, q_targ, mdp.obs_mat)
        updates, opt_state = q_opt.update(grad, opt_state, q_prm)
        q_prm = optax.apply_updates(q_prm, updates)
        return loss, q_prm, opt_state

    return jax.jit(calc_params)


class DeepDpStepMixIn:
    """ Step mixin for dynamic programming with NN. """

    def initialize(self, env: gym.Env, config: Optional[ViConfig] = None) -> None:
        super().initialize(env, config)
        self.calc_params = build_calc_params_dp(
            self.config, self.q_net, self.q_opt, self.target_deep_dp, self.env.mdp
        )

    def step(self) -> None:
        # Update Q-Net params
        loss, q_prm, opt_state = self.calc_params(self.prms_dict)
        self.add_scalar("Loss", loss.item())
        self.prms_dict.set("QNet", q_prm)
        self.prms_dict.set("QOpt", opt_state)

        # Update target Q-Net params
        if (self.n_step + 1) % self.config.target_update_interval == 0:
            self.prms_dict.set("TargQNet", deepcopy(self.prms_dict["QNet"]))


def build_calc_params_rl(
    config: ViConfig,
    q_net: hk.Transformed,
    q_opt: GradientTransformation,
    target_deep_rl_fn: Callable[[srl.ParamsDict, srl.Sample], Array],
):
    """Build a function to compute a new params for q_net.
    Use collected samples for loss calculation

    Args:
        config (ViConfig)
        q_net (hk.Transformed): Q-net forward function.
        q_opt (GradientTransformation): Q-net optimizer.
        target_deep_rl_fn (Callable[[srl.ParamsDict, srl.Sample], Array]):
            Target function for deep_rl. See target_mixin.py.
    """

    loss_fn = getattr(srl, config.loss_fn.name)

    def calc_loss(q_prm: hk.Params, targ: Array, obs: Array, act: Array):
        pred = q_net.apply(q_prm, obs)
        pred = jnp.take_along_axis(pred, act, axis=1)  # Bx1
        chex.assert_equal_shape((pred, targ))
        return loss_fn(pred, targ)

    def calc_params(prms_dict: srl.ParamsDict, samples: srl.Sample):
        q_targ = target_deep_rl_fn(prms_dict, samples)
        act, q_prm, opt_state = samples.act, prms_dict["QNet"], prms_dict["QOpt"]
        loss, grad = jax.value_and_grad(calc_loss)(q_prm, q_targ, samples.obs, act)
        updates, opt_state = q_opt.update(grad, opt_state, q_prm)
        q_prm = optax.apply_updates(q_prm, updates)
        return loss, q_prm, opt_state

    return jax.jit(calc_params)


class DeepRlStepMixIn:
    """ Step mixin for deep RL. """

    def initialize(self, env: gym.Env, config: Optional[ViConfig] = None) -> None:
        super().initialize(env, config)
        self.buffer = srl.make_replay_buffer(self.env, self.config.buffer_size)
        self.calc_params = build_calc_params_rl(
            self.config,
            self.q_net,
            self.q_opt,
            self.target_deep_rl,
        )

    def step(self) -> None:
        samples = self.collect_samples(store_to_buffer=True)
        samples = srl.Sample(**self.buffer.sample(self.config.batch_size))

        # Update Q-Net params
        loss, q_prm, opt_state = self.calc_params(self.prms_dict, samples)
        self.add_scalar("Loss", loss.item())
        self.prms_dict.set("QNet", q_prm)
        self.prms_dict.set("QOpt", opt_state)

        # Update target Q-Net
        if (self.n_step + 1) % self.config.target_update_interval == 0:
            self.prms_dict.set("TargQNet", deepcopy(self.prms_dict["QNet"]))
