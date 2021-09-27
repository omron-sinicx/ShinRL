"""MixIns to compute the target value of VI-based algorithms.
Author: Toshinori Kitamura
Affiliation: NAIST
"""
from typing import Optional

import chex
import gym
import jax
import jax.numpy as jnp
from chex import Array

import shinrl as srl

from .config import ViConfig


class TargetMixIn:
    def target_tabular_dp(self, tb_dict: srl.TbDict) -> Array:
        raise NotImplementedError

    def target_tabular_rl(self, tb_dict: srl.TbDict, samples: srl.Sample) -> Array:
        raise NotImplementedError

    def target_deep_dp(self, prms_dict: srl.ParamsDict) -> Array:
        raise NotImplementedError

    def target_deep_rl(self, prms_dict: srl.ParamsDict, samples: srl.Sample) -> Array:
        raise NotImplementedError


class QTargetMixIn(TargetMixIn):
    """MixIn to compute the vanilla Q target."""

    def target_tabular_dp(self, tb_dict: srl.TbDict) -> Array:
        return srl.optimal_backup_dp(
            tb_dict["Q"],
            self.env.mdp.rew_mat,
            self.env.mdp.tran_mat,
            self.config.discount,
        )

    def target_tabular_rl(self, tb_dict: srl.TbDict, samples: srl.Sample) -> Array:
        return srl.optimal_backup_rl(
            tb_dict["Q"][samples.next_state.squeeze(axis=1)],  # BxA
            samples.rew,
            samples.done,
            self.config.discount,
        )

    def target_deep_dp(self, prms_dict: srl.ParamsDict) -> Array:
        return srl.optimal_backup_dp(
            self.q_net.apply(prms_dict["TargQNet"], self.env.mdp.obs_mat),
            self.env.mdp.rew_mat,
            self.env.mdp.tran_mat,
            self.config.discount,
        )

    def target_deep_rl(self, prms_dict: srl.ParamsDict, samples: srl.Sample) -> Array:
        return srl.optimal_backup_rl(
            self.q_net.apply(prms_dict["TargQNet"], samples.next_obs),
            samples.rew,
            samples.done,
            self.config.discount,
        )


class DoubleQTargetMixIn(TargetMixIn):
    """MixIn to compute the Double Q target.
    Paper: https://arxiv.org/abs/1509.06461
    """

    def target_deep_dp(self, prms_dict: srl.ParamsDict) -> Array:
        return srl.double_backup_dp(
            self.q_net.apply(prms_dict["TargQNet"], self.env.mdp.obs_mat),
            self.q_net.apply(prms_dict["QNet"], self.env.mdp.obs_mat),
            self.env.mdp.rew_mat,
            self.env.mdp.tran_mat,
            self.config.discount,
        )

    def target_deep_rl(self, prms_dict: srl.ParamsDict, samples: srl.Sample) -> Array:
        return srl.double_backup_rl(
            self.q_net.apply(prms_dict["TargQNet"], samples.next_obs),
            self.q_net.apply(prms_dict["QNet"], samples.next_obs),
            samples.rew,
            samples.done,
            self.config.discount,
        )


class MunchausenTargetMixIn(TargetMixIn):
    """MixIn to compute the Munchausen Q target.
    Paper: https://arxiv.org/abs/2007.14430
    """

    def target_tabular_dp(self, tb_dict: srl.TbDict) -> Array:
        return srl.munchausen_backup_dp(
            tb_dict["Q"],
            self.env.mdp.rew_mat,
            self.env.mdp.tran_mat,
            self.config.discount,
            self.config.kl_coef,
            self.config.er_coef,
            self.config.logp_clip,
        )

    def target_tabular_rl(self, tb_dict: srl.TbDict, samples: srl.Sample) -> Array:
        return srl.munchausen_backup_rl(
            tb_dict["Q"][samples.next_state.squeeze(axis=1)],  # BxA
            tb_dict["Q"][samples.state.squeeze(axis=1)],  # BxA
            samples.rew,
            samples.done,
            samples.act,
            self.config.discount,
            self.config.kl_coef,
            self.config.er_coef,
            self.config.logp_clip,
        )

    def target_deep_dp(self, prms_dict: srl.ParamsDict) -> Array:
        return srl.munchausen_backup_dp(
            self.q_net.apply(prms_dict["TargQNet"], self.env.mdp.obs_mat),
            self.env.mdp.rew_mat,
            self.env.mdp.tran_mat,
            self.config.discount,
            self.config.kl_coef,
            self.config.er_coef,
            self.config.logp_clip,
        )

    def target_deep_rl(self, prms_dict: srl.ParamsDict, samples: srl.Sample) -> Array:
        return srl.munchausen_backup_rl(
            self.q_net.apply(prms_dict["TargQNet"], samples.next_obs),  # BxA
            self.q_net.apply(prms_dict["TargQNet"], samples.obs),  # BxA
            samples.rew,
            samples.done,
            samples.act,
            self.config.discount,
            self.config.kl_coef,
            self.config.er_coef,
            self.config.logp_clip,
        )
