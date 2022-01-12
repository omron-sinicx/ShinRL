"""MixIns to compute the target value of PI-based algorithms. 
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from abc import ABC, abstractmethod

import distrax
import jax
import jax.numpy as jnp
from chex import Array

import shinrl as srl


class TargetMixIn(ABC):
    @abstractmethod
    def target_log_pol(self, q: Array, data: srl.DataDict) -> Array:
        pass

    @abstractmethod
    def target_q_tabular_dp(self, data: srl.DataDict) -> Array:
        pass

    @abstractmethod
    def target_q_tabular_rl(self, data: srl.DataDict, samples: srl.Sample) -> Array:
        pass

    @abstractmethod
    def target_q_deep_dp(self, data: srl.DataDict) -> Array:
        pass

    @abstractmethod
    def target_q_deep_rl(self, data: srl.DataDict, samples: srl.Sample) -> Array:
        pass


class QTargetMixIn(TargetMixIn):
    def initialize(self, env, config=None) -> None:
        super().initialize(env, config)
        self.pol_loss_fn = srl.cross_entropy_loss

    def target_log_pol(self, q: Array, data: srl.DataDict) -> Array:
        eps = srl.calc_eps(
            data["n_step"],
            self.config.eps_decay_target_pol,
            0,
            1e-5,  # for numerical stability
        )
        return distrax.EpsilonGreedy(q, eps).logits

    def target_q_tabular_dp(self, data: srl.DataDict) -> Array:
        q = data["Q"]
        policy = jax.nn.softmax(data["LogPolicy"])
        return srl.expected_backup_dp(
            q,
            policy,
            self.env.mdp.rew_mat,
            self.env.mdp.tran_mat,
            self.config.discount,
        )

    def target_q_tabular_rl(self, data: srl.DataDict, samples: srl.Sample) -> Array:
        next_state = samples.next_state.squeeze(axis=1)
        next_q = data["Q"][next_state]  # BxA
        next_policy = jax.nn.softmax(data["LogPolicy"][next_state])  # BxA
        return srl.expected_backup_rl(
            next_q,
            next_policy,
            samples.rew,
            samples.done,
            self.config.discount,
        )

    def target_q_deep_dp(self, data: srl.DataDict) -> Array:
        obs = self.env.mdp.obs_mat
        q = self.q_net.apply(data["QNetTargParams"], obs)
        log_policy = self.log_pol_net.apply(data["LogPolNetParams"], obs)
        policy = jax.nn.softmax(log_policy)
        return srl.expected_backup_dp(
            q,
            policy,
            self.env.mdp.rew_mat,
            self.env.mdp.tran_mat,
            self.config.discount,
        )

    def target_q_deep_rl(self, data: srl.DataDict, samples: srl.Sample) -> Array:
        next_obs = samples.next_obs
        next_q = self.q_net.apply(data["QNetTargParams"], next_obs)
        next_log_pol = self.log_pol_net.apply(data["LogPolNetParams"], next_obs)
        next_pol = jax.nn.softmax(next_log_pol, axis=-1)
        return srl.expected_backup_rl(
            next_q,
            next_pol,
            samples.rew,
            samples.done,
            self.config.discount,
        )


# ----- Soft Q algorithm -----


class SoftQTargetMixIn(TargetMixIn):
    def initialize(self, env, config=None) -> None:
        super().initialize(env, config)
        self.pol_loss_fn = srl.kl_loss

    def target_log_pol(self, q: Array, data: srl.DataDict) -> Array:
        return q / self.config.er_coef

    def target_q_tabular_dp(self, data: srl.DataDict) -> Array:
        q = data["Q"]
        log_policy = data["LogPolicy"]
        policy = jax.nn.softmax(log_policy)
        return srl.soft_expected_backup_dp(
            q,
            policy,
            log_policy,
            self.env.mdp.rew_mat,
            self.env.mdp.tran_mat,
            self.config.discount,
            self.config.er_coef,
        )

    def target_q_tabular_rl(self, data: srl.DataDict, samples: srl.Sample) -> Array:
        next_q = data["Q"][samples.next_state.squeeze(axis=1)]  # BxA
        next_log_pol = data["LogPolicy"][samples.next_state.squeeze(axis=1)]  # BxA
        next_pol = jax.nn.softmax(next_log_pol, axis=-1)
        return srl.soft_expected_backup_rl(
            next_q,
            next_pol,
            next_log_pol,
            samples.rew,
            samples.done,
            self.config.discount,
            self.config.er_coef,
        )

    def target_q_deep_dp(self, data: srl.DataDict) -> Array:
        obs = self.env.mdp.obs_mat
        q = self.q_net.apply(data["QNetTargParams"], obs)
        log_policy = self.log_pol_net.apply(data["LogPolNetParams"], obs)
        policy = jax.nn.softmax(log_policy)
        return srl.soft_expected_backup_dp(
            q,
            policy,
            log_policy,
            self.env.mdp.rew_mat,
            self.env.mdp.tran_mat,
            self.config.discount,
            self.config.er_coef,
        )

    def target_q_deep_rl(self, data: srl.DataDict, samples: srl.Sample) -> Array:
        next_obs = samples.next_obs
        next_q = self.q_net.apply(data["QNetTargParams"], next_obs)
        next_log_pol = self.log_pol_net.apply(data["LogPolNetParams"], next_obs)
        next_pol = jax.nn.softmax(next_log_pol, axis=-1)
        return srl.soft_expected_backup_rl(
            next_q,
            next_pol,
            next_log_pol,
            samples.rew,
            samples.done,
            self.config.discount,
            self.config.er_coef,
        )
