"""MixIns to compute the target value of PI-based algorithms. 
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from abc import ABC, abstractmethod

import distrax
from chex import Array
from distrax import Categorical

import shinrl as srl


class TargetMixIn(ABC):
    @abstractmethod
    def target_pol_dist(self, q: Array) -> Categorical:
        pass

    @abstractmethod
    def target_q_tabular_dp(self, data: srl.DataDict, pol_dist: Categorical) -> Array:
        pass

    @abstractmethod
    def target_q_tabular_rl(
        self, data: srl.DataDict, next_pol_dist: Categorical, samples: srl.Sample
    ) -> Array:
        pass

    @abstractmethod
    def target_q_deep_dp(self, data: srl.DataDict, next_pol_dist: Categorical) -> Array:
        pass

    @abstractmethod
    def target_q_deep_rl(
        self, data: srl.DataDict, pol_dist: Categorical, samples: srl.Sample
    ) -> Array:
        pass


class QTargetMixIn(TargetMixIn):
    def target_pol_dist(self, q: Array) -> Categorical:
        return distrax.Greedy(q)

    def target_q_tabular_dp(self, data: srl.DataDict, pol_dist: Categorical) -> Array:
        return srl.expected_backup_dp(
            data["Q"],
            pol_dist.probs,
            self.env.mdp.rew_mat,
            self.env.mdp.tran_mat,
            self.config.discount,
        )

    def target_q_tabular_rl(
        self, data: srl.DataDict, next_pol_dist: Categorical, samples: srl.Sample
    ) -> Array:
        return srl.expected_backup_rl(
            data["Q"][samples.next_state.squeeze(axis=1)],  # BxA
            next_pol_dist.probs[samples.next_state.squeeze(axis=1)],  # BxA
            samples.rew,
            samples.done,
            self.config.discount,
        )

    def target_q_deep_dp(self, data: srl.DataDict, pol_dist: Categorical) -> Array:
        return srl.expected_backup_dp(
            self.q_net.apply(data["QNetTargParams"], self.env.mdp.obs_mat),
            pol_dist.probs,
            self.env.mdp.rew_mat,
            self.env.mdp.tran_mat,
            self.config.discount,
        )

    def target_q_deep_rl(
        self, data: srl.DataDict, next_pol_dist: Categorical, samples: srl.Sample
    ) -> Array:
        return srl.expected_backup_rl(
            self.q_net.apply(data["QNetTargParams"], samples.next_obs),
            next_pol_dist.probs,
            samples.rew,
            samples.done,
            self.config.discount,
        )


# ----- Soft Q algorithm -----


class SoftQTargetMixIn(TargetMixIn):
    def target_pol_dist(self, q: Array) -> Categorical:
        return distrax.Softmax(q, temperature=self.config.er_coef)

    def target_q_tabular_dp(self, data: srl.DataDict, pol_dist: Categorical) -> Array:
        return srl.soft_expected_backup_dp(
            data["Q"],
            pol_dist.probs,
            pol_dist.logits,
            self.env.mdp.rew_mat,
            self.env.mdp.tran_mat,
            self.config.discount,
            self.config.er_coef,
        )

    def target_q_tabular_rl(
        self, data: srl.DataDict, pol_dist: Categorical, samples: srl.Sample
    ) -> Array:
        return srl.soft_expected_backup_rl(
            data["Q"][samples.next_state.squeeze(axis=1)],  # BxA
            pol_dist.probs[samples.next_state.squeeze(axis=1)],  # BxA
            pol_dist.logits[samples.next_state.squeeze(axis=1)],  # BxA
            samples.rew,
            samples.done,
            self.config.discount,
            self.config.er_coef,
        )

    def target_q_deep_dp(self, data: srl.DataDict, pol_dist: Categorical) -> Array:
        return srl.soft_expected_backup_dp(
            self.q_net.apply(data["QNetTargParams"], self.env.mdp.obs_mat),
            pol_dist.probs,
            pol_dist.logits,
            self.env.mdp.rew_mat,
            self.env.mdp.tran_mat,
            self.config.discount,
            self.config.er_coef,
        )

    def target_q_deep_rl(
        self, data: srl.DataDict, pol_dist: Categorical, samples: srl.Sample
    ) -> Array:
        return srl.soft_expected_backup_rl(
            self.q_net.apply(data["QNetTargParams"], samples.next_obs),
            pol_dist.probs,
            pol_dist.logits,
            samples.rew,
            samples.done,
            self.config.discount,
            self.config.er_coef,
        )
