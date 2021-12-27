"""MixIns to compute the target value of VI-based algorithms.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from chex import Array

import shinrl as srl


class TargetMixIn:
    def target_tabular_dp(self, data: srl.DataDict) -> Array:
        raise NotImplementedError

    def target_tabular_rl(self, data: srl.DataDict, samples: srl.Sample) -> Array:
        raise NotImplementedError

    def target_deep_dp(self, data: srl.DataDict) -> Array:
        raise NotImplementedError

    def target_deep_rl(self, data: srl.DataDict, samples: srl.Sample) -> Array:
        raise NotImplementedError


class QTargetMixIn(TargetMixIn):
    """MixIn to compute the vanilla Q target."""

    def target_tabular_dp(self, data: srl.DataDict) -> Array:
        return srl.optimal_backup_dp(
            data["Q"],
            self.env.mdp.rew_mat,
            self.env.mdp.tran_mat,
            self.config.discount,
        )

    def target_tabular_rl(self, data: srl.DataDict, samples: srl.Sample) -> Array:
        return srl.optimal_backup_rl(
            data["Q"][samples.next_state.squeeze(axis=1)],  # BxA
            samples.rew,
            samples.done,
            self.config.discount,
        )

    def target_deep_dp(self, data: srl.DataDict) -> Array:
        return srl.optimal_backup_dp(
            self.q_net.apply(data["QNetTargParams"], self.env.mdp.obs_mat),
            self.env.mdp.rew_mat,
            self.env.mdp.tran_mat,
            self.config.discount,
        )

    def target_deep_rl(self, data: srl.DataDict, samples: srl.Sample) -> Array:
        return srl.optimal_backup_rl(
            self.q_net.apply(data["QNetTargParams"], samples.next_obs),
            samples.rew,
            samples.done,
            self.config.discount,
        )


class DoubleQTargetMixIn(TargetMixIn):
    """MixIn to compute the Double Q target.
    Paper: https://arxiv.org/abs/1509.06461
    """

    def target_deep_dp(self, data: srl.DataDict) -> Array:
        return srl.double_backup_dp(
            self.q_net.apply(data["QNetTargParams"], self.env.mdp.obs_mat),
            self.q_net.apply(data["QNetParams"], self.env.mdp.obs_mat),
            self.env.mdp.rew_mat,
            self.env.mdp.tran_mat,
            self.config.discount,
        )

    def target_deep_rl(self, data: srl.DataDict, samples: srl.Sample) -> Array:
        return srl.double_backup_rl(
            self.q_net.apply(data["QNetTargParams"], samples.next_obs),
            self.q_net.apply(data["QNetParams"], samples.next_obs),
            samples.rew,
            samples.done,
            self.config.discount,
        )


class MunchausenTargetMixIn(TargetMixIn):
    """MixIn to compute the Munchausen Q target.
    Paper: https://arxiv.org/abs/2007.14430
    """

    def target_tabular_dp(self, data: srl.DataDict) -> Array:
        return srl.munchausen_backup_dp(
            data["Q"],
            self.env.mdp.rew_mat,
            self.env.mdp.tran_mat,
            self.config.discount,
            self.config.kl_coef,
            self.config.er_coef,
            self.config.logp_clip,
        )

    def target_tabular_rl(self, data: srl.DataDict, samples: srl.Sample) -> Array:
        return srl.munchausen_backup_rl(
            data["Q"][samples.next_state.squeeze(axis=1)],  # BxA
            data["Q"][samples.state.squeeze(axis=1)],  # BxA
            samples.rew,
            samples.done,
            samples.act,
            self.config.discount,
            self.config.kl_coef,
            self.config.er_coef,
            self.config.logp_clip,
        )

    def target_deep_dp(self, data: srl.DataDict) -> Array:
        return srl.munchausen_backup_dp(
            self.q_net.apply(data["QNetTargParams"], self.env.mdp.obs_mat),
            self.env.mdp.rew_mat,
            self.env.mdp.tran_mat,
            self.config.discount,
            self.config.kl_coef,
            self.config.er_coef,
            self.config.logp_clip,
        )

    def target_deep_rl(self, data: srl.DataDict, samples: srl.Sample) -> Array:
        return srl.munchausen_backup_rl(
            self.q_net.apply(data["QNetTargParams"], samples.next_obs),  # BxA
            self.q_net.apply(data["QNetTargParams"], samples.obs),  # BxA
            samples.rew,
            samples.done,
            samples.act,
            self.config.discount,
            self.config.kl_coef,
            self.config.er_coef,
            self.config.logp_clip,
        )
