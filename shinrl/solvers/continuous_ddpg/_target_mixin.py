"""MixIns to compute the target value of DDPG. 
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from abc import ABC, abstractmethod

import distrax
import jax.numpy as jnp
from chex import Array
from distrax import Categorical

import shinrl as srl


class TargetMixIn(ABC):
    @abstractmethod
    def target_q_deep_dp(self, data: srl.DataDict, next_pol_dist: Categorical) -> Array:
        pass

    @abstractmethod
    def target_q_deep_rl(
        self, data: srl.DataDict, pol_dist: Categorical, samples: srl.Sample
    ) -> Array:
        pass


class QTargetMixIn(TargetMixIn):
    def target_q_deep_dp(self, data: srl.DataDict) -> Array:
        obs = self._sa_obs_mat  # (dSxdA) x obs_shape
        act = self._sa_act_mat  # (dSxdA) x act_shape

        mean = self.pol_net.apply(data["PolNetTargParams"], obs)  # (dSxdA) x act_shape
        dist = distrax.Normal(mean, 1.0)  # (dSxdA) x act_shape
        log_prob = dist.log_prob(act).mean(axis=-1)  # (dSxdA) x 1
        log_prob = log_prob.reshape(self.dS, self.dA)  # dS x dA
        pol_dist = distrax.Greedy(log_prob)  # dS x dA
        q = self.q_net.apply(data["QNetTargParams"], obs, act)
        q = q.reshape(self.dS, self.dA)  # dS x dA
        return srl.expected_backup_dp(
            q,
            pol_dist.probs,
            self.env.mdp.rew_mat,
            self.env.mdp.tran_mat,
            self.config.discount,
        )

    def target_q_deep_rl(self, data: srl.DataDict, samples: srl.Sample) -> Array:
        next_obs = samples.next_obs
        next_act = self.pol_net.apply(data["PolNetTargParams"], next_obs)  # ?x1
        next_q = self.q_net.apply(data["QNetTargParams"], next_obs, next_act)  # ?x1
        dummy_next_pol = jnp.ones_like(next_q)
        return srl.expected_backup_rl(
            next_q,
            dummy_next_pol,
            samples.rew,
            samples.done,
            self.config.discount,
        )
