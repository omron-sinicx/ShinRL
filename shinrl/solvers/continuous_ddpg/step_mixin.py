"""MixIns to execute one-step update.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from typing import Optional

import gym
import haiku as hk
import jax

import shinrl as srl

from .config import DdpgConfig


@jax.jit
def soft_target_update(
    params: hk.Params, targ_params: hk.Params, polyak: float = 0.005
) -> hk.Params:
    return jax.tree_multimap(
        lambda p, tp: (1 - polyak) * tp + polyak * p, params, targ_params
    )


class DeepDpStepMixIn:
    def step(self) -> None:
        # Compute new Pol and Q Net params
        pol_res, q_res = self.calc_params(self.data)
        pol_loss, pol_prm, pol_state = pol_res
        q_loss, q_prm, q_state = q_res

        # Update params
        self.data.update(
            {
                "PolNetParams": pol_prm,
                "PolOptState": pol_state,
                "QNetParams": q_prm,
                "QOptState": q_state,
            }
        )

        self.data["QNetTargParams"] = soft_target_update(
            self.data["QNetParams"],
            self.data["QNetTargParams"],
            self.config.polyak_rate,
        )
        self.data["PolNetTargParams"] = soft_target_update(
            self.data["PolNetParams"],
            self.data["PolNetTargParams"],
            self.config.polyak_rate,
        )
        # Update Q & Policy tables
        self.update_tb_data()
        return {"PolLoss": pol_loss.item(), "QLoss": q_loss.item()}


class DeepRlStepMixIn:
    def initialize(self, env: gym.Env, config: Optional[DdpgConfig] = None) -> None:
        super().initialize(env, config)
        self.buffer = srl.make_replay_buffer(self.env, self.config.buffer_size)

    def step(self) -> None:
        samples = self.explore(store_to_buffer=True)
        samples = srl.Sample(**self.buffer.sample(self.config.batch_size))

        if self.buffer.get_stored_size() < self.config.replay_start_size:
            return {}

        # Compute new Pol and Q Net params
        pol_res, q_res = self.calc_params(self.data, samples)
        pol_loss, pol_prm, pol_state = pol_res
        q_loss, q_prm, q_state = q_res

        # Update params
        self.data.update(
            {
                "PolNetParams": pol_prm,
                "PolOptState": pol_state,
                "QNetParams": q_prm,
                "QOptState": q_state,
            }
        )

        self.data["QNetTargParams"] = soft_target_update(
            self.data["QNetParams"],
            self.data["QNetTargParams"],
            self.config.polyak_rate,
        )
        self.data["PolNetTargParams"] = soft_target_update(
            self.data["PolNetParams"],
            self.data["PolNetTargParams"],
            self.config.polyak_rate,
        )
        # Update Q & Policy tables
        if self.is_shin_env:
            self.update_tb_data()
        return {"PolLoss": pol_loss.item(), "QLoss": q_loss.item()}
