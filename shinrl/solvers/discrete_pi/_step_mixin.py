"""MixIns to execute one-step update.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from copy import deepcopy
from typing import Optional

import gym

import shinrl as srl

from .config import PiConfig


class TabularDpStepMixIn:
    def step(self):
        # Update Policy & Q tables
        self.data["LogPolicy"] = self.target_log_pol(self.data["Q"], self.data)
        self.data["Q"] = self.target_q_tabular_dp(self.data)

        # Update ExplorePolicy & EvaluatePolicy tables
        self.update_tb_data()
        return {}


class TabularRlStepMixIn:
    def step(self):
        # Collect samples
        samples = self.explore()

        # Update Policy & Q tables
        self.data["LogPolicy"] = self.target_log_pol(self.data["Q"], self.data)
        q_targ = self.target_q_tabular_rl(self.data, samples)
        state, act = samples.state, samples.act  # B
        q = self.data["Q"]
        self.data["Q"] = srl.calc_ma(self.config.q_lr, state, act, q, q_targ)

        # Update ExplorePolicy & EvaluatePolicy tables
        self.update_tb_data()
        return {}


class DeepDpStepMixIn:
    def step(self) -> None:
        # Compute new parameters
        pol_res, q_res = self.calc_params(self.data)
        pol_loss, pol_prm, pol_state = pol_res
        q_loss, q_prm, q_state = q_res

        # Update parameters
        self.data.update(
            {
                "LogPolNetParams": pol_prm,
                "LogPolOptState": pol_state,
                "QNetParams": q_prm,
                "QOptState": q_state,
            }
        )
        if (self.n_step + 1) % self.config.target_update_interval == 0:
            self.data["QNetTargParams"] = deepcopy(self.data["QNetParams"])

        # Update ExplorePolicy & EvaluatePolicy tables
        self.update_tb_data()
        return {"PolLoss": pol_loss.item(), "QLoss": q_loss.item()}


class DeepRlStepMixIn:
    def initialize(self, env: gym.Env, config: Optional[PiConfig] = None) -> None:
        super().initialize(env, config)
        self.buffer = srl.make_replay_buffer(self.env, self.config.buffer_size)

    def step(self) -> None:
        # Collect samples
        samples = self.explore(store_to_buffer=True)
        samples = srl.Sample(**self.buffer.sample(self.config.batch_size))

        # Compute new parameters
        pol_res, q_res = self.calc_params(self.data, samples)
        pol_loss, pol_prm, pol_state = pol_res
        q_loss, q_prm, q_state = q_res

        # Update parameters
        self.data.update(
            {
                "LogPolNetParams": pol_prm,
                "LogPolOptState": pol_state,
                "QNetParams": q_prm,
                "QOptState": q_state,
            }
        )
        if (self.n_step + 1) % self.config.target_update_interval == 0:
            self.data["QNetTargParams"] = deepcopy(self.data["QNetParams"])

        if self.is_shin_env:
            # Update ExplorePolicy & EvaluatePolicy tables
            self.update_tb_data()
        return {"PolLoss": pol_loss.item(), "QLoss": q_loss.item()}
