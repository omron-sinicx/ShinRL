"""MixIns to execute one-step update.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from copy import deepcopy
from typing import Optional

import gym

import shinrl as srl

from .config import ViConfig


class TabularDpStepMixIn:
    def step(self):
        # Update Q table
        self.data["Q"] = self.target_tabular_dp(self.data)

        # Update ExplorePolicy & EvaluatePolicy tables
        self.update_tb_data()
        return {}


class TabularRlStepMixIn:
    def step(self):
        # Collect samples
        samples = self.explore()

        # Update Q table
        q_targ = self.target_tabular_rl(self.data, samples)
        state, act = samples.state, samples.act  # B
        q = self.data["Q"]
        self.data["Q"] = srl.calc_ma(self.config.lr, state, act, q, q_targ)

        # Update ExplorePolicy & EvaluatePolicy tables
        self.update_tb_data()
        return {}


class DeepDpStepMixIn:
    def step(self):
        # Compute new parameters
        loss, q_prm, opt_state = self.calc_params(self.data)

        # Update parameters
        self.data.update(
            {
                "QNetParams": q_prm,
                "QOptState": opt_state,
            }
        )
        if (self.n_step + 1) % self.config.target_update_interval == 0:
            self.data["QNetTargParams"] = deepcopy(self.data["QNetParams"])

        # Update ExplorePolicy & EvaluatePolicy tables
        self.update_tb_data()
        return {"Loss": loss.item()}


class DeepRlStepMixIn:
    def initialize(self, env: gym.Env, config: Optional[ViConfig] = None) -> None:
        super().initialize(env, config)
        self.buffer = srl.make_replay_buffer(self.env, self.config.buffer_size)

    def step(self):
        # Collect samples
        self.explore(store_to_buffer=True)
        samples = srl.Sample(**self.buffer.sample(self.config.batch_size))

        # Compute new parameters
        loss, q_prm, opt_state = self.calc_params(self.data, samples)

        # Update parameters
        self.data.update(
            {
                "QNetParams": q_prm,
                "QOptState": opt_state,
            }
        )
        if (self.n_step + 1) % self.config.target_update_interval == 0:
            self.data["QNetTargParams"] = deepcopy(self.data["QNetParams"])

        if self.is_shin_env:
            # Update ExplorePolicy & EvaluatePolicy tables
            self.update_tb_data()
        return {"Loss": loss.item()}
