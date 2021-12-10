"""Common mixins for evaluation and exploration.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import distrax
import jax
from chex import Array, PRNGKey

from shinrl import ACT_FN, Sample, collect_samples


class ShinEvalMixIn:
    def evaluate(self) -> Dict[str, float]:
        """Do oracle evaluation using `ExploitPolicy` table.

        Returns:
            Dict[str, float]: Dict of evaluation results.
        """
        assert self.is_shin_env
        assert (
            "ExploitPolicy" in self.tb_dict
        ), "ExploitPolicy table is not set. Call self.tb_dict.set('ExploitPolicy', Array)."
        pol = self.tb_dict["ExploitPolicy"]
        ret = self.env.calc_return(pol)
        return {"Return": ret}


@jax.jit
def tb_act(key: PRNGKey, state: int, policy: Array) -> Tuple[PRNGKey, Array, Array]:
    """Act according to a policy table.

    Args:
        key (PRNGKey)
        state (int): state id.
        policy (Array): dSxdA policy table.

    Returns:
        new_key (PRNGKey): generated new key.
        act (Array):
        log_prob (Array):
    """
    key, new_key = jax.random.split(key)
    dist = distrax.Categorical(probs=policy[state])
    act = dist.sample(seed=key)
    log_prob = dist.log_prob(act)
    return new_key, act, log_prob


class ShinExploreMixIn:
    def collect_samples(self, store_to_buffer=True) -> Sample:
        """
        Collect samples using `ExplorePolicy` table.

        Returns:
            Sample
        """
        assert self.is_shin_env
        assert (
            "ExplorePolicy" in self.tb_dict
        ), "ExplorePolicy table is not set. Call self.tb_dict.set('ExplorePolicy', Array)."

        buffer = self.buffer if store_to_buffer else None
        policy = self.tb_dict["ExplorePolicy"]
        self.key, samples = collect_samples(
            key=self.key,
            env=self.env,
            act_fn=lambda key, state: tb_act(key, state, policy),
            num_samples=self.config.num_samples,
            buffer=buffer,
            use_state=True,
        )
        return samples


class GymEvalMixIn(ABC):
    @abstractmethod
    def make_exploit_act(self) -> ACT_FN:
        """Make an act function for exploitation. Used in evaluation.

        Returns:
            ACT_FN
        """
        pass

    def evaluate(self) -> Dict[str, float]:
        """
        Do oracle evaluation if the env is ShinEnv.
        Otherwise do sampling-based evaluation.

        Returns:
            Dict[str, float]: Dict of evaluation results.
        """
        if self.is_shin_env:
            # oracle evaluation
            pol = self.tb_dict["ExploitPolicy"]
            ret = self.env.calc_return(pol)
        else:
            # sampling-based evaluation
            self.key, samples = collect_samples(
                key=self.key,
                env=self.env,
                act_fn=self.make_exploit_act(),
                num_episodes=self.config.eval_trials,
                use_state=False,
            )
            ret = (samples.rew.sum() / self.config.eval_trials).item()
        return {"Return": ret}


class GymExploreMixIn(ABC):
    @abstractmethod
    def make_explore_act(
        self, key: PRNGKey, obs: Array, **kwargs: Array
    ) -> Tuple[PRNGKey, Array, Array]:
        """Make an act function for exploration.
        Returns:
            ACT_FN
        """
        pass

    def collect_samples(self, store_to_buffer=True) -> Sample:
        """
        Collect samples using explore_act function.

        Returns:
            Sample
        """
        buffer = self.buffer if store_to_buffer else None
        self.key, samples = collect_samples(
            key=self.key,
            env=self.env,
            act_fn=self.make_explore_act(),
            num_samples=self.config.num_samples,
            buffer=buffer,
            use_state=False,
        )
        return samples
