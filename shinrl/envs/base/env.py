from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Dict, Optional, Tuple

import gym
import jax
import jax.numpy as jnp
import numpy as np

"""
Author: Toshinori Kitamura
Affiliation: NAIST
"""
from chex import Array, PRNGKey
from cpprb import ReplayBuffer

import shinrl as srl
from shinrl import MDP, EnvConfig

Trans = Tuple[Array, float, bool, Dict[str, Any]]  # obs, rew, done, info


@jax.jit
def choice_and_step_key(key: PRNGKey, a: Array, p: Array) -> Tuple[Array, PRNGKey]:
    new_key, key = jax.random.split(key)
    return jax.random.choice(key, a, p=p), new_key


class ShinEnv(ABC, gym.Env):
    """
    Args:
        config (EnvConfig): Environment's configuration.
    """

    DefaultConfig = EnvConfig

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self._state: int = -1
        self.elapsed_steps: int = 0
        self.key: PRNGKey = None
        config = self.DefaultConfig() if config is None else config
        self._config = config

        # set mdp
        obs_shape = self.observation_space.shape
        dS, dA = self.dS, self.dA
        self._init_states = jnp.arange(0, dS)
        self.mdp: MDP = MDP(
            dS=dS,
            dA=dA,
            obs_shape=obs_shape,
            obs_mat=MDP.make_obs_mat(self.observation, dS, obs_shape),
            rew_mat=MDP.make_rew_mat(self.reward, dS, dA),
            tran_mat=MDP.make_tran_mat(self.transition, dS, dA),
            init_probs=self.init_probs,
            discount=config.discount,
        )
        self.seed()

        # jit main functions
        # TODO: Config is fixed at instantiation. Better to implement with initialize function like solvers.
        def step(key, state, action):
            states, probs = self.transition(state, action)
            next_state, new_key = choice_and_step_key(key, states, probs)
            reward = self.reward(state, action)
            next_obs = self.observation(next_state)
            return new_key, next_state, reward, next_obs

        def reset(key):
            init_states, init_probs = self._init_states, self.mdp.init_probs
            init_state, new_key = choice_and_step_key(key, init_states, init_probs)
            init_obs = self.observation(init_state)
            return new_key, init_state, init_obs

        # TODO: bug fix with jax.jit
        self._step = step
        self._reset = reset

    @property
    @abstractmethod
    def dS(self) -> int:
        pass

    @property
    @abstractmethod
    def dA(self) -> int:
        pass

    @cached_property
    @abstractmethod
    def init_probs(self) -> Array:
        """(dS Array): Probability of initial states."""
        pass

    @cached_property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Space:
        pass

    @cached_property
    @abstractmethod
    def action_space(self) -> gym.spaces.Space:
        pass

    @abstractmethod
    def transition(self, state: int, action: int) -> Tuple[Array, Array]:
        pass

    @abstractmethod
    def reward(self, state: int, action: int) -> float:
        pass

    @abstractmethod
    def observation(self, state: int) -> Array:
        pass

    @property
    def config(self) -> EnvConfig:
        return self._config

    def get_state(self) -> int:
        """Return the current state"""
        return self._state

    def seed(self, seed: int = 0) -> None:
        """
        NOTE: sample from gym.spaces are not recommended
        because np_random and jax do not share the same PRNG.
        Use jax.random as far as possible.
        """

        self.key = jax.random.PRNGKey(seed)
        self.action_space.np_random.seed(seed)
        self.observation_space.np_random.seed(seed)

    def step(self, action: int) -> Trans:
        """Simulate the environment by one timestep.

        Args:
          action (int): Action to take

        Returns:
          next_obs (Array): Next observation
          reward (float): Reward incurred by agent
          done (bool): A boolean indicating the end of an episode
          info (dict): A debug info dictionary.
        """

        state = self._state
        new_key, next_state, reward, next_obs = self._step(self.key, state, action)
        self.key, self._state = new_key, next_state.item()
        done = False
        self.elapsed_steps += 1
        info = {"state": self._state}
        if self.elapsed_steps >= self.config.horizon:
            info["TimeLimit.truncated"] = True
            done = True
        else:
            info["TimeLimit.truncated"] = False
        trans = (next_obs, reward, done, info)
        return trans

    def reset(self) -> Array:
        """Resets the state of the environment and returns an initial observation.

        Returns:
          observation (Array): The agent's initial observation.
        """

        self.elapsed_steps = 0
        self.key, init_state, init_obs = self._reset(self.key)
        self._state = init_state.item()
        return init_obs

    def render(self) -> None:
        pass

    def calc_return(self, policy: Array) -> float:
        ret = srl.calc_return(
            policy,
            self.mdp.rew_mat,
            self.mdp.tran_mat,
            self.mdp.init_probs,
            self.config.horizon,
        )
        return ret.item()

    def calc_optimal_q(self) -> Array:
        q = srl.calc_optimal_q(
            self.mdp.rew_mat,
            self.mdp.tran_mat,
            self.config.discount,
            self.config.horizon,
        )
        return q

    def calc_q(self, policy: Array) -> Array:
        q = srl.calc_q(
            policy,
            self.mdp.rew_mat,
            self.mdp.tran_mat,
            self.config.discount,
            self.config.horizon,
        )
        return q

    def calc_visit(self, policy: Array) -> Array:
        visit = srl.calc_visit(
            policy,
            self.mdp.rew_mat,
            self.mdp.tran_mat,
            self.mdp.init_probs,
            self.config.discount,
            self.config.horizon,
        )
        return visit

    def count_visit(self, buffer: ReplayBuffer) -> Array:
        """
        Count the number of state-action pairs in a buffer
        Args:
            buffer (cpprb.ReplayBuffer)
        Returns:
            SA: SxA matrix
        """
        samples = buffer.get_all_transitions()
        state, act = samples["state"].reshape(-1), samples["act"].reshape(-1)
        assert act.dtype == int
        sa = np.zeros((self.dS, self.dA))  # SxA
        for s, a in zip(state.reshape(-1), act.reshape(-1)):
            sa[s, a] += 1
        return sa
