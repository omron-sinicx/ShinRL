from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import gym
import jax
import jax.numpy as jnp
import numpy as np

"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from chex import Array, PRNGKey
from cpprb import ReplayBuffer

import shinrl as srl
from shinrl import MDP, EnvConfig

OBS, REW, DONE, INFO = Array, float, bool, Dict[str, Any]
S, A = int, int
STATES, PROBS = Array, Array
TRAN_FN = Callable[[S, A], Tuple[STATES, PROBS]]
REW_FN = Callable[[S, A], REW]
OBS_FN = Callable[[S], OBS]


class ShinEnv(ABC, gym.Env):
    """
    ABC to implement a new shin-environment.
    You need to implement four functions:
    * _init_probs: Return a probability array (dS).
    * _make_reward_fn: Return a jittable reward function (REW_FN)
    * _make_transition_fn: Return a jittable transition function (TRAN_FN)
    * _make_observation_fn: Return a jittable observation function (OBS_FN)
    Args:
        config (EnvConfig): Environment's configuration.
    """

    DefaultConfig = EnvConfig

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.initialize(config)

    def initialize(self, config: Optional[EnvConfig] = None):
        self._state: int = -1
        self.elapsed_steps: int = 0
        config = self.DefaultConfig() if config is None else config
        self._config = config
        self.key: PRNGKey = None
        self.seed()

        # set core functions
        self.transition = self._make_transition_fn()
        self.reward = self._make_reward_fn()
        self.observation = self._make_observation_fn()
        self.init_probs = self._init_probs()

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

        # jit main functions
        # TODO: Config is fixed at instantiation. Better to implement with initialize function like solvers.
        def step(key, state, action):
            states, probs = self.transition(state, action)
            new_key, key = jax.random.split(key)
            next_state = jax.random.choice(key, states, p=probs)
            reward = self.reward(state, action)
            next_obs = self.observation(next_state)
            return new_key, next_state, reward, next_obs

        def reset(key):
            init_states, init_probs = self._init_states, self.init_probs
            new_key, key = jax.random.split(key)
            init_state = jax.random.choice(key, init_states, p=init_probs)
            init_obs = self.observation(init_state)
            return new_key, init_state, init_obs

        # TODO: bug fix with jax.jit
        self._step = jax.jit(step)
        self._reset = jax.jit(reset)

        self.reset()

    @property
    @abstractmethod
    def dS(self) -> int:
        pass

    @property
    @abstractmethod
    def dA(self) -> int:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Space:
        pass

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Space:
        pass

    @abstractmethod
    def _init_probs(self) -> PROBS:
        """
        Returns:
            PROBS: (dS Array) Probabilities of initial states.
        """
        pass

    @abstractmethod
    def _make_transition_fn(self) -> TRAN_FN:
        """
        Returns:
            TRAN_FN: Jittable transition function.
        """
        pass

    @abstractmethod
    def _make_reward_fn(self) -> REW_FN:
        """
        Returns:
            REW_FN: Jittable reward function.
        """
        pass

    @abstractmethod
    def _make_observation_fn(self) -> OBS_FN:
        """
        Returns:
            OBS_FN: Jittable observation function.
        """
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

    def step(self, action: int) -> Tuple[OBS, REW, DONE, INFO]:
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

    def reset(self) -> OBS:
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
