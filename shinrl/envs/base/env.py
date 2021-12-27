from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

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

from .config import EnvConfig
from .mdp import MDP

OBS, REW, DONE, INFO = Array, float, bool, Dict[str, Any]


class ShinEnv(ABC, gym.Env):
    """ ABC to implement a new shin-environment. """

    # ########## YOU NEED TO IMPLEMENT HERE ##########

    DefaultConfig = EnvConfig

    @property
    @abstractmethod
    def dS(self) -> int:
        """Number of states."""
        pass

    @property
    @abstractmethod
    def dA(self) -> int:
        """Number of actions."""
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
    def init_probs(self) -> Array:
        """A function that returns the probabilities of initial states.

        Returns:
            probs (dS Array): Probabilities of initial states.
        """
        pass

    @abstractmethod
    def transition(self, state: int, act: int) -> Tuple[Array, Array]:
        """Transition function of the MDP.

        Args:
            state (int): state id.
            act (int): action id.

        Returns:
            next_states (Array): Next state ids.
            probs (Array): Probabilities of next state ids.
        """
        pass

    @abstractmethod
    def reward(self, state: int, act: int) -> float:
        """Reward function of the MDP.

        Args:
            state (int): state id.
            act (int): action id.

        Returns:
            rew (float): reward.
        """
        pass

    @abstractmethod
    def observation(self, state: int) -> Array:
        """Observation function of the MDP.

        Args:
            state (int): state id.

        Returns:
            obs (Array): observation Array.
        """
        pass

    def continuous_action(self, act: int) -> Array:
        """A function that converts a discrete action to a continuous action.

        Args:
            act (int): action id.

        Returns:
            c_act (Array): continuous action Array.
        """
        raise NotImplementedError

    def discrete_action(self, c_act: Array) -> int:
        """A function that converts a continuous action to a discrete action.

        Args:
            c_act (Array): continuous action Array.

        Returns:
            act (int): action id.
        """
        raise NotImplementedError

    # ################################################

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.initialize(config)

    def initialize(self, config: Optional[EnvConfig] = None):
        self._config = self.DefaultConfig() if config is None else config
        self._state: int = -1
        self.elapsed_steps: int = 0
        self.key: PRNGKey = None
        self.seed()

        is_continuous = isinstance(self.action_space, gym.spaces.Box)
        if is_continuous:
            is_high_normalized = (self.action_space.high == 1.0).all()
            is_low_normalized = (self.action_space.low == -1.0).all()
            assert_msg = """
            The current ShinEnv assumes that the env.actions_space is in range [-1, 1].
            Please normalize the action_space of the implemented env.
            """
            assert is_high_normalized and is_low_normalized, assert_msg

        self.mdp = self._build_mdp()
        self._step = self._build_c_step() if is_continuous else self._build_step()
        self._reset = self._build_reset()

        self.reset()

    def _build_mdp(self) -> MDP:
        is_continuous = isinstance(self.action_space, gym.spaces.Box)
        obs_shape = self.observation_space.shape
        dS, dA = self.dS, self.dA
        act_shape, act_mat = None, None
        if is_continuous:
            act_shape = self.action_space.shape
            act_mat = MDP.make_act_mat(self.continuous_action, dA, act_shape)
        mdp = MDP(
            dS=dS,
            dA=dA,
            obs_shape=obs_shape,
            obs_mat=MDP.make_obs_mat(self.observation, dS, obs_shape),
            rew_mat=MDP.make_rew_mat(self.reward, dS, dA),
            tran_mat=MDP.make_tran_mat(self.transition, dS, dA),
            init_probs=self.init_probs(),
            discount=self.config.discount,
            act_shape=act_shape,
            act_mat=act_mat,
        )
        mdp.is_valid_mdp(mdp)
        return mdp

    def _build_step(self):
        def step(key, state, action):
            states, probs = self.transition(state, action)
            new_key, key = jax.random.split(key)
            next_state = jax.random.choice(key, states, p=probs)
            reward = self.reward(state, action)
            next_obs = self.observation(next_state)
            return new_key, next_state, reward, next_obs

        return jax.jit(step)

    def _build_c_step(self):
        def continuous_step(key, state, action):
            action = self.discrete_action(action)
            states, probs = self.transition(state, action)
            new_key, key = jax.random.split(key)
            next_state = jax.random.choice(key, states, p=probs)
            reward = self.reward(state, action)
            next_obs = self.observation(next_state)
            return new_key, next_state, reward, next_obs

        return jax.jit(continuous_step)

    def _build_reset(self):
        init_probs = self.init_probs()
        init_states = jnp.arange(0, self.dS)

        def reset(key):
            new_key, key = jax.random.split(key)
            init_state = jax.random.choice(key, init_states, p=init_probs)
            init_obs = self.observation(init_state)
            return new_key, init_state, init_obs

        return jax.jit(reset)

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
        assert 0 <= self._state, "Invalid state. Check the transition function."
        assert self._state < self.dS, "Invalid state. Check the transition function"
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
