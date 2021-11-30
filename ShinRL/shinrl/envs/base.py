from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gym.utils import seeding
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from scipy import sparse

from shinrl.utils import lazy_property


class ShinEnv(gym.Env):
    """
    Args:
        dS (int): Number of states.
        dA (int): Number of actions
        init_state_dist(Dict): Initial state distribution represented as {state: probability}
        horizon (int, optional): Horizon of the environment.
    """

    def __init__(
        self,
        dS: int,
        dA: int,
        init_state_dist: Dict[int, float],
        horizon: int = np.iinfo(np.int32).max,
    ):
        self._state = -1
        self.observation_space = gym.spaces.Discrete(dS)
        self.action_space = gym.spaces.Discrete(dA)
        self.dS = dS
        self.dA = dA
        self.init_state_dist = init_state_dist
        self.horizon = horizon
        self._elapsed_steps = 0
        self.seed()
        super().__init__()

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def elapsed_steps(self) -> int:
        return self._elapsed_steps

    def get_state(self) -> int:
        return self._state

    def sample_from_transition(self, transition: Dict[int, float]) -> int:
        next_states = np.array(list(transition.keys()))
        ps = np.array(list(transition.values()))
        return self.np_random.choice(next_states, p=ps)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """Simulate the environment by one timestep.

        Args:
          action (int): Action to take

        Returns:
          observation (np.ndarray): Next observation
          reward (float): Reward incurred by agent
          done (bool): A boolean indicating the end of an episode
          info (dict): A debug info dictionary.
        """
        transition = self.transition(self._state, action)
        next_state = self.sample_from_transition(transition)
        reward = self.reward(self._state, action)
        done = False
        self._state = next_state
        next_obs = self.observation(self._state)
        self._elapsed_steps += 1
        infos = {"state": self._state}
        if self._elapsed_steps >= self.horizon:
            infos["TimeLimit.truncated"] = True
            done = True
        else:
            infos["TimeLimit.truncated"] = False
        return next_obs, reward, done, infos

    def reset(self) -> np.ndarray:
        """Resets the state of the environment and returns an initial observation.

        Returns:
          observation (np.ndarray): The agent's initial observation.
        """

        self._elapsed_steps = 0
        init_state = self.sample_from_transition(self.init_state_dist)
        self._state = init_state
        return self.observation(init_state)

    @abstractmethod
    def transition(self, state: int, action: int) -> Dict[int, float]:
        """Return the transition probabilities p(ns|s,a).

        Args:
          state (int): State id
          action (int): Action to take

        Returns:
          A python dict from {next state: probability}.
          (Omitted states with probability 0)
        """

    @abstractmethod
    def reward(self, state: int, action: int) -> float:
        """Return the reward

        Args:
          state (int): State id
          action (int): Action to take

        Returns:
          reward (float)
        """

    @abstractmethod
    def observation(self, state: int) -> np.ndarray:
        """Return the observation for a state.

        Args:
          state (int): State id

        Returns:
          observation (np.ndarray): Agent's observation of state, conforming with observation_space
        """

    def to_continuous_action(self, action: int) -> Optional[float]:
        """Map the discrete action to a continuous action

        Args:
            action (int): action_id

        Returns:
            float: A continuous action
        """
        return None

    def render(self) -> None:
        pass

    @lazy_property
    def all_observations(self) -> np.ndarray:
        """
        Returns:
            Observations of all the states (ndarray with dS x obs_dim shape)
        """
        obss = []
        for s in range(self.dS):
            s = self.observation(s)
            obss.append(np.expand_dims(s, axis=0))
        obss = np.vstack(obss)  # S x obs_dim
        return obss

    @lazy_property
    def all_actions(self) -> np.ndarray:
        """
        Returns:
            All the actions (ndarray with dA x 1 shape)
        """
        acts = []
        for a in range(self.dA):
            a = self.to_continuous_action(a)
            acts.append(np.expand_dims(a, axis=0))
        acts = np.vstack(acts)  # A x 1
        return acts

    @lazy_property
    def transition_matrix(self) -> sparse.csr.csr_matrix:
        """Construct the transition matrix.

        Returns:
          A (dS x dA) x dS array where the entry transition_matrix[sa, ns]
          corresponds to the probability of transitioning into state ns after taking
          action a from state s.
        """
        ds = self.dS
        da = self.dA
        row = []
        col = []
        data = []
        for s in range(ds):
            for a in range(da):
                transition = self.transition(s, a)
                for next_s, prob in transition.items():
                    row.append(da * s + a)
                    col.append(next_s)
                    data.append(prob)
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        return sparse.csr_matrix((data, (row, col)), shape=(ds * da, ds))

    @lazy_property
    def reward_matrix(self) -> sparse.csr.csr_matrix:
        """Construct the reward matrix.

        Returns:
          A dS x dA scipy.sparse array where the entry reward_matrix[s, a]
          reward given to an agent when transitioning into state ns after taking
          action a from state s.
        """
        ds = self.dS
        da = self.dA
        row = []
        col = []
        data = []
        for s in range(ds):
            for a in range(da):
                rew = self.reward(s, a)
                row.append(s)
                col.append(a)
                data.append(rew)
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        return sparse.csr_matrix((data, (row, col)), shape=(ds, da))

    def calc_q(
        self,
        policy: np.ndarray,
        discount: float = 0.99,
        base_policy: Optional[np.ndarray] = None,
        er_coef: float = 0.0,
        kl_coef: float = 0.0,
    ) -> np.ndarray:
        """
        Compute the oracle action values of the policy.

        Args:
            policy (ndarray): dS x dA policy matrix.
            base_policy (ndarray): dS x dA policy matrix.
            discount (float): discount factor
            er_coef (float): entropy regularization coefficient
            kl_coef (float): KL regularization coefficient

        Returns:
            values: dS x dA ndarray
        """

        values = np.zeros((self.dS, self.dA))  # SxA
        reward_matrix = self.reward_matrix  # SxA
        entropy = -np.sum(policy * np.log(policy + 1e-8), axis=-1, keepdims=True)  # Sx1
        reward_matrix = reward_matrix + er_coef * entropy  # SxA

        if base_policy is not None:
            kl = np.sum(
                policy * (np.log(policy + 1e-8) - np.log(base_policy + 1e-8)),
                axis=-1,
                keepdims=True,
            )  # Sx1
            reward_matrix = reward_matrix - kl_coef * kl  # SxA

        def backup(curr_q_val, policy):
            curr_v_val = np.sum(policy * curr_q_val, axis=-1)  # S
            prev_q = reward_matrix + discount * (
                self.transition_matrix * curr_v_val
            ).reshape(self.dS, self.dA)
            prev_q = np.asarray(prev_q)
            return prev_q

        for _ in range(self.horizon):
            values = backup(values, policy)  # SxA
        return values

    def calc_optimal_q(self, discount: float = 0.99) -> np.ndarray:
        """
        Compute the optimal action values

        Args:
            discount (float): discount factor

        Returns:
            values: dS x dA ndarray
        """

        values = np.zeros((self.dS, self.dA))  # SxA
        reward_matrix = self.reward_matrix  # SxA

        def backup(curr_q_val):
            curr_v_val = curr_q_val.max(axis=-1)  # S
            prev_q = reward_matrix + discount * (
                self.transition_matrix * curr_v_val
            ).reshape(self.dS, self.dA)
            prev_q = np.asarray(prev_q)
            return prev_q

        for _ in range(self.horizon):
            values = backup(values)  # SxA
        return values

    def calc_visit(self, policy: np.ndarray, discount: float = 0.99) -> np.ndarray:
        """
        Compute the discounted state and action frequency of a policy.
        You can compute the stationary distribution :math:`(1-\gamma)\sum^{H}_{t=0}\gamma^t P(s_t=s, a_t=a|\pi)` by calc_visit(policy, discount).sum(axis=0).

        Args:
            policy (ndarray): dS x dA policy matrix.
            discount (float): discount factor

        Returns:
            visit: TxSxA matrix
        """
        sa_visit = np.zeros((self.horizon, self.dS, self.dA))  # TxSxA
        s_visit_t = np.zeros((self.dS, 1))  # Sx1
        for (state, prob) in self.init_state_dist.items():
            s_visit_t[state] = prob

        norm_factor = 0.0
        for t in range(self.horizon):
            cur_discount = discount ** t
            norm_factor += cur_discount
            sa_visit_t = s_visit_t * policy  # SxA
            sa_visit[t, :, :] = cur_discount * sa_visit_t
            # sum-out (SA)S
            new_s_visit_t = (
                sa_visit_t.reshape(self.dS * self.dA) * self.transition_matrix
            )
            s_visit_t = np.expand_dims(new_s_visit_t, axis=1)
        visit = sa_visit / norm_factor
        return visit

    def count_visit(self, buffer) -> np.ndarray:
        """
        Count the number of state-action pairs in a buffer

        Args:
            buffer (cpprb.ReplayBuffer)

        Returns:
            SA: SxA matrix
        """
        samples = buffer.get_all_transitions()
        state, act = samples["state"].reshape(-1), samples["act"].reshape(-1)
        assert act.dtype == np.int32
        sa = np.zeros((self.dS, self.dA))  # SxA
        for s, a in zip(state.reshape(-1), act.reshape(-1)):
            sa[s, a] += 1
        return sa

    def calc_return(self, policy: np.ndarray) -> float:
        """
        Compute expected return of the policy

        Args:
            policy (ndarray): dS x dA policy matrix.

        Returns:
            ret (float)
        """

        q_values = np.zeros((self.dS, self.dA))  # SxA
        for t in reversed(range(1, self.horizon + 1)):
            # backup q values
            curr_vval = np.sum(policy * q_values, axis=-1)  # S
            prev_q = (self.transition_matrix * curr_vval).reshape(
                self.dS, self.dA
            )  # SxA
            q_values = np.asarray(prev_q + self.reward_matrix)  # SxA

        init_vval = np.sum(policy * q_values, axis=-1)  # S
        init_probs = np.zeros(self.dS)  # S
        for (state, prob) in self.init_state_dist.items():
            init_probs[state] = prob
        ret = np.sum(init_probs * init_vval)
        return ret

    def plot_S(
        self,
        values: np.ndarray,
        title: Optional[str] = None,
        ax: Optional[Axes] = None,
        cbar_ax: Optional[Axes] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        fontsize: Optional[int] = 10,
        **kwargs,
    ) -> None:
        """Plot values with shape [dS, ]

        Args:
            values (np.ndarray): Values to plot. Shape must be one-dimensional
            title (str, optional): Title of the axis. Defaults to None.
            ax (plt.axes.Axes, optional): Axis to plot. Defaults to None.
            cbar_ax (plt.axes.Axes, optional): Axis to plot the color bar. Defaults to None.
            vmin (Optional[float], optional): Minimum value of the color bar. Defaults to None.
            vmax (Optional[float], optional): Maximum value of the color bar. Defaults to None.
            fontsize (Optional[int], optional): Font size of the title and labels.
        """
        # values: dS
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        vmin = values.min() if vmin is None else vmin
        vmax = values.max() if vmax is None else vmax

        data = pd.DataFrame(values)
        sns.heatmap(
            data,
            ax=ax,
            cbar=True,
            cbar_ax=cbar_ax,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
        ax.set_title(title, fontsize=fontsize)
        ax.set_ylabel("State", fontsize=fontsize)
        ax.set_xlabel("Action", fontsize=fontsize)
