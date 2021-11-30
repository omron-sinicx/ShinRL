from copy import deepcopy
from typing import Optional, Union

import gym
import numpy as np
import torch
import torch.nn.functional as F
from scipy import special
from torch import Tensor

from shinrl import utils
from shinrl.solvers import BaseSolver
from shinrl.utils.tb import get_tb_act

from .core.config import ViConfig
from .core.net import make_net_opt
from .core.policy import get_net_act, get_q_to_pol

Array = Union[Tensor, np.ndarray]


def to_np(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def to_tnsr(nd_array: np.ndarray, device: str) -> Tensor:
    return torch.tensor(nd_array, dtype=torch.float32, device=device)


def take(input: Array, index: Array, is_nn: bool) -> Array:
    """A wrapper function of np.take_along_axis and torch.take_along_dim.

    Returns:
        A vector of shape == (input.shape[0], )
    """
    if is_nn:
        if len(index.shape) == len(input.shape) - 1:
            index = index.unsqueeze(-1)
        return torch.take_along_dim(input, index, dim=-1).squeeze(-1)
    else:
        if len(index.shape) == len(input.shape) - 1:
            index = np.expand_dims(index, axis=-1)
        token = np.take_along_axis(input, index, axis=-1)
        return np.squeeze(token, axis=-1)


class ViSolver(BaseSolver):
    DefaultConfig = ViConfig

    @staticmethod
    def factory(config: ViConfig):
        if config.approx == "tabular" and config.explore == "oracle":
            return ViSolverTabularDP()
        elif config.approx == "tabular" and config.explore != "oracle":
            return ViSolverTabularRL()
        elif config.approx == "nn" and config.explore == "oracle":
            return ViSolverDeepDP()
        elif config.approx == "nn" and config.explore != "oracle":
            return ViSolverDeepRL()

    @property
    def config(self) -> ViConfig:
        return self._config

    def initialize(self, env: gym.Env, config: Optional[ViConfig] = None) -> None:
        super().initialize(env, config)
        if self.is_shin_env:
            self._tbs_init()
        else:
            assert (
                self.config.approx != "tabular" and self.config.explore != "oracle"
            ), "'tabular' and 'oracle' configs are allowed only with ShinEnv"

        if self.config.approx == "nn":
            self._nets_init()

    def _tbs_init(self) -> None:
        def map_nn_to_q() -> np.ndarray:
            return to_np(self.history.nets["Q"](self.all_obss))

        def map_q_to_exploit() -> np.ndarray:
            q = self.history.tbs["Q"]
            return get_q_to_pol(self.config.exploit)(q, self.config)

        def map_q_to_explore() -> np.ndarray:
            q = self.history.tbs["Q"]
            return get_q_to_pol(self.config.explore)(q, self.config, self.history.step)

        if self.config.approx == "nn":
            self.history.set_tb("Q", map_nn_to_q)
        else:
            self.history.set_tb("Q", np.zeros((self.dS, self.dA)))
        self.history.set_tb("ExploitPolicy", map_q_to_exploit)
        self.history.set_tb("ExplorePolicy", map_q_to_explore)

        self.all_obss = to_tnsr(self.env.all_observations, self.config.device)  # SxO

    def _nets_init(self) -> None:
        q_net, self.q_opt = make_net_opt(self.env, self.config)
        self.history.set_net("Q", q_net)
        self.history.set_net("TargetQ", deepcopy(q_net))

    @torch.no_grad()
    def evaluate(self) -> None:
        if self.is_shin_env:
            self._shin_evaluate()
        else:
            self._gym_evaluate()

    def _shin_evaluate(self) -> None:
        pol = self.history.tbs["ExploitPolicy"]
        q_star = self.env.calc_optimal_q(discount=self.config.discount)
        q = self.env.calc_q(pol)
        self.history.add_scalar("OptimalityGap", np.abs(q_star - q).max())
        self.history.add_scalar("Return", self.env.calc_return(pol))

    def _gym_evaluate(self) -> None:
        samples = utils.collect_samples(
            env=self.env,
            get_act=get_net_act(self.config.exploit),
            num_episodes=self.config.eval_trials,
            get_act_args={
                "net": self.history.nets["Q"],
                "step": self.history.step,
                "config": self.config,
            },
        )
        self.history.add_scalar("Return", samples.rew.sum() / self.config.eval_trials)

    @torch.no_grad()
    def _calc_q_target(
        self,
        q: Array,
        next_q: Array,
        rew: Array,
        dones: Array,
        act: Optional[Array] = None,
        next_double_q: Optional[Array] = None,
    ):
        is_nn = isinstance(q, Tensor)
        is_dp = act is None
        f = F if is_nn else special
        double_q = next_double_q is not None

        if self.config.er_coef == self.config.kl_coef == 0.0:
            next_arg_max = next_double_q.argmax(-1) if double_q else next_q.argmax(-1)
            next_max = take(next_q, next_arg_max, is_nn)
            munchausen = 0.0
        else:
            # Munchausen Value Iteration
            tau = self.config.kl_coef + self.config.er_coef
            alpha = self.config.kl_coef / tau

            log_cur_pol = f.log_softmax(q / tau, -1)
            if not is_dp:
                log_cur_pol = take(log_cur_pol, act, is_nn)
            munchausen = alpha * (tau * log_cur_pol).clip(
                min=self.config.logp_clip, max=1
            )
            next_pol = f.softmax(next_q / tau, -1)
            next_logpol = f.log_softmax(next_q / tau, -1)
            next_max = (next_pol * (next_q - tau * next_logpol)).sum(-1)

        if is_dp:
            next_max = self.env.transition_matrix * next_max
            next_max = next_max.reshape(self.dS, self.dA)

        target = munchausen + rew + self.config.discount * next_max * (~dones)
        target = target if is_nn else np.asarray(target)
        if self.config.noise_scale > 0:
            noise = self._make_noise(rew)
            target += noise * self.config.noise_scale
        return target

    def _make_noise(self, rew) -> Array:
        is_nn = isinstance(rew, Tensor)
        noise = torch.randn_like(rew) if is_nn else np.random.randn(*rew.shape)
        return noise

    def _update_nets(
        self, inputs: Tensor, targets: Tensor, acts: Tensor = None
    ) -> None:
        pred = self.history.nets["Q"](inputs)
        if acts is not None:
            pred = pred.gather(1, acts.reshape(-1, 1)).squeeze()
        loss = getattr(F, self.config.loss_fn)(pred, targets)
        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()
        self.history.add_scalar("Loss", loss.detach().cpu().item())

        if (self.history.step + 1) % self.config.target_update_interval == 0:
            self.history.nets["TargetQ"].load_state_dict(
                self.history.nets["Q"].state_dict()
            )


class ViSolverTabularDP(ViSolver):
    def step(self) -> None:
        q = next_q = self.history.tbs["Q"]  # SxA
        rew, dones = self.env.reward_matrix, np.array([0], dtype=bool)
        new_tb = self._calc_q_target(q, next_q, rew, dones)
        self.history.set_tb("Q", new_tb)


class ViSolverDeepDP(ViSolver):
    def step(self) -> None:
        q = to_np(self.history.nets["TargetQ"](self.all_obss))
        double_q = (
            to_np(self.history.nets["Q"](self.all_obss))
            if self.config.use_double_q
            else None
        )
        rew, dones = self.env.reward_matrix, np.array([0], dtype=bool)
        target = self._calc_q_target(q, q, rew, dones, next_double_q=double_q)
        target = to_tnsr(target, self.config.device)
        self._update_nets(self.all_obss, target, None)


class ViSolverTabularRL(ViSolver):
    def step(self) -> None:
        explore = self.history.tbs["ExplorePolicy"]
        samples = utils.collect_samples(
            env=self.env,
            num_samples=self.config.num_samples,
            get_act=get_tb_act,
            get_act_args={"policy": explore},
        )
        s, a, ns = samples.state, samples.act, samples.next_state
        q = self.history.tbs["Q"][s]  # BxA
        next_q = self.history.tbs["Q"][ns]  # BxA
        rew, done = samples.rew, samples.done * (~samples.timeout)

        # update tables
        targets = self._calc_q_target(q, next_q, rew, done, a)
        tb = self.history.tbs["Q"]
        lr = self.config.lr
        for ss, aa, target in zip(s, a, targets):
            tb[ss, aa] = (1 - lr) * tb[ss, aa] + lr * target
        self.history.set_tb("Q", tb)


class ViSolverDeepRL(ViSolver):
    def initialize(self, env: gym.Env, config: Optional[ViConfig] = None) -> None:
        super().initialize(env, config)
        self.buffer = utils.make_replay_buffer(self.env, self.config.buffer_size)

    def step(self) -> None:
        samples = utils.collect_samples(
            env=self.env,
            num_samples=self.config.num_samples,
            get_act=get_net_act(self.config.explore),
            buffer=self.buffer,
            get_act_args={
                "net": self.history.nets["Q"],
                "step": self.history.step,
                "config": self.config,
            },
        )
        samples = utils.Samples(**self.buffer.sample(self.config.minibatch_size))
        samples = samples.np_to_tnsr(device=self.config.device)

        # update networks
        q = self.history.nets["TargetQ"](samples.obs)  # BxA
        next_q = self.history.nets["TargetQ"](samples.next_obs)  # BxA
        next_double_q = (
            self.history.nets["Q"](samples.next_obs).detach()
            if self.config.use_double_q
            else None
        )
        rew, done, act = samples.rew, samples.done * (~samples.timeout), samples.act
        targets = self._calc_q_target(q, next_q, rew, done, act, next_double_q)
        self._update_nets(samples.obs, targets, act)
