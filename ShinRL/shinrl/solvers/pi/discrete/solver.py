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

from .core.config import PiConfig
from .core.net import make_net_opt
from .core.policy import get_net_act, get_q_to_pol

Array = Union[Tensor, np.ndarray]


def to_np(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def to_tnsr(array: np.ndarray, device: str) -> Tensor:
    return torch.tensor(array, dtype=torch.float32, device=device)


class PiSolver(BaseSolver):
    DefaultConfig = PiConfig

    def factory(config: PiConfig):
        if config.approx == "tabular" and config.explore == "oracle":
            return PiSolverTabularDP()
        elif config.approx == "tabular" and config.explore != "oracle":
            return PiSolverTabularRL()
        elif config.approx == "nn" and config.explore == "oracle":
            return PiSolverDeepDP()
        elif config.approx == "nn" and config.explore != "oracle":
            return PiSolverDeepRL()

    @property
    def config(self) -> PiConfig:
        return self._config

    def initialize(self, env: gym.Env, config: Optional[PiConfig] = None) -> None:
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

        def map_nn_to_pol() -> np.ndarray:
            return to_np(self.history.nets["Policy"](self.all_obss))

        def map_pol_to_exploit() -> np.ndarray:
            pol = self.history.tbs["Policy"]
            return get_q_to_pol(self.config.exploit)(pol, self.config)

        def map_pol_to_explore() -> np.ndarray:
            pol = self.history.tbs["Policy"]
            return get_q_to_pol(self.config.explore)(
                pol, self.config, self.history.step
            )

        if self.config.approx == "nn":
            self.history.set_tb("Q", map_nn_to_q)
            self.history.set_tb("Policy", map_nn_to_pol)
        else:
            self.history.set_tb("Q", np.zeros((self.dS, self.dA)))
            self.history.set_tb("Policy", np.ones((self.dS, self.dA)) / self.dA)
        self.history.set_tb("ExplorePolicy", map_pol_to_explore)
        self.history.set_tb("ExploitPolicy", map_pol_to_exploit)

        self.all_obss = to_tnsr(self.env.all_observations, self.config.device)  # SxO

    def _nets_init(self) -> None:
        q_net, self.q_opt = make_net_opt(self.env, self.config.critic_lr, self.config)
        self.history.set_net("Q", q_net)
        self.history.set_net("TargetQ", deepcopy(q_net))
        pol_net, self.pol_opt = make_net_opt(
            self.env, self.config.actor_lr, self.config
        )
        self.history.set_net("Policy", pol_net)

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
        ret = self.env.calc_return(pol)
        self.history.add_scalar("Return", ret)

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
        ret = samples.rew.sum() / self.config.eval_trials
        self.history.add_scalar("Return", ret)

    @torch.no_grad()
    def _calc_pol_target(self, q: Array):
        is_nn = isinstance(q, Tensor)
        f = F if is_nn else special
        lib = torch if is_nn else np

        if self.config.noise_scale > 0:
            noise = torch.randn_like(q) if is_nn else np.random.randn(*q.shape)
            q += noise * self.config.noise_scale
        if self.config.er_coef == 0.0:
            # Policy Iteration
            target_pol = lib.zeros_like(q)
            target_pol[lib.arange(q.shape[0]), q.argmax(-1)] = 1.0
            log_pol = lib.zeros_like(q)
        else:
            # Soft Policy Iteration
            target_pol = f.softmax(q / self.config.er_coef, -1)
            log_pol = f.log_softmax(q / self.config.er_coef, -1)
        return target_pol, log_pol

    @torch.no_grad()
    def _calc_q_target(
        self,
        next_pol: Array,
        next_log_pol: Array,
        q: Array,
        next_q: Array,
        rew: Array,
        dones: Array,
    ):
        is_nn = isinstance(q, Tensor)

        # policy evaluation
        next_v = (next_pol * (next_q - self.config.er_coef * next_log_pol)).sum(-1)
        if self.config.explore == "oracle":
            next_v = self.env.transition_matrix * next_v
            next_v = next_v.reshape(self.dS, self.dA)
        target = rew + self.config.discount * next_v * (~dones)
        target = target if is_nn else np.asarray(target)
        if self.config.noise_scale > 0:
            noise = self._make_noise(rew)
            target += noise * self.config.noise_scale
        return target

    def _make_noise(self, rew) -> Array:
        is_nn = isinstance(rew, Tensor)
        noise = torch.randn_like(rew) if is_nn else np.random.randn(*rew.shape)
        return noise

    def _update_pol_nets(self, inputs: Tensor, targ_log_pol: Tensor) -> None:
        logits = self.history.nets["Policy"](inputs)
        if self.config.er_coef == 0.0:
            loss = F.cross_entropy(logits, targ_log_pol.argmax(dim=-1))
        else:
            log_pol = F.log_softmax(logits, -1)
            pol = F.softmax(logits, -1)
            loss = (pol * (log_pol - targ_log_pol)).sum(-1, keepdim=True).mean()

        self.pol_opt.zero_grad()
        loss.backward()
        self.pol_opt.step()
        self.history.add_scalar("Loss", loss.detach().cpu().item())

    def _update_q_nets(self, inputs: Tensor, targets: Tensor, acts: Tensor) -> None:
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


class PiSolverTabularDP(PiSolver):
    def step(self) -> None:
        q = next_q = self.history.tbs["Q"]  # SxA
        rew, dones = self.env.reward_matrix, np.array([0], dtype=bool)

        # policy improvement step
        pol, log_pol = self._calc_pol_target(q)
        self.history.set_tb("Policy", pol)

        # policy evaluation step
        new_q = self._calc_q_target(pol, log_pol, q, next_q, rew, dones)
        self.history.set_tb("Q", new_q)


class PiSolverDeepDP(PiSolver):
    def step(self) -> None:
        q = next_q = self.history.tbs["Q"]  # SxA

        # update policy network
        pol, log_pol = self._calc_pol_target(q)
        log_pol = to_tnsr(log_pol, self.config.device)
        self._update_pol_nets(self.all_obss, log_pol)
        logits = self.history.nets["Policy"](self.all_obss)
        log_pol = to_np(F.log_softmax(logits, -1))  # SxA
        pol = to_np(F.softmax(logits, -1))  # SxA
        self.history.set_tb("Policy", pol)

        # update Q network
        q = next_q = self.history.tbs["Q"]  # SxA
        rew, dones = self.env.reward_matrix, np.array([0], dtype=bool)
        target = self._calc_q_target(pol, log_pol, q, next_q, rew, dones)
        target = to_tnsr(target, self.config.device)
        self._update_q_nets(self.all_obss, target, None)
        new_q = to_np(self.history.nets["Q"](self.all_obss))
        self.history.set_tb("Q", new_q)


class PiSolverTabularRL(PiSolver):
    def step(self) -> None:
        # update policy
        q = self.history.tbs["Q"]  # BxA
        pol, log_pol = self._calc_pol_target(q)  # BxA
        self.history.set_tb("Policy", pol)

        # update q
        samples = utils.collect_samples(
            env=self.env,
            num_samples=self.config.num_samples,
            get_act=get_tb_act,
            get_act_args={"policy": self.history.tbs["ExplorePolicy"]},
        )
        s, a, ns = samples.state, samples.act, samples.next_state
        q = self.history.tbs["Q"][s]  # BxA
        next_q = self.history.tbs["Q"][ns]  # BxA
        next_pol = pol[ns]  # BxA
        next_log_pol = log_pol[ns]  # B
        rew, done = samples.rew, samples.done * (~samples.timeout)

        targets = self._calc_q_target(next_pol, next_log_pol, q, next_q, rew, done)
        q_tb = self.history.tbs["Q"]
        lr = self.config.critic_lr
        for ss, aa, target in zip(s, a, targets):
            q_tb[ss, aa] = (1 - lr) * q_tb[ss, aa] + lr * target
        self.history.set_tb("Q", q_tb)


class PiSolverDeepRL(PiSolver):
    def initialize(self, env: gym.Env, config: Optional[PiConfig] = None) -> None:
        super().initialize(env, config)
        self.buffer = utils.make_replay_buffer(self.env, self.config.buffer_size)

    def step(self) -> None:
        samples = utils.collect_samples(
            env=self.env,
            num_samples=self.config.num_samples,
            get_act=get_net_act(self.config.explore),
            buffer=self.buffer,
            get_act_args={
                "net": self.history.nets["Policy"],
                "step": self.history.step,
                "config": self.config,
            },
        )
        samples = utils.Samples(**self.buffer.sample(self.config.minibatch_size))
        samples = samples.np_to_tnsr(device=self.config.device)

        # update policy networks
        q = self.history.nets["TargetQ"](samples.obs)  # BxA
        _, log_pol = self._calc_pol_target(q)  # BxA
        self._update_pol_nets(samples.obs, log_pol)

        # update q networks
        next_logits = self.history.nets["Policy"](samples.next_obs)
        next_pol = F.softmax(next_logits, -1)
        next_log_pol = F.log_softmax(next_logits, -1)
        next_q = self.history.nets["TargetQ"](samples.next_obs)  # BxA
        rew, done, act = samples.rew, samples.done * (~samples.timeout), samples.act
        targets = self._calc_q_target(next_pol, next_log_pol, q, next_q, rew, done)
        self._update_q_nets(samples.obs, targets, act)

        if self.is_shin_env:
            # update tables
            q = to_np(self.history.nets["Q"](self.all_obss))
            self.history.set_tb("Q", q)
            pol = to_np(self.history.nets["Policy"](self.all_obss))
            self.history.set_tb("Policy", pol)
