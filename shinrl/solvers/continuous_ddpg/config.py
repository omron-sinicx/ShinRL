"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
import enum
from enum import auto
from inspect import getmembers
from typing import ClassVar, Type

import chex
import jax
import optax

import shinrl as srl
from shinrl import SolverConfig


class EXPLORE(enum.IntEnum):
    oracle = auto()
    normal = auto()


class EVALUATE(enum.IntEnum):
    greedy = auto()


class APPROX(enum.IntEnum):
    nn = auto()


ACTIVATION = enum.IntEnum(
    "ACTIVATION",
    {fnc[0]: enum.auto() for fnc in getmembers(jax.nn) if callable(fnc[1])},
)


OPTIMIZER = enum.IntEnum(
    "OPTIMIZER",
    {fnc[0]: enum.auto() for fnc in getmembers(optax._src.alias) if callable(fnc[1])},
)


LOSS = enum.IntEnum(
    "LOSS",
    {fnc[0]: enum.auto() for fnc in getmembers(srl._calc.loss) if callable(fnc[1])},
)


@chex.dataclass
class DdpgConfig(SolverConfig):
    """Config for DdpgSolver.

    Args:
        explore (EXPLORE):
            Type of the policy in exploration. The solver uses all the state-action pairs if 'oracle'.
        evaluate (EVALUATE): Type of the policy in evaluation.
        approx (APPROX): Type of the function approximation.

        eps_end (float): Epsilon value at the end of the eps-greedy exploration.
        eps_warmup (int): Epsilon value is set 1.0 until 'eps_warmup'.
        eps_decay (int): Interval to decrease the epsilon value.
        softmax_tmp (float): temperature parameter for softmax.

        pol_lr (float): Learning rate of the pol.
        q_lr (float): Learning rate of the q.
        num_samples (int): Number of samples to collect when explore != 'oracle'.
        batch_size (int): Size of the minibatch.
        buffer_size (int): Replay buffer capacity.

        hidden (int): Size of the linear layer.
        depth (int): Depth of the linear layer.
        polyak_rate (float): Speed with which to update the target network towards the online network.
        activation (str): Activation function.
        optimizer (str): Optimizer for nn.
        q_loss_fn (str): Loss function for q.
        pol_loss_fn (str): Loss function for policy. Automatically changed in make_mixin.
    """

    # class variables
    EXPLORE: ClassVar[Type[EXPLORE]] = EXPLORE
    EVALUATE: ClassVar[Type[EVALUATE]] = EVALUATE
    APPROX: ClassVar[Type[APPROX]] = APPROX
    ACTIVATION: ClassVar[Type[ACTIVATION]] = ACTIVATION
    LOSS: ClassVar[Type[LOSS]] = LOSS
    OPTIMIZER: ClassVar[Type[OPTIMIZER]] = OPTIMIZER

    explore: EXPLORE = EXPLORE.oracle
    evaluate: EVALUATE = EVALUATE.greedy
    approx: APPROX = APPROX.nn

    # algorithm configs
    pol_lr: float = 1e-4
    q_lr: float = 1e-3
    num_samples: int = 4
    buffer_size: int = int(1e6)
    batch_size: int = 32
    replay_start_size: int = 5000

    # policy settings
    normal_scale: float = 0.3

    # network configs
    hidden: int = 128
    depth: int = 2
    polyak_rate: float = 0.01
    activation: ACTIVATION = ACTIVATION.relu
    optimizer: OPTIMIZER = OPTIMIZER.adam
    q_loss_fn: LOSS = LOSS.l2_loss
