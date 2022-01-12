from gym.envs.registration import register  # NOQA isort:skip

# Common mathmetical (jitted) functions & classes.
from ._calc.backup_dp import calc_optimal_q  # NOQA isort:skip
from ._calc.backup_dp import calc_q  # NOQA isort:skip
from ._calc.backup_dp import calc_return  # NOQA isort:skip
from ._calc.backup_dp import calc_visit  # NOQA isort:skip
from ._calc.backup_dp import double_backup_dp  # NOQA isort:skip
from ._calc.backup_dp import expected_backup_dp  # NOQA isort:skip
from ._calc.backup_dp import munchausen_backup_dp  # NOQA isort:skip
from ._calc.backup_dp import optimal_backup_dp  # NOQA isort:skip
from ._calc.backup_dp import soft_expected_backup_dp  # NOQA isort:skip
from ._calc.backup_rl import double_backup_rl  # NOQA isort:skip
from ._calc.backup_rl import expected_backup_rl  # NOQA isort:skip
from ._calc.backup_rl import munchausen_backup_rl  # NOQA isort:skip
from ._calc.backup_rl import optimal_backup_rl  # NOQA isort:skip
from ._calc.backup_rl import soft_expected_backup_rl  # NOQA isort:skip
from ._calc.build_net import build_obs_act_forward_conv  # NOQA isort:skip
from ._calc.build_net import build_obs_act_forward_fc  # NOQA isort:skip
from ._calc.build_net import build_obs_forward_conv  # NOQA isort:skip
from ._calc.build_net import build_obs_forward_fc  # NOQA isort:skip
from ._calc.build_net_act import build_discrete_greedy_net_act  # NOQA isort:skip
from ._calc.build_net_act import build_eps_greedy_net_act  # NOQA isort:skip
from ._calc.build_net_act import build_normal_diagonal_net_act  # NOQA isort:skip
from ._calc.build_net_act import build_softmax_net_act  # NOQA isort:skip
from ._calc.collect_samples import ACT_FN  # NOQA isort:skip
from ._calc.collect_samples import Sample  # NOQA isort:skip
from ._calc.collect_samples import collect_samples  # NOQA isort:skip
from ._calc.collect_samples import make_replay_buffer  # NOQA isort:skip
from ._calc.distributions import SquashedNormal  # NOQA isort:skip
from ._calc.draw import draw_circle, line_aa  # NOQA isort:skip
from ._calc.epsilon_greedy import calc_eps  # NOQA isort:skip
from ._calc.loss import (  # NOQA isort:skip
    cross_entropy_loss,
    huber_loss,
    kl_loss,
    l2_loss,
)

from ._calc.moving_average import calc_ma  # NOQA isort:skip
from ._calc.sparse import SparseMat, sp_mul, sp_mul_t  # NOQA isort:skip

# Common useful functions & classes.
from ._utils.config import Config  # NOQA isort:skip
from ._utils.log import add_logfile_handler, initialize_log_style  # NOQA isort:skip
from ._utils.minatar import make_minatar  # NOQA isort:skip
from ._utils.scalars import Scalars  # NOQA isort:skip
from ._utils.wrapper import NormalizeActionWrapper  # NOQA isort:skip

# Implemented environments with access to the *oracle* quantities.
from .envs.base.config import EnvConfig  # NOQA isort:skip
from .envs.base.env import ShinEnv  # NOQA isort:skip
from .envs.base.mdp import MDP  # NOQA isort:skip
from .envs.cartpole.env import CartPole  # NOQA isort:skip
from .envs.maze.env import Maze  # NOQA isort:skip
from .envs.mountaincar.env import MountainCar  # NOQA isort:skip
from .envs.pendulum.env import Pendulum  # NOQA isort:skip
from .solvers.base.base_mixin import BaseGymEvalMixIn  # NOQA isort:skip
from .solvers.base.base_mixin import BaseGymExploreMixIn  # NOQA isort:skip
from .solvers.base.base_mixin import BaseShinEvalMixIn  # NOQA isort:skip
from .solvers.base.base_mixin import BaseShinExploreMixIn  # NOQA isort:skip

# Implemented environments with access to the *oracle* quantities.
from .solvers.base.config import SolverConfig  # NOQA isort:skip
from .solvers.base.history import (  # NOQA isort:skip
    DataDict,
    History,
    prepare_history_dir,
)

from .solvers.base.solver import BaseSolver  # NOQA isort:skip
from .solvers.continuous_ddpg.solver import ContinuousDdpgSolver  # NOQA isort:skip
from .solvers.discrete_pi.solver import DiscretePiSolver  # NOQA isort:skip
from .solvers.discrete_vi.solver import DiscreteViSolver  # NOQA isort:skip

initialize_log_style()


# ----- register envs -----


register(
    id="ShinMaze-v0",
    entry_point="shinrl:Maze",
    kwargs={"config": Maze.DefaultConfig()},
)

register(
    id="ShinPendulum-v0",
    entry_point="shinrl:Pendulum",
    kwargs={"config": Pendulum.DefaultConfig()},
)

register(
    id="ShinPendulumContinuous-v0",
    entry_point="shinrl:Pendulum",
    kwargs={"config": Pendulum.DefaultConfig(act_mode="continuous", dA=50)},
)

register(
    id="ShinPendulumImage-v0",
    entry_point="shinrl:Pendulum",
    kwargs={"config": Pendulum.DefaultConfig(obs_mode="image")},
)

register(
    id="ShinPendulumImageContinuous-v0",
    entry_point="shinrl:Pendulum",
    kwargs={
        "config": Pendulum.DefaultConfig(act_mode="continuous", dA=50, obs_mode="image")
    },
)

register(
    id="ShinMountainCar-v0",
    entry_point="shinrl:MountainCar",
    kwargs={"config": MountainCar.DefaultConfig()},
)

register(
    id="ShinMountainCarContinuous-v0",
    entry_point="shinrl:MountainCar",
    kwargs={"config": MountainCar.DefaultConfig(act_mode="continuous", dA=50)},
)

register(
    id="ShinMountainCarImage-v0",
    entry_point="shinrl:MountainCar",
    kwargs={"config": MountainCar.DefaultConfig(obs_mode="image")},
)

register(
    id="ShinMountainCarImageContinuous-v0",
    entry_point="shinrl:MountainCar",
    kwargs={
        "config": MountainCar.DefaultConfig(
            act_mode="continuous", dA=50, obs_mode="image"
        )
    },
)

register(
    id="ShinCartPole-v0",
    entry_point="shinrl:CartPole",
    kwargs={"config": CartPole.DefaultConfig()},
)


register(
    id="ShinCartPoleContinuous-v0",
    entry_point="shinrl:CartPole",
    kwargs={"config": CartPole.DefaultConfig(act_mode="continuous", dA=50)},
)
