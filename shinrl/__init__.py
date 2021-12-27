from gym.envs.registration import register

# Common mathmetical (jitted) functions & classes.
from ._calc.backup_dp import (
    calc_optimal_q,
    calc_q,
    calc_return,
    calc_visit,
    double_backup_dp,
    expected_backup_dp,
    munchausen_backup_dp,
    optimal_backup_dp,
    soft_expected_backup_dp,
)
from ._calc.backup_rl import (
    double_backup_rl,
    expected_backup_rl,
    munchausen_backup_rl,
    optimal_backup_rl,
    soft_expected_backup_rl,
)
from ._calc.build_net import (
    build_obs_act_forward_conv,
    build_obs_act_forward_fc,
    build_obs_forward_conv,
    build_obs_forward_fc,
)
from ._calc.build_net_act import (
    build_continuous_greedy_net_act,
    build_discrete_greedy_net_act,
    build_eps_greedy_net_act,
    build_fixed_scale_normal_net_act,
    build_softmax_net_act,
)
from ._calc.collect_samples import ACT_FN, Sample, collect_samples, make_replay_buffer
from ._calc.draw import draw_circle, line_aa
from ._calc.epsilon_greedy import calc_eps
from ._calc.loss import cross_entropy_loss, huber_loss, kl_loss, l2_loss
from ._calc.moving_average import calc_ma
from ._calc.sparse import SparseMat, sp_mul, sp_mul_t

# Common useful functions & classes.
from ._utils.config import Config
from ._utils.log import add_logfile_handler, initialize_log_style
from ._utils.minatar import make_minatar
from ._utils.scalars import Scalars
from ._utils.wrapper import NormalizeActionWrapper

# Implemented environments with access to the *oracle* quantities.
from .envs.base.config import EnvConfig
from .envs.base.env import ShinEnv
from .envs.base.mdp import MDP
from .envs.cartpole.env import CartPole
from .envs.maze.env import Maze
from .envs.mountaincar.env import MountainCar
from .envs.pendulum.env import Pendulum
from .solvers.base.base_mixin import (
    BaseGymEvalMixIn,
    BaseGymExploreMixIn,
    BaseShinEvalMixIn,
    BaseShinExploreMixIn,
)

# Implemented environments with access to the *oracle* quantities.
from .solvers.base.config import SolverConfig
from .solvers.base.history import DataDict, History, prepare_history_dir
from .solvers.base.solver import BaseSolver
from .solvers.continuous_ddpg.solver import ContinuousDdpgSolver
from .solvers.discrete_pi.solver import DiscretePiSolver
from .solvers.discrete_vi.solver import DiscreteViSolver

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
