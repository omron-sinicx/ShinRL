import chex
import gym
import jax.numpy as jnp

import shinrl as srl


def test_build_net_mixin():
    from shinrl.solvers.continuous_ddpg._build_net_mixin import BuildNetMixIn
    from shinrl.solvers.continuous_ddpg.config import DdpgConfig

    class MockSolver(srl.BaseSolver):
        @staticmethod
        def make_mixins():
            return [BuildNetMixIn, MockSolver]

        def step(self):
            pass

        def evaluate(self):
            pass

    env = gym.make("ShinPendulumContinuous-v0")
    config = DdpgConfig()
    mixins = MockSolver.make_mixins()
    solver = MockSolver.factory(env, config, mixins)
    obs = jnp.expand_dims(env.observation_space.sample(), axis=0)
    act = jnp.expand_dims(env.action_space.sample(), axis=0)
    output = solver.q_net.apply(solver.data["QNetParams"], obs, act)
    chex.assert_shape(output, (1, 1))

    obs = jnp.expand_dims(env.observation_space.sample(), axis=0)
    output = solver.pol_net.apply(solver.data["PolNetParams"], obs)
    chex.assert_shape(output, (1, 1))


def test_net_act_mixin():
    from shinrl.solvers.continuous_ddpg._build_net_mixin import BuildNetMixIn
    from shinrl.solvers.continuous_ddpg._net_act_mixin import NetActMixIn
    from shinrl.solvers.continuous_ddpg.config import DdpgConfig

    class MockSolver(srl.BaseSolver):
        @staticmethod
        def make_mixins():
            return [NetActMixIn, BuildNetMixIn, MockSolver]

        def step(self):
            pass

        def evaluate(self):
            pass

    env = gym.make("ShinPendulumContinuous-v0")
    config = DdpgConfig()
    mixins = MockSolver.make_mixins()
    solver = MockSolver.factory(env, config, mixins)
    obs = env.observation_space.sample()
    _, act, log_p = solver.explore_act(solver.key, obs)
    chex.assert_rank(act, 1)
    chex.assert_rank(log_p, 1)


def test_build_table_mixin():
    from shinrl.solvers.continuous_ddpg._build_net_mixin import BuildNetMixIn
    from shinrl.solvers.continuous_ddpg._build_table_mixin import BuildTableMixIn
    from shinrl.solvers.continuous_ddpg.config import DdpgConfig

    class MockSolver(srl.BaseSolver):
        @staticmethod
        def make_mixins():
            return [BuildTableMixIn, BuildNetMixIn, MockSolver]

        def step(self):
            pass

        def evaluate(self):
            pass

    env = gym.make("ShinPendulumContinuous-v0")
    config = DdpgConfig()
    mixins = MockSolver.make_mixins()
    solver = MockSolver.factory(env, config, mixins)
    solver.update_tb_data()
    chex.assert_rank(solver.data["Q"], 2)
    chex.assert_rank(solver.data["ExplorePolicy"], 2)
    chex.assert_rank(solver.data["EvaluatePolicy"], 2)
