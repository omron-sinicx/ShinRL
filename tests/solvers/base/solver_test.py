import distrax
import gym
import jax
import jax.numpy as jnp

from shinrl import (
    GymEvalMixIn,
    GymExploreMixIn,
    Pendulum,
    ShinEnv,
    ShinEvalMixIn,
    ShinExploreMixIn,
    Solver,
)


class MockSolver(Solver):
    @staticmethod
    def make_mixins(env, config):
        MixIn = TbInitMixIn if isinstance(env, ShinEnv) else NetInitMixIn
        return [MixIn, MockSolver]

    def step(self):
        i = self.n_step
        self.add_scalar("test1", i)
        self.add_scalar("test2", i * 10)


class TbInitMixIn(ShinExploreMixIn, ShinEvalMixIn):
    def initialize(self, env, config) -> None:
        super().initialize(env, config=config)
        key = jax.random.PRNGKey(0)
        pol = jax.random.uniform(key, shape=(env.mdp.dS, env.mdp.dA))
        pol /= pol.sum(axis=1, keepdims=True)
        self.tb_dict.set("ExploitPolicy", pol)
        self.tb_dict.set("ExplorePolicy", pol)


class NetInitMixIn(GymExploreMixIn, GymEvalMixIn):
    def initialize(self, env, config) -> None:
        super().initialize(env, config=config)

        def explore_act(key, obs):
            key, new_key = jax.random.split(key)
            dist = distrax.Categorical(logits=jnp.ones(self.env.action_space.n))
            act = dist.sample(seed=key)
            log_prob = dist.log_prob(act)
            return new_key, act, log_prob

        self.explore_act = jax.jit(explore_act)

        def exploit_act(key, obs):
            key, new_key = jax.random.split(key)
            dist = distrax.Categorical(logits=jnp.ones(self.env.action_space.n))
            act = dist.sample(seed=key)
            log_prob = dist.log_prob(act)
            return new_key, act, log_prob

        self.exploit_act = jax.jit(exploit_act)

    def make_explore_act(self):
        return self.explore_act

    def make_exploit_act(self):
        return self.exploit_act


def test_solver_id():
    env = gym.make("CartPole-v0")
    config = MockSolver.DefaultConfig()
    mixins = MockSolver.make_mixins(env, config)
    solver1 = MockSolver.factory(env, config, mixins)
    assert solver1.solver_id == "MixedSolver-0"
    assert solver1.env_id == 0
    env = gym.make("CartPole-v0")
    solver1.set_env(env)
    assert solver1.env_id == 1

    mixins = MockSolver.make_mixins(env, config)
    solver2 = MockSolver.factory(env, config, mixins)
    assert solver2.solver_id == "MixedSolver-1"


def test_mixin():
    env = Pendulum()
    config = Solver.DefaultConfig(eval_trials=2)
    mixins = MockSolver.make_mixins(env, config)
    solver = MockSolver.factory(env, config, mixins)
    res = solver.evaluate()
    assert type(res["Return"]) is float

    env = gym.make("CartPole-v0")
    mixins = MockSolver.make_mixins(env, config)
    solver = MockSolver.factory(env, config, mixins)
    res = solver.evaluate()
    assert type(res["Return"]) is float


def test_solver_run():
    env = Pendulum()
    config = Solver.DefaultConfig(add_interval=5, steps_per_epoch=10)
    mixins = MockSolver.make_mixins(env, config)
    solver = MockSolver.factory(env, config, mixins)
    solver.run()

    env = gym.make("CartPole-v0")
    mixins = MockSolver.make_mixins(env, config)
    solver = MockSolver.factory(env, config, mixins)
    solver.run()
    assert solver.n_step == 10
    assert solver.scalars["test1"] == {"x": [0, 5], "y": [0, 5]}

    solver.config.steps_per_epoch = 20
    solver.run()
    assert solver.n_step == 30
    assert solver.scalars["test2"] == {
        "x": [0, 5, 10, 15, 20, 25],
        "y": [0, 50, 100, 150, 200, 250],
    }


def test_save_load_solver(tmpdir):
    path = tmpdir.mkdir("tmp")

    env = gym.make("CartPole-v0")
    config = Solver.DefaultConfig(add_interval=5, steps_per_epoch=10)
    mixins = MockSolver.make_mixins(env, config)
    solver = MockSolver.factory(env, config, mixins)
    solver.run()
    solver.save(path)
    solver.load(path)
