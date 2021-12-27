import gym

import shinrl as srl


class MockSolver(srl.BaseSolver):
    @staticmethod
    def make_mixins(env, config):
        return [MockSolver]

    def step(self):
        i = self.n_step
        return {"test1": i, "test2": i * 10}

    def evaluate(self):
        return {"test_eval": self.n_step}


class MockActMixIn:
    def eval_act(self, key, obs):
        act = self.env.action_space.sample()
        log_prob = 0.0
        return key, act, log_prob

    def explore_act(self, key, obs):
        act = self.env.action_space.sample()
        log_prob = 0.0
        return key, act, log_prob

    def step(self):
        return {}


def test_solver_id():
    from itertools import count

    env = gym.make("CartPole-v0")
    MockSolver._id = count(0)
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


def test_solver_run():
    env = gym.make("CartPole-v0")
    config = MockSolver.DefaultConfig(add_interval=5, steps_per_epoch=10)
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
    config = MockSolver.DefaultConfig(add_interval=5, steps_per_epoch=10)
    mixins = MockSolver.make_mixins(env, config)
    solver = MockSolver.factory(env, config, mixins)
    solver.run()
    solver.save(path)
    solver.load(path)
