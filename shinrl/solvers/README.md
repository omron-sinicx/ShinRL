# Solver Tutorial

Try the following tutorial at: [experiments/SolverTutorial.ipynb](../../experiments/SolverTutorial.ipynb).

* 1. [Create Custom Solver](#CreatecustomSolver)
	* 1.1. [ Create Config](#CreateConfig)
	* 1.2. [ Create Solver](#CreateSolver)
		* 1.2.1. [solver.scalars](#solver.scalars)
		* 1.2.2. [solver.data](#solver.data)
* 2. [Useful MixIns](#UsefulMixIns)
	* 2.1. [GymEnv Solver Example](#GymEnvSolverExample)
	* 2.2. [ShinEnv Solver Example](#ShinEnvSolverExample)
	* 2.3. [GymEnv & ShinEnv Solver Example](#GymEnvShinEnvSolverExample)


##  1. <a name='CreateCustomSolver'></a>Create custom Solver

This tutorial demonstrates how to create a custom solver.
We are going to implement a very simple solver that tries to maximize the one-step reward.

You need to implement two classes, 
1. a config class inheriting `shinrl.SolverConfig` and 
2. a solver class inheriting `shinrl.BaseSolver`.


###  1.1. <a name='CreateConfig'></a> Create Config

The config class is a dataclass inheriting `shinrl.SolverConfig`.
It holds hyperparameters of a solver.

```python
@chex.dataclass
class ExampleConfig(shinrl.SolverConfig):
    seed: int = 0
    discount: float = 0.99
    eval_trials: int = 10
    eval_interval: int = 100
    add_interval: int = 100
    steps_per_epoch: int = 1000
```

###  1.2. <a name='CreateSolver'></a> Create Solver

The main solver class must inherit `shinrl.BaseSolver`.
You need to implement three functions (See details in [shinrl/solvers/base/solver.py](../shinrl/solvers/base/solver.py)):

* **make_mixins** (staticmethod): Make a list of mixins from env and config. A solver is instantiated by mixing generated mixins.
* **evaluate** (function): Evaluate the solver and return the dict of results. Called every self.config.eval_interval steps.
* **step** (function): Execute the solver by one step and return the dict of results.

The following code implements `evaluate` and `step` functions through mixins:


```python
class ExampleStepMixIn:
    def initialize(self, env, config=None) -> None:
        super().initialize(env, config)
        
        dA = self.env.action_space.n
        policy = jnp.ones(dA)
        policy = policy / policy.sum()
        
        # Any jittable object (e.g., network parameters, Q-table, etc.)
        # should be stored in this `solver.data` dictionary
        self.data["Policy"] = policy

    def step(self):
        policy = self.data["Policy"]
        dist = distrax.Greedy(policy)
        act = dist.sample(seed=self.key).item()
        self.env.obs, rew, done, _ = self.env.step(act)
        
        # Update policy
        policy = policy.at[act].add(rew)
        self.data["Policy"] = policy / policy.sum()
        
        # Return any scalar data you want to record
        return {"Rew": rew}

    
class ExampleEvalMixIn:
    def initialize(self, env, config=None) -> None:
        super().initialize(env, config)
        self._eval_env = deepcopy(self.env)

    def evaluate(self):
        self._eval_env.reset()
        
        policy = self.data["Policy"]
        dist = distrax.Greedy(policy)
        ret = 0
        done = False
        while not done:
            act = dist.sample(seed=self.key).item()       
            self._eval_env.obs, rew, done, _ = self._eval_env.step(act)
            ret += rew
            
        # Return any scalar data you want to record
        return {"Return": ret}


class ExampleSolver(shinrl.BaseSolver):
    DefaultConfig = ExampleConfig
    @staticmethod
    def make_mixins(env, config):
        return [ExampleStepMixIn, ExampleEvalMixIn, ExampleSolver]
```


```python
env = gym.make("CartPole-v0")
config = ExampleSolver.DefaultConfig(add_interval=5, steps_per_epoch=20, eval_interval=10)
mixins = ExampleSolver.make_mixins(env, config)
solver = ExampleSolver.factory(env, config, mixins)
solver.run()
```

####  1.2.1. <a name='solver.scalars'></a>solver.scalars

The results from `step` and `evaluate` functions are stored in solver.scalars:


```python
solver.scalars
```


##  2. <a name='UsefulMixIns'></a>Useful MixIns

For ease of implementation, we provide the following base mixins (See details in [shinrl/solvers/base/base_mixin.py](../shinrl/solvers/base/base_mixin.py):

* **BaseGymEvalMixIn**: Base mixin for gym.Env evaluation. `explore` function is implemented. Need to implement `eval_act` function.
* **BaseGymExploreMixIn**: Base mixin for gym.Env exploration. `evaluate` function is implemented. Need to implement `explore_act` function.
* **BaseShinEvalMixIn**: Base mixin for ShinEnv evaluation. `explore` function is implemented. solver.data need to have `EvaluatePolicy` table.
* **BaseShinExploreMixIn**: Base mixin for ShinEnv exploration. `evaluate` function is implemented. solver.data need to have `ExplorePolicy` table.

###  2.1. <a name='GymEnvSolverExample'></a>GymEnv Solver Example

`BaseGymEvalMixIn` and `BaseGymExploreMixIn` conduct **sampling-based** evaluation and exploration.

You need to implement three functions: 
* `step` 
* `eval_act`
* `explore_act`

Here we implement the step function in `GymStepMixIn` and the act functions in `GymActMixIn`:


```python
@chex.dataclass
class GymConfig(shinrl.SolverConfig):
    seed: int = 0
    discount: float = 0.99
    eval_trials: int = 10
    eval_interval: int = 100
    add_interval: int = 100
    steps_per_epoch: int = 1000
    num_samples: int = 10
        

class GymStepMixIn:
    def step(self):
        samples = self.explore()
        dummy_loss = (samples.rew).mean()
        return {"DummyLoss": dummy_loss.item()}

    
class GymActMixIn:
    def eval_act(self, key, obs):
        new_key = jax.random.split(self.key)
        act = self._eval_env.action_space.sample()
        log_prob = 0.0
        return new_key, act, log_prob

    def explore_act(self, key, obs):
        new_key = jax.random.split(self.key)
        act = self.env.action_space.sample()
        log_prob = 0.0
        return new_key, act, log_prob


class GymSolver(shinrl.BaseSolver):
    DefaultConfig = GymConfig
    @staticmethod
    def make_mixins(env, config):
        return [GymStepMixIn, GymActMixIn, shinrl.BaseGymExploreMixIn, shinrl.BaseGymEvalMixIn, GymSolver]
```


```python
env = gym.make("CartPole-v0")
config = GymSolver.DefaultConfig(add_interval=5, steps_per_epoch=20, eval_interval=10)
mixins = GymSolver.make_mixins(env, config)
solver = GymSolver.factory(env, config, mixins)
solver.run()
```


###  2.2. <a name='ShinEnvSolverExample'></a>ShinEnv Solver Example

`BaseShinEvalMixIn` and `BaseShinExploreMixIn` conduct **oracle** evaluation and exploration.

You need to set two arrays to solver.data:

* `ExplorePolicy`: dS x dA probability array
* `EvaluatePolicy`: dS x dA probability array

Here we implement them in `BuildTableMixIn`.


```python
@chex.dataclass
class ShinConfig(shinrl.SolverConfig):
    seed: int = 0
    discount: float = 0.99
    eval_trials: int = 10
    eval_interval: int = 100
    add_interval: int = 100
    steps_per_epoch: int = 1000
    num_samples: int = 10
        
        
class BuildTableMixIn:
    def initialize(self, env, config=None) -> None:
        # build tables
        super().initialize(env, config)
        self.data["Q"] = jnp.zeros((self.dS, self.dA))
        self.data["ExplorePolicy"] = jnp.ones((self.dS, self.dA)) / self.dA
        self.data["EvaluatePolicy"] = jnp.ones((self.dS, self.dA)) / self.dA


class ShinStepMixIn:
    def step(self):
        samples = self.explore()
        dummy_loss = (samples.rew).mean()
        return {"DummyLoss": dummy_loss.item()}

    
class ShinSolver(shinrl.BaseSolver):
    DefaultConfig = ShinConfig
    @staticmethod
    def make_mixins(env, config):
        return [ShinStepMixIn, BuildTableMixIn, shinrl.BaseShinExploreMixIn, shinrl.BaseShinEvalMixIn, ShinSolver]
```


```python
env = gym.make("ShinMountainCar-v0")
config = ShinSolver.DefaultConfig(add_interval=5, steps_per_epoch=20, eval_interval=10)
mixins = ShinSolver.make_mixins(env, config)
solver = ShinSolver.factory(env, config, mixins)
solver.run()
```

###  2.3. <a name='GymEnvShinEnvSolverExample'></a>GymEnv & ShinEnv Solver Example

A solver can support both gym.Env & ShinEnv by modifing the `make_mixin` function:


```python
@chex.dataclass
class GymAndShinConfig(shinrl.SolverConfig):
    seed: int = 0
    discount: float = 0.99
    eval_trials: int = 10
    eval_interval: int = 100
    add_interval: int = 100
    steps_per_epoch: int = 1000
    num_samples: int = 10
        
        
class GymAndShinSolver(shinrl.BaseSolver):
    DefaultConfig = GymAndShinConfig
    @staticmethod
    def make_mixins(env, config):
        is_shin_env = isinstance(env, shinrl.ShinEnv)
        
        if is_shin_env:
            mixin_list = [ShinStepMixIn, BuildTableMixIn, shinrl.BaseShinExploreMixIn, shinrl.BaseShinEvalMixIn, GymAndShinSolver]
        else:
            mixin_list = [GymStepMixIn, GymActMixIn, shinrl.BaseGymExploreMixIn, shinrl.BaseGymEvalMixIn, GymAndShinSolver]
        
        return mixin_list
```


```python
# GymEnv
env = gym.make("MountainCar-v0")
config = GymAndShinSolver.DefaultConfig(add_interval=5, steps_per_epoch=20, eval_interval=10)
mixins = GymAndShinSolver.make_mixins(env, config)
solver = GymAndShinSolver.factory(env, config, mixins)
solver.run()
```

```python
# ShinEnv
env = gym.make("ShinMountainCar-v0")
config = GymAndShinSolver.DefaultConfig(add_interval=5, steps_per_epoch=20, eval_interval=10)
mixins = GymAndShinSolver.make_mixins(env, config)
solver = GymAndShinSolver.factory(env, config, mixins)
solver.run()
```