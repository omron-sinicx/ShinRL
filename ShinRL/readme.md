# ShinRL: A Library for Evaluating RL Algorithms from Theoretical and Practical Perspectives


# Installation

1. Create Conda Environment
```bash
conda create -n shinrl python=3.9 -y
conda activate shinrl
```

2. Install shinrl

```bash
cd ShinRL
pip install -e .[tests]
```

3. Install MinAtar for Case Study 2

```bash
cd MinAtar
pip install -e .
```

4. Check if shinrl installed successfully

```bash
cd ShinRL
make test
```

# Reproduction 


ShinRL is an open-source library specialized for the evaluation of reinforcement learning (RL) algorithms from both theoretical and practical perspectives.
ShinRL provides tools to analyze *shin* values (*shin* means oracle in Japanese) in a MDP, making it easy to debug RL algorithms.

```python
import gym
import matplotlib.pyplot as plt
from shinrl.solvers.vi.discrete import ViSolver

# Set up a solver
env = gym.make("ShinPendulum-v0")
solver = ViSolver()
config = ViSolver.DefaultConfig(explore="eps_greedy", approx="nn", steps_per_epoch=500)
solver.initialize(env=env, config=config)

# Do Value Iteration
solver.run()

# Learning information is stored in solver.history

# plot performance
returns = solver.history.scalars["Return"]
plt.plot(returns["x"], returns["y"])

# plot q-value matrix
q = solver.history.tbs["Q"]
env.plot_SA(q, title="Q values")
```

See [quickstart.py](experiments/tutorial/quickstart.py) and [tutorial.ipynb](experiments/tutorial/tutorial.ipynb) for the basic usages.
For more information, you can refer to [ShinRL's documentation](https://shinrl.readthedocs.io/en/latest/?).

![Ant](assets/ant.gif)
![Pendulum](assets/pendulum.gif)
![Tabular](assets/tabular.gif)


# :pencil: Experiments

The goal of ShinRL is not to provide state-of-the-art agents but to analyze the behaviors of RL algorithms.
To this end, ShinRL provides `experiments` that reproduce various analyses from many studies.
For example, ``experiments/VI_Error_Propagation`` analyzes the effects of entropy and KL regularization on the error tolerance and convergence speed of VI.

|                                    Experiment                                    |                       Objective                        |                                              Papers                                              |
| :------------------------------------------------------------------------------: | :----------------------------------------------------: | :----------------------------------------------------------------------------------------------: |
|                         [Tutorial](experiments/Tutorial)                         |                Learn how to use ShinRL                 |                                                                                                  |
| [VI Performance Bound](https://shinrl.readthedocs.io/en/latest/experiments.html) | Examine the performance bound of various VI algorithms | [Leverage the Average: an Analysis of KL Regularization in RL](https://arxiv.org/abs/2003.14089) |
|                     [Deadly Triad](experiments/DeadlyTriad)                      |         Investigate the cause of deadly triad          |     [Towards Characterizing Divergence in Deep Q-Learning](https://arxiv.org/abs/1903.08894)     |


# :zap: Key features

## Oracle analysis with ShinEnv
* A flexible class `ShinEnv` provides useful functions for RL analysis. For example, you can calculate the oracle action-values with ``calc_q`` method.
* Subclasses of `ShinEnv` can be used as the regular OpenAI-Gym environments.
* Some environments support continuous action space and image observation.

|                   Environment                    |   Dicrete action   | Continuous action  | Image Observation  | Tuple Observation  |
| :----------------------------------------------: | :----------------: | :----------------: | :----------------: | :----------------: |
|        [Maze](shinrl/envs/Maze)        | :heavy_check_mark: |        :x:         |        :x:         | :heavy_check_mark: |
| [TabularMountainCar-v0](shinrl/envs/mountaincar) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|    [TabularPendulum-v0](shinrl/envs/pendulum)    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|    [TabularCartPole-v0](shinrl/envs/cartpole)    | :heavy_check_mark: | :heavy_check_mark: |        :x:         | :heavy_check_mark: |

See [shinrl/\_\_init\_\_.py](shinrl/__init__.py) for the available environments.

## Gym solvers
* `ShinRL` provides algorithms to solve MDPs as `Solver`.
* ShinRL adopts a special directory structure. Common RL libraries implement algorithms (e.g., DQN and SAC) first and then each algorithm supports multiple environments (e.g., Classic Control and Mujoco). On the other hand, ShinRL implements algorithms on a task-by-task basis, which makes it easier to implement algorithms for new environments.
* Easy to visualize the training progress with [ClearML](https://github.com/allegroai/clearml).

|                                       Supported Environments                                       |                                                                                                                                                                                                Algorithms                                                                                                                                                                                                |                                                      Solvers                                                       |
| :------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------: |
|                                 [ShinEnv Discrete](shinrl/envs)                                 | Value Iteration (VI), [Conservative Value Iteration (CVI)](http://proceedings.mlr.press/v89/kozuno19a.html), [Munchausen Value Iteration (MVI)](https://arxiv.org/abs/2007.14430), <br> Policy Iteration (PI), [Conservative Policy Iteration (CPI)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.7.7601), <br>[Safe Policy Iteration (SPI)](http://proceedings.mlr.press/v28/pirotta13.html) | <sup id="a1">[1](#f1)</sup>(Oracle, Sampling, ExactFitted, SamplingFitted)(C, M)ViSolver, <br> (Oracle)(C)PiSolver |
|                                [ShinEnv Continuous](shinrl/envs)                                |                                                                                                                                                       Policy Gradient (PG), [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)                                                                                                                                                       |                                       (Exact, Sampling)(PG)Solver, PpoSolver                                       |
| [Mujoco](https://gym.openai.com/envs/#mujoco), [PyBullet](https://github.com/benelot/pybullet-gym) |                                                                                                                                                                       [Soft Actor Critic (SAC)](https://arxiv.org/abs/1801.01290)                                                                                                                                                                        |                                                     SacSolver                                                      |
|                          [MinAtar](https://github.com/kenjyoung/MinAtar)                           |                                                                                                                      [Deep Q Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf), [Munchausen DQN (M-DQN)](https://arxiv.org/abs/2007.14430), SAC                                                                                                                       |                                          DqnSolver, MDqnSolver, SacSolver                                          |

<b id="f1">1</b> The naming rule follows [Diagnosing Bottlenecks in Deep Q-learning Algorithms](https://arxiv.org/abs/1902.10250): 
* *Oracle-* solvers don't contain any errors. 
* *Sampling-* solvers use data sampled from MDP.
* *Exact Fitted-* solvers use function approximation but don't use sampled data.
* *Sampling Fitted-* solvers use both function approximation and sampled data. 
