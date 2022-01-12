**Status:** Under development (expect bug fixes and huge updates)

# ShinRL: A Library for Evaluating RL Algorithms from Theoretical and Practical Perspectives

ShinRL is an open-source JAX library specialized for the evaluation of reinforcement learning (RL) algorithms **from both theoretical and practical perspectives**.
Please take a look at [the paper](https://arxiv.org/abs/2112.04123) for details.
Try ShinRL at [experiments/QuickStart.ipynb](experiments/QuickStart.ipynb).

## QuickStart

![QuickStart](assets/quickstart.png)

```python
import gym
from shinrl import DiscreteViSolver
import matplotlib.pyplot as plt

# make an env & a config
env = gym.make("ShinPendulum-v0")
config = DiscreteViSolver.DefaultConfig(explore="eps_greedy", approx="nn", steps_per_epoch=10000)

# make & run a solver
mixins = DiscreteViSolver.make_mixins(env, config)
dqn_solver = DiscreteViSolver.factory(env, config, mixins)
dqn_solver.run()

# plot performance
returns = dqn_solver.scalars["Return"]
plt.plot(returns["x"], returns["y"])

# plot learned q-values  (action == 0)
q0 = dqn_solver.data["Q"][:, 0]
env.plot_S(q0, title="Learned")
```

![Example](assets/continual.gif)


# :zap: Key Modules

![overview](assets/overview.png)

## :microscope: ShinEnv for Oracle Analysis

* `ShinEnv` provides small environments with **oracle** methods that can compute exact quantities.
* Some environments support **continuous action space** and **image observation**:
* See the tutorial for details: [experiments/Tutorials/ShinEnvTutorial.ipynb](../../experiments/Tutorials/ShinEnvTutorial.ipynb).

|                  Environment                  |  Discrete action   | Continuous action  | Image Observation  | Tuple Observation  |
| :-------------------------------------------: | :----------------: | :----------------: | :----------------: | :----------------: |
|         [ShinMaze](shinrl/envs/Maze)          | :heavy_check_mark: |        :x:         |        :x:         | :heavy_check_mark: |
| [ShinMountainCar-v0](shinrl/envs/mountaincar) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|    [ShinPendulum-v0](shinrl/envs/pendulum)    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|    [ShinCartPole-v0](shinrl/envs/cartpole)    | :heavy_check_mark: | :heavy_check_mark: |        :x:         | :heavy_check_mark: |


## :factory: Flexible Solver by MixIn

* A `Solver` solves an environment with specified algorithms.
* A "mixin" is a class which defines and implements a single feature. ShinRL's solvers are instantiated by mixing some mixins.
* See the tutorial for details: [experiments/Tutorials/SolverTutorial.ipynb](../../experiments/Tutorials/SolverTutorial.ipynb).

![MixIn](assets/MixIn.png)

# Implemented Popular Algorithms

* The table bellow lists the implemented popular algorithms. 
* Note that it does not list all the implemented algorithms (e.g., DDP <sup id="a1">[1](#f1)</sup> version of the DQN algorithm). See `make_mixin` functions of solvers for implemented variants.
* Note that the implemented algorithms may differ from the original implementation for simplicity (e.g., Discrete SAC). See source code of solvers for details.


|                                          Algorithm                                           |                              Solver                              |                                 Configuration                                 | Type <sup id="a1">[1](#f1)</sup> |
| :------------------------------------------------------------------------------------------: | :--------------------------------------------------------------: | :---------------------------------------------------------------------------: | :------------------------------: |
|     [Value Iteration (VI)](https://www.science.org/doi/abs/10.1126/science.153.3731.34)      |     [DiscreteViSolver](shinrl/solvers/discrete_vi/solver.py)     |                ```approx == "tabular" & explore == "oracle"```                |             **TDP**              |
|            [Policy Iteration (PI)](https://psycnet.apa.org/record/1961-01474-000)            |     [DiscretePiSolver](shinrl/solvers/discrete_pi/solver.py)     |                ```approx == "tabular" & explore == "oracle"```                |             **TDP**              |
|    [Conservative Value Iteration (CVI)](http://proceedings.mlr.press/v89/kozuno19a.html)     |     [DiscreteViSolver](shinrl/solvers/discrete_vi/solver.py)     | ```approx == "tabular" & explore == "oracle & er_coef != 0 & kl_coef != 0"``` |             **TDP**              |
|      [Tabular Q Learning](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf)      |     [DiscreteViSolver](shinrl/solvers/discrete_vi/solver.py)     |                ```approx == "tabular" & explore != "oracle"```                |             **TRL**              |
|    [SARSA](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)    |     [DiscretePiSolver](shinrl/solvers/discrete_pi/solver.py)     |  ```approx == "tabular" & explore != "oracle" & eps_decay_target_pol > 0```   |             **TRL**              |
| [Deep Q Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) |     [DiscreteViSolver](shinrl/solvers/discrete_vi/solver.py)     |                  ```approx == "nn" & explore != "oracle"```                   |             **DRL**              |
|                         [Soft DQN](https://arxiv.org/abs/1702.08165)                         |     [DiscreteViSolver](shinrl/solvers/discrete_vi/solver.py)     |           ```approx == "nn" & explore != "oracle" & er_coef != 0```           |             **DRL**              |
|                      [Munchausen-DQN](https://arxiv.org/abs/2007.14430)                      |     [DiscreteViSolver](shinrl/solvers/discrete_vi/solver.py)     |   ```approx == "nn" & explore != "oracle" & er_coef != 0 & kl_coef != 0```    |             **DRL**              |
|                        [Double-DQN](https://arxiv.org/abs/1509.06461)                        |     [DiscreteViSolver](shinrl/solvers/discrete_vi/solver.py)     |       ```approx == "nn" & explore != "oracle" & use_double_q == True```       |             **DRL**              |
|                [Discrete Soft Actor Critic](https://arxiv.org/abs/1910.07207)                |     [DiscretePiSolver](shinrl/solvers/discrete_pi/solver.py)     |           ```approx == "nn" & explore != "oracle" & er_coef != 0```           |             **DRL**              |
|        [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971)         | [ContinuousDdpgSolver](shinrl/solvers/continuous_ddpg/solver.py) |                  ```approx == "nn" & explore != "oracle"```                   |             **DRL**              |


<b id="f1">1</b> Algorithm Type:
* **TDP** (```approx=="tabular" & explore=="oracle"```): Tabular Dynamic Programming algorithms. No exploration & no approximation & the complete specification about the MDP is given.
* **TRL** (```approx=="tabular" & explore!="oracle"```): Tabular Reinforcement Learning algorithms. No approximation & the dynamics and the reward functions are unknown.
* **DDP** (```approx=="nn"      & explore=="oracle"```): Deep Dynamic Programming algorithms. It is the same as TDP, except that neural networks approximate computed values.
* **DRL** (```approx=="nn"      & explore!="oracle"```): Deep Reinforcement Learning algorithms. It is the same as TRL, except that neural networks approximate computed values.

# Installation

```bash
git clone git@github.com:omron-sinicx/ShinRL.git
cd ShinRL
pip install -e .
```

# Test

```bash
cd ShinRL
make test
```

# Format

```bash
cd ShinRL
make format
```

# Docker

```bash
cd ShinRL
docker-compose up
```

# Citation

```
# Neurips DRL WS 2021 version (pytorch branch)
@inproceedings{toshinori2021shinrl,
    author = {Kitamura, Toshinori and Yonetani, Ryo},
    title = {ShinRL: A Library for Evaluating RL Algorithms from Theoretical and Practical Perspectives},
    year = {2021},
    booktitle = {Proceedings of the NeurIPS Deep RL Workshop},
}

# Arxiv version (commit 2d3da)
@article{toshinori2021shinrlArxiv,
    author = {Kitamura, Toshinori and Yonetani, Ryo},
    title = {ShinRL: A Library for Evaluating RL Algorithms from Theoretical and Practical Perspectives},
    year = {2021},
    url = {https://arxiv.org/abs/2112.04123},
    journal={arXiv preprint arXiv:2112.04123},
}
```
