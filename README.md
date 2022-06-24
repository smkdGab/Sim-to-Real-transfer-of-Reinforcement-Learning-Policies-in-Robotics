# Sim-to-Real-transfer-of-Reinforcement-Learning-Policies-in-Robotics
##### Gabriele Spina, Marco Sorbi, Christian Montecchiani.

The repository contains the code for the project "Sim-to-Real-transfer-of-Reinforcement-Learning-Policies-in-Robotics" for the Machine Learning and Deep Learning 2021/2022 class.
The repository is intended as a support tool for the report of the project and it contains toy examples of some well-known algorithms and methods in the fields of Reinforcement Learning and Sim-to-Real transfer.

*Abstract*: [to be updated]

# Requirements
- [Mujoco-py](https://github.com/openai/mujoco-py)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

# Environment
The tests were performed using the [Hopper environment](https://www.gymlibrary.ml/environments/mujoco/hopper/) of Mujoco. The environment contains a flat world and a one-legged robot and the goal is to teach the leg how to move (jump) forward as fast as possible. In particular, two environments (*CustomHopper-source-v0* and *CustomHopper-target-v0*) were used to perform the *Sim-to-Sim* transfer. The two environments differ for the mass of the torso (`source=2.53429174; target=3.53429174`), while the other masses are unchanged. 


# Algorithms
The repository contains different implementations of some well-known algorithms in the field of Reinforcement Learning, and an implementation of the *Automatic Domain Randomization* algorithm of [OpenAI](https://openai.com/) introduced in [^fn1].

## REINFORCE
Three variaions of the REINFORCE algorithm, following a slightly modified version of the one proposed by [^fn2]. The implementations differ for the usage of the *baseline* term:
1. no baseline;
2. whitening transformation baseline [^fn3];
3. state-value function baseline.

### How to run the code
[...]

## A2C
Four variations of the vanilla Advantage Actor Critic algorithm, also slightly modified starting from [^fn2]. The implementations differ for:
- loss functions of the critic network (`A2C_mse` and `A2C_v`);
- whether the updates are batched or not (`A2C_batched` and `A2C_stepwise`).

### How to run the code
[...]

## TRPO and PPO
TRPO and PPO were imported from the [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) package and tested on the Hopper environment. In particular, PPO is the algorithm chosen for the *Domain Randomization* implementation.

## Hyperparameters tuning
Contents of Tuning folder:
- `gridsearch` for TRPO and PPO (\*) (\*\*)
- `gridsearch` for REINFORCE and A2C (\*) (\*\*)
- `utils` for REINFORCE and A2C (\*\*)
- `utils` for TRPO (`TRPO_train_test.py`)
- `utils` for PPO (`PPO_train_test.py`)
- `env` (source)

To do before running the code:
- open the interested file
- change imports (TRPO/PPO or REINFORCE/A2C)
- change hyperparameters to be tuned
- set `multiple_starts` (optional, default=4)
- set `log_path` (optional, default='outputs')


(\*):	these are the only files supposed to be runned

(\*\*):	modify the imports first for being sure that the code is importing the correct algorithm


# Uniform Domain Randomization
Uniform Domain Randomization uses a new custom randomized environment `CustomHopper-source-randomized-v0` created for the purpose. This randomized environment allows to set the bounds of the parameter distribution and it randomly samples the parameters at the beginning of each episode.
```
env = gym.make('CustomHopper-source-randomized-v0')
env.set_bounds(bounds)
# bounds = [(a, b), ...] is a list of tuples, each containing lower and upper bounds of the distribution
env.set_debug() # prints the current values for the masses
```

## How to run the code
`env` in the UDR folder contains the randomized source environment to train the agent using UDR.

# Automatic Domain Randomization
Automatic Domain Randomization was introduced by OpenAI [OpenAI](https://openai.com/) as an adversarial approach to the Domain Randomization techniques. Our implementation rely on the algorithms and the callbacks provided by [stable-baselines3](https://github.com/DLR-RM/stable-baselines3). For simplicity purposes the randomized environment of the UDR was not used in this context; instead we provided the `ADR` class and implemented a custom callback (`ADRCallback`) class to handle better the modification of the bounds. The two classes are thought to be complementary and they have to be used together:
- `ADR` class contains the hyperparameters of the method and the core methods for handling the buffers, evaluating the performances and updating the bounds;
- `ADRCallback` class implements the `BaseCallback` of `stable-baselines3`. It is used to call the `ADR` methods at each episode and for debugging and logging purposes.

## How to run the code
[...]


[^fn1]: "https://arxiv.org/abs/1910.07113"
[^fn2]: "Reinforcement Learning: An Introduction (2nd edition) by Richard S. Sutton, Andrew G. Barto"
[^fn3]: "https://medium.com/@fork.tree.ai/understanding-baseiline-techniques-for-reinforce-53a1e2279b57"
