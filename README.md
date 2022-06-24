# Sim-to-Real-transfer-of-Reinforcement-Learning-Policies-in-Robotics
##### Gabriele Spina, Marco Sorbi, Christian Montecchiani.

Repository that contains the code for the project "Sim-to-Real-transfer-of-Reinforcement-Learning-Policies-in-Robotics" for the Machine Learning and Deep Learning 2021/2022 class.

*Abstract*: [to be updated]

# Requirements
- [Mujoco-py](https://github.com/openai/mujoco-py)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

# Environment
[...]

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
- loss functions of the critic network;
- whether the updates are batched or not.

### How to run the code
[...]

## TRPO and PPO
TRPO and PPO were imported from the [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) package and tested on the Hopper environment. In particular, PPO is the algorithm chosen for the *Domain Randomization* implementation.

# Uniform Domain Randomization
Uniform Domain Randomization uses a new custom randomized environment `CustomHopper-source-randomized-v0` created for the purpose. This randomized environment allows to set the bounds of the parameter distribution and it randomly samples the parameters at the beginning of each episode.
```
	env = gym.make('CustomHopper-source-randomized-v0')
	env.set_bounds(bounds)
	# bounds = [(a, b), ...] is a list of tuples, each containing lower and upper bounds of the distribution
	env.set_debug() # prints the current values for the masses
```


# Automatic Domain Randomization


[^fn1]: "https://arxiv.org/abs/1910.07113"
[^fn2]: "Reinforcement Learning: An Introduction (2nd edition) by Richard S. Sutton, Andrew G. Barto"
[^fn3]: "https://medium.com/@fork.tree.ai/understanding-baseline-techniques-for-reinforce-53a1e2279b57"
