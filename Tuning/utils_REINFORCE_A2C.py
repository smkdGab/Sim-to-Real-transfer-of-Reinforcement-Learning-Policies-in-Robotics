# from agent_A2C_batched_v import Agent, Policy
from agent_REINFORCE_NN_BASELINE import Agent, Policy
import gym
from env.custom_hopper import *

def train(train_env:str='CustomHopper-source-v0', device='cpu', episodes:int=20000, **kwargs) -> Agent:
	env = gym.make(train_env)

	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)


	agent = Agent(policy, device=device, **kwargs)
	
	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())

	for episode in range(episodes):

		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over

			action, action_probabilities = agent.get_action(state)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done)

			train_reward += reward
		
	return agent


def test(agent:Agent, episodes:int=100, test_env:str='CustomHopper-source-v0') -> float:

	env = gym.make(test_env)

	# print('Action space:', env.action_space)
	# print('State space:', env.observation_space)
	# print('Dynamics parameters:', env.get_parameters())
	
	test_return = 0
	for episode in range(episodes):
		done = False
		state = env.reset()

		while not done:

			action, _ = agent.get_action(state, evaluation=True)
			
			state, reward, done, info = env.step(action.detach().cpu().numpy())

			# if args.render:
			# 	env.render()

			test_return += reward
		
	return test_return / episodes
