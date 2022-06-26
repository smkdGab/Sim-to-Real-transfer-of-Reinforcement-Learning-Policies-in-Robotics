"""Train an RL agent on the OpenAI Gym Hopper environment
"""

import torch
import gym
import argparse
import pickle
import re

from env.custom_hopper import *
from agent import AgentREINFORCE, PolicyREINFORCE

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n-episodes', default=20000, type=int, help='Number of training episodes')
	parser.add_argument('--print-every', default=1000, type=int, help='Print info every <> episodes')
	parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
	parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
	parser.add_argument('--model', default=None, type=str, help='Model path')
	return parser.parse_args()

args = parse_args()

def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = PolicyREINFORCE(observation_space_dim, action_space_dim)

	if args.model is None:
		beginning_episode = 0
	else:
		beginning_episode = re.search(r'\d+',args.model)
		print(beginning_episode)
		if beginning_episode is None:
			beginning_episode = 0
		else:
			beginning_episode = int(beginning_episode.group(0))
		policy.load_state_dict(torch.load(args.model), strict=True)

	agent = AgentREINFORCE(policy, device=args.device)
	print(args.model)
	print(beginning_episode)

	try:
	
		epoch_returns = []
		train_returns = []
		episode = 0
		for episode in range(beginning_episode, args.n_episodes):
			done = False
			train_reward = 0
			state = env.reset()  # Reset the environment and observe the initial state

			while not done:  # Loop until the episode is over

				action, action_probabilities = agent.get_action(state)
	

				state, reward, done,_  = env.step(action.detach().cpu().numpy())
				
				if args.render:
					env.render()

				agent.store_outcome(action_probabilities, reward, done)

				train_reward += reward
			
			train_returns.append(train_reward)
			epoch_returns.append(train_reward)

			if (episode+1)%args.print_every == 0:
				print('Training episode:', episode)
				print('Episode return:', train_reward)
				er = np.array(epoch_returns)
				print('avg_return:', er.mean())
				print('std_return:', er.std())
				epoch_returns = []
				torch.save(agent.policy.state_dict(), f"model-{episode}.mdl")
				with open(f"train_returns-{episode}.pickle","wb") as outf:
					pickle.dump(obj=train_returns, file=outf)
			
	except KeyboardInterrupt:
		torch.save(agent.policy.state_dict(), f"model-{episode}.mdl")
		with open(f"train_returns-{episode}.pickle","wb") as outf:
			pickle.dump(obj=train_returns, file=outf)
	
		raise KeyboardInterrupt


	torch.save(agent.policy.state_dict(), f"model_local-{episode}.mdl")
	with open(f"train_returns_local-{episode}.pickle","wb") as outf:
		pickle.dump(obj=train_returns, file=outf)

	

if __name__ == '__main__':
	main()
