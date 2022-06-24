"""Train an RL agent on the OpenAI Gym Hopper environment

TODO: implement 2.2.a and 2.2.b
"""

import torch
import gym
import argparse
import pickle
import signal
import re
import os
import random
import string

from learner.env.custom_hopper import *
from learner.agent import Agent, Policy

interrupted = False

def sigint_handler(signal, frame):
	global interrupted
	interrupted = True

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
	parser.add_argument('--print-every', default=1000, type=int, help='Print info every <> episodes')
	parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
	parser.add_argument('--model', default=None, type=str, help='pretrained model from which resume training')
	parser.add_argument('--train-env', default='target', type=str, help='training environment in (source, target)')
	parser.add_argument('--test-envs', default='target', type=str, help='testing environment in (source, target)')
	parser.add_argument('--xd', default=250, type=int, help='number of training episodes between two tests (for plot)')
	parser.add_argument('--yd', default=50, type=int, help='number of test episodes for a plot point')
	parser.add_argument('--use-nn-baseline', default=True, type=bool, help='use neural network as baseline (may be ignored by some agents)')
	#parser.add_argument('--use-entropy', default=False, type=bool, help='use entropy regularization (may be ignored by some agents)')
	return parser.parse_args()


def test(agent:Agent, test_env:str, episodes:int):

	env = gym.make(test_env)

	# print('Action space:', env.action_space)
	# print('State space:', env.observation_space)
	# print('Dynamics parameters:', env.get_parameters())
	
	returns = []
	for episode in range(episodes):
		done = False
		test_return = 0
		state = env.reset()

		while not done:

			action, _ = agent.get_action(state, evaluation=True)
			
			state, reward, done, info = env.step(action.detach().cpu().numpy())

			# if args.render:
			# 	env.render()

			test_return += reward
		
		returns.append(test_return)
	return returns


def main():
	args = parse_args()
	test_envs = args.test_envs.split(',')
	assert all(x in ('source','target') for x in [args.train_env] + test_envs)
	train_env = f'CustomHopper-{args.train_env}-v0'
	test_envs = {x:f'CustomHopper-{x}-v0' for x in test_envs}
	env = gym.make(train_env)

	beginning_episode = 0
	returns = {e:{} for e in test_envs.keys()}

	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	if args.model is None:
		while True:
			workdir = ''.join(random.choice(string.ascii_lowercase) for _ in range(8))
			try:
				os.mkdir(workdir)
				break
			except:
				pass
		print('workdir:', workdir)
	else:
		print("hello")
		s = args.model.rsplit('/', 1)
		if len(s) == 1:
			workdir = '.'
		else:
			workdir = s[0]
		model = s[-1]
		beginning_episode = re.search(r'\d+',model)
		if beginning_episode is None:
			beginning_episode = 0
		else:
			beginning_episode = int(beginning_episode.group(0))
			
			for e in returns.keys():
					returns_filename = f'returns-{e}-{beginning_episode}.pickle'
					with open(returns_filename, 'rb') as returns_file:
						returns[e] = pickle.load(file=returns_file)
			# except:
			# 	print(returns_filename, 'not found')
			# 	exit(0)
			print('resuming from episode', beginning_episode)
		policy.load_state_dict(torch.load(args.model), strict=True)
	
	agent = Agent(policy, device=args.device, nn_baseline=args.use_nn_baseline)

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	signal.signal(signal.SIGINT, sigint_handler)

	for episode in range(beginning_episode, args.n_episodes):

		if episode%args.xd == 0:
			for e,test_env in test_envs.items():
				returns[e][episode] = (test(agent, test_env, args.yd))

		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over

			action, action_probabilities = agent.get_action(state)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done)

			train_reward += reward
		
		
		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode+1)
			# print('Episode return:', train_reward)
			try:
				ret = list(returns.values())[-1]
				print('Last test mean return:', sum(ret)/len(ret))
			except:
				pass
		
		if interrupted:
			break

	episode += 1
	# if episode%args.xd == 0:
	for e,test_env in test_envs.items():
		returns[e][episode] = (test(agent, test_env, args.yd))

	torch.save(agent.policy.state_dict(), f"{workdir}/model-{episode}.mdl")
	for e,r in returns.items():
		with open(f'{workdir}/returns-{e}-{episode}.pickle','wb') as returns_file:
			pickle.dump(obj=r, file=returns_file)

	

if __name__ == '__main__':
	main()
