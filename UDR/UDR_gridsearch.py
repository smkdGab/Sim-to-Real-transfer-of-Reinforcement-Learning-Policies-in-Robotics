# Pretty prints
from pprint import pprint

# Policy
from UDR_env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
# Standard import
import numpy as np
import gym
import argparse
import os
from tqdm import tqdm
import time

# Hyperparameters optimization import
from sklearn.model_selection import ParameterGrid
import pickle

import itertools


# Save info
history = []
dump_counter = 0

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env-train', default='CustomHopper-source-randomized-v0', type=str, help='Train environment')
	parser.add_argument('--env-test', default='CustomHopper-target-v0', type=str, help='Test environment')
	parser.add_argument('--device', default='cpu', type=str, help='Device [cpu, cuda]')
	parser.add_argument('--train-steps', default=1e6, type=int, help='Train timesteps for a single policy')
	parser.add_argument('--verbose-ppo', default=0, type=int, help='Verbose parameter of PPO')
	parser.add_argument('--verbose', default='False', type=bool, help='Print hyperparameters and mean return at each iteration')
	parser.add_argument('--debug', default='False', type=str, help='Print model masses at each episode')
	return parser.parse_args()

args = parse_args()
args.debug = (args.debug == 'True')
args.verbose = (args.verbose == 'True')

DEVICE = args.device
VERBOSE = args.verbose  # print hyperparameters and mean return at each iteration
VERBOSE_PPO = args.verbose_ppo
DEBUG = args.debug  # print model masses at each episode
TRAIN_STEPS = args.train_steps  # default = 15
OUT_OF_BOUNDS_RETURN = "OUT of BOUNDS"
ENV_TRAINING = args.env_train  
ENV_TESTING = args.env_test
MODEL_NAME = "PPO_DR.mdl"

LEARNING_RATE = 0.00025
BATCH_SIZE = 128
N_STEPS = 1024


def gridify(parameters_dict: dict) -> list:
	return list(ParameterGrid(parameters_dict))

def compute_bounds(params):
	bounds = list((m-hw, m+hw) for m, hw in [(params['thigh_mean'], params['thigh_hw']), (params['leg_mean'], params['leg_hw']), (params['foot_mean'], params['foot_hw'])])
	if VERBOSE:
		print(f"Masses bounds: {bounds}")
	return bounds

def _create_source_env(bounds):
	source_env = gym.make(ENV_TRAINING)
	source_env.set_bounds(bounds)
	if DEBUG:
		source_env.set_debug()
	return source_env

def _train(params: dict, env) -> None:
	"""
	params:         hyperparameters for training
	env:            train environment
	"""
	model = PPO("MlpPolicy", env, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, n_steps=N_STEPS, device=DEVICE, verbose=VERBOSE_PPO)
	start_time = time.time()
	model.learn(total_timesteps=TRAIN_STEPS)
	
	finish_time = time.time()
	if DEBUG:
		print(f"Training PPO finished in: {finish_time-start_time:.2f} seconds")
	return model

def train_and_test(params: dict) -> dict:
	global dump_counter
	if VERBOSE:
		pprint(params)
	
	dumpv = {
		'history': history
	}
	with open(f'opt_{dump_counter}.log', 'wb') as outf:
		pickle.dump(obj=dumpv, file=outf)
	print(dump_counter)
	dump_counter += 1

	print('start')
	# Check hyperparameters out of bounds
	if any(x < 0 for x in params.values()):# or params['learning_rate'] > 1 or params['ent_coef'] > 1:
		return OUT_OF_BOUNDS_RETURN

	bounds = compute_bounds(params) # Compute bounds
	
	if any(x[0] < 0 for x in bounds): # Check masses not to be less than zero
		return OUT_OF_BOUNDS_RETURN

	source_env = _create_source_env(bounds)
	target_env = gym.make(ENV_TESTING)

	model = _train(params, source_env) 
	mean, std_dev = evaluate_policy(model, target_env, n_eval_episodes=50, deterministic=True)
	print('end test')
	if VERBOSE:
		pprint(mean)

	source_env.close()
	target_env.close()
	history.append((params, - mean))  # negative because the method is made for minimizing
	
	return {'loss': -mean, 'status': True}


################################# END IMPORT EVALUATOR #################################
def main():
	torso_s = 2.53429174
	torso_t = 3.53429174
	tr = torso_s / torso_t
	b0 = 3.92699082
	b1 = 2.71433605
	b2 = 5.08938010

	default_params = {
		'thigh_mean': b0,
		'leg_mean': b1,
		'foot_mean': b2,
	}

	space = {
		'scale': [1, tr, 1/tr],
		'hw': [0.5, 1, 1.5]
	}

	keys = list(space.keys())
	for p in tqdm(itertools.product(*space.values())):
		kw = dict(zip(keys, p))
		for k, v in default_params.items():
			kw[k] = v*kw['scale'] 
		for bp in ['thigh', 'leg', 'foot']:
			kw[f'{bp}_hw'] = kw['hw']
		train_and_test(kw)


	print('len(history):', len(history))
	best = min(history, key=lambda tpl: tpl[1])

	print(f"Best results: (the score is negated if the value has to be maximized): \n {best}")

	dumpv = {
		'history': history
	}

	with open('opt.log', 'wb') as outf:
		pickle.dump(obj=dumpv, file=outf)

if __name__ == '__main__':
	main()
