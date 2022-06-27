from env.custom_hopper import *
from ADR import AutomaticDomainRandomization, ADRCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

import gym
import argparse
import os
from tqdm import tqdm
import time
import dill
import re
import signal

adr_callback = None

def sigint_handler(signal, frame):
	print('exiting at next rollout end')
	adr_callback.interrupted = True

def parse_args():
	parser = argparse.ArgumentParser()
	n_envs = os.cpu_count()
	parser.add_argument('--pb', default=0.25, type=float, help='ADR | Probability of evalutating the training performance with ADR (Algorithm 1)')
	parser.add_argument('--m', default=50, type=int, help='ADR | Data buffer size')
	parser.add_argument('--low-th', default=1000, type=int, help='ADR | Lower threshold for performance evaluation')
	parser.add_argument('--high-th', default=1500, type=int, help='ADR | Upper threshold for performance evaluation')
	parser.add_argument('--delta', default=0.02, type=float, help='ADR | Initial step size for bounds increasing')
	parser.add_argument('--step', default='constant', type=str, help='ADR | Step method for bounds incrasing')
	parser.add_argument('--n-envs', default=n_envs, type=int, help='PPO | Number of parallel environments (DummyVecEnv)')
	parser.add_argument('--callback-verbose', default=0, type=int, help='ADR_callback | verbosity level')
	parser.add_argument('--ppo-train-timesteps', default=int(1e8), type=int, help='PPO | Timesteps for train')
	parser.add_argument('--log-path', default='./logs/', type=str, help='ADR_callback | Log path')
	parser.add_argument('--log-path-tb', default='./logs_TensorBoard/', type=str, help='ADR_callback | Tensorboard log path')
	parser.add_argument('--save-path', default='./models/', type=str, help='ADR_callback | Path to save models')
	parser.add_argument('--save-freq', default=int(1e6) // n_envs, type=int, help='ADR_callback | Frequence of saving the model')
	parser.add_argument('--model', default='', type=str, help='Path of the previous model. To use only to continue the training from a previous saved model')
	parser.add_argument('--render-eval', default=False, type=None, help='EvalCallback | Used for rendering the target evaluation during training')
	parser.add_argument('--log-path-eval', default='./logs/eval/', type=None, help='EvalCallback | Log path for target evaluation during training')
	parser.add_argument('--best-model-path', default='./models/best_eval/', type=None, help='EvalCallback | Log path for the best model (according to the target evaluation procedure)found so far during training')
	# parser.add_argument('', default='', type=None, help='')
	return parser.parse_args()

args = parse_args()

def main():
	global adr_callback
	"""
	Train process
	"""
	ppo_train_timesteps = args.ppo_train_timesteps
	if args.model:
		with open(args.model, 'rb') as model_file:
			obj_load = dill.load(model_file)
		adr_callback = obj_load['callback']
		eval_callback = adr_callback.eval_callback
		model = obj_load['model']
		beginning_timestep = re.search(r'\d+',args.model)
		if beginning_timestep is None:
			beginning_timestep = 0
		else:
			beginning_timestep = int(beginning_timestep.group(0))
		adr_callback.reload(beginning_timestep, verbose=args.callback_verbose, log_path_tb=args.log_path_tb, log_path=args.log_path, save_freq=args.save_freq, save_path=args.save_path)
		ppo_train_timesteps -= beginning_timestep
	else:
		init_params = {"thigh": 3.92699082,  "leg": 2.71433605, "foot": 5.0893801}

		handlerADR = AutomaticDomainRandomization(init_params, p_b=args.pb, m=args.m, step=args.step, delta=args.delta, thresholds=[args.low_th, args.high_th])
		
		train_env = make_vec_env('CustomHopper-source-v0', n_envs=args.n_envs, vec_env_cls=DummyVecEnv)
		train_env.set_attr(attr_name="bounds", value=handlerADR.get_bounds())
		
		test_env = gym.make('CustomHopper-target-v0') # TO DUMP
		eval_callback = EvalCallback(eval_env=test_env, n_eval_episodes=50, eval_freq=100000//args.n_envs, log_path=args.log_path_eval, deterministic=True, render=args.render_eval, best_model_save_path=args.best_model_path) # TO DUMP
		adr_callback = ADRCallback(handlerADR, train_env, eval_callback, n_envs=args.n_envs, log_path_tb=args.log_path_tb, verbose=args.callback_verbose, log_path=args.log_path, save_freq=args.save_freq, save_path=args.save_path)
		
		model = PPO('MlpPolicy', train_env, verbose=1, tensorboard_log=args.log_path_tb, learning_rate=0.00025, batch_size=128, n_steps=1024)
	
	callbacks = CallbackList([adr_callback, eval_callback]) # TO DUMP
	signal.signal(signal.SIGINT, sigint_handler)
	model.learn(total_timesteps=ppo_train_timesteps, callback=callbacks)

	train_env.close()

if __name__ == '__main__':
	main()
