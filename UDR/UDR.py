from UDR_env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
import gym
import argparse
import os
import numpy as np

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env-train', default='CustomHopper-source-randomized-v0', type=str, help='Train environment')
	parser.add_argument('--env-test', default='CustomHopper-target-v0', type=str, help='Test environment')
	parser.add_argument('--device', default='cpu', type=str, help='Device [cpu, cuda]')
	parser.add_argument('--eval-freq', default=100_000, type=int, help='Evaluation frequency')
	parser.add_argument('--verbose-ppo', default=0, type=int, help='Verbose parameter of PPO')
	parser.add_argument('--debug', default='False', type=str, help='Print model masses at each episode')
	parser.add_argument('--render', default='False', type=str, help='Render the scene')
	parser.add_argument('--target-log-path', type=str, help='Log path')
	parser.add_argument('--train-episodes', default=1e8, type=int, help='Training episodes')
	parser.add_argument('--checkpoint', default='True', type=str, help='Save the model periodically (train_episodes/100)')
	return parser.parse_args()

args = parse_args()

args.debug = (args.debug=='True')
args.render = (args.render=='True')
args.checkpoint = (args.checkpoint=='True')

def _create_source_env(bounds):
	source_env = gym.make(args.env_train)
	source_env.set_bounds(bounds)
	if args.debug:
		source_env.set_debug()
	return source_env

def compute_bounds(params):
	bounds = list((m-hw, m+hw) for m, hw in [(params['thigh_mean'], params['thigh_hw']), (params['leg_mean'], params['leg_hw']), (params['foot_mean'], params['foot_hw'])])
	return bounds

def main():
	params = {
			'thigh_mean': 3.92699082,
			'leg_mean': 2.71433605,
			'foot_mean': 5.0893801,
			'hw': 0.5,
			'thigh_hw': 0.5,
			'leg_hw': 0.5,
			'foot_hw': 0.5
			}
	bounds = compute_bounds(params) # Compute bounds

	source_env = _create_source_env(bounds)
	target_env = gym.make(args.env_test)
	target_eval_callback = EvalCallback(eval_env=target_env, n_eval_episodes=50, eval_freq=args.eval_freq, log_path=args.target_log_path) # evaluation of target during training (to do indipendently of the train env)
	cb_list = [target_eval_callback]
	if args.checkpoint:
		checkpoint_callback = CheckpointCallback(save_freq=int(np.ceil(args.train_episodes / 100)), save_path='./', name_prefix="UDR_model_")
		cb_list.append(checkpoint_callback)
	callback = CallbackList(cb_list)

	model = PPO("MlpPolicy", env=source_env, n_steps=1024, batch_size=128, learning_rate=0.00025, device=args.device, verbose=args.verbose_ppo, tensorboard_log="./UDR_tensorboard/")
	model.learn(total_timesteps=args.train_episodes, callback=callback, tb_log_name='UDR')
	model.save("UDR_model_1e8")
	
if __name__ == '__main__':
	main()
