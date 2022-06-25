from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str)
    parser.add_argument('--source-log-path', type=str)
    parser.add_argument('--target-log-path', type=str)
    parser.add_argument('--episodes', default=100_000, type=int)
    return parser.parse_args()

args = parse_args()

if args.train is None or args.source_log_path is None or args.target_log_path is None:
    exit('Arguments required, run --help for more information')


N_ENVS = os.cpu_count()
MAX_EPS = args.episodes
ENV_EPS = int(np.ceil(MAX_EPS / N_ENVS))

def main():
    source_env = make_vec_env('CustomHopper-source-v0', n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
    target_env = make_vec_env('CustomHopper-target-v0', n_envs=N_ENVS, vec_env_cls=DummyVecEnv)

    stop_callback = StopTrainingOnMaxEpisodes(max_episodes=ENV_EPS, verbose=1) # callback for stopping at 100_000 episodes
    target_eval_callback = EvalCallback(eval_env=target_env, n_eval_episodes=50, eval_freq=15000, log_path=args.target_log_path) # evaluation of target during training
    callback_list = [stop_callback, target_eval_callback]

    if args.train == 'source':
        train_env = source_env # sets the train to source
        source_eval_callback = EvalCallback(eval_env=source_env, n_eval_episodes=50, eval_freq=15000, log_path=args.source_log_path) # if we are training in source, evaluate also in source
        callback_list.append(source_eval_callback)
    else:
        train_env = target_env

    callback = CallbackList(callback_list)

    model = PPO('MlpPolicy', n_steps=1024, batch_size=128, learning_rate=0.00025, env=train_env, verbose=1, device='cpu', tensorboard_log="./ppo_train_tensorboard/")
    model.learn(total_timesteps=int(1e10), callback=callback, tb_log_name=args.train)
    model.save("ppo_model_"+args.train)

if __name__ == '__main__':
    main()
