from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes, CallbackList, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.monitor import Monitor
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str)
    parser.add_argument('--source-log-path', type=str)
    parser.add_argument('--target-log-path', type=str)
    parser.add_argument('--timesteps', type=int)
    return parser.parse_args()
args = parse_args()

N_ENVS = os.cpu_count()

def main():
    source_env = make_vec_env('CustomHopper-source-v0', n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
    target_env = make_vec_env('CustomHopper-target-v0', n_envs=N_ENVS, vec_env_cls=DummyVecEnv)

    checkpoint_callback = CheckpointCallback(save_freq=int(np.ceil(1e7 / 12)), save_path='./', name_prefix="model_"+args.train+"_")
    target_eval_callback = EvalCallback(eval_env=target_env, n_eval_episodes=50, eval_freq=15000, log_path=args.target_log_path) # evaluation of target during training 
    callback_list = [checkpoint_callback, target_eval_callback]

    if args.train == 'source':
        source_eval_callback = EvalCallback(eval_env=source_env, n_eval_episodes=50, eval_freq=15000, log_path=args.source_log_path) # if we are training in source, evaluate also in source
        callback_list.append(source_eval_callback)
    
    # model = PPO.load("model_"+args.train, env=target_env, device='cpu', print_system_info=True)
    model = PPO('MlpPolicy', n_steps=1024, batch_size=128, learning_rate=0.00025, env=train_env, verbose=1, device='cpu', tensorboard_log="./ppo_train_tensorboard/")

    callback = CallbackList(callback_list)
    model.learn(total_timestep=args.timesteps, callback=callback, tb_log_name=args.train)
    model.save(f"ppo_model_{args.train}_{args.timesteps}")


if __name__ == '__main__':
    main()
