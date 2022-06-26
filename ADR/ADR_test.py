from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='CustomHopper-target-v0', type=str)
    parser.add_argument('--model', type=str)
    return parser.parse_args()
args = parse_args()

if args.model is None:
    exit('Model path required, see --help')

env = gym.make(args.env)
model = PPO.load(args.model, env=env, print_system_info=True)
mean, std_dev = evaluate_policy(model, env, render=True)
print(f'Mean return: {mean} +- {std_dev}')
