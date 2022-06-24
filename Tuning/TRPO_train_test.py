from env.custom_hopper import *
from sb3_contrib import TRPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

train_env = make_vec_env('CustomHopper-source-v0', n_envs=12, vec_env_cls=DummyVecEnv)
test_env = gym.make('CustomHopper-source-v0')

def train(args):
    model = TRPO('MlpPolicy', env=train_env, device='cpu', **args, verbose=1) 
    model.learn(total_timesteps=4e5)
    return model


def test(model):
    mean, _ = evaluate_policy(model, test_env, n_eval_episodes=100)
    return mean
