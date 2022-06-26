import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from pprint import pprint
import warnings
import dill
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
# from weighted import quantile

class AutomaticDomainRandomization():

	def __init__(self, init_params: dict, p_b=0.5, m=50, delta=0.2, step='constant', thresholds:list = [1500, 1700]) -> None:
		self.init_params = init_params # phi_0, should be a dict of the form {'torso':val, 'leg':val, 'foot':val}
		self.thresholds = thresholds
		self.delta = delta
		self.m = m  # size data buffer to check each end of episode
		self.bounds = self._init_bounds() # bounds is a dict {'torso_low':val, 'torso_high':val, ...}
		self.p_b = p_b  # prob of using algo1 or algo2
		self.step = getattr(self, '_' + step)
		self.hinge_mass = None
		self.leg_mass = None
		self.foot_mass = None
		self.rewards = []
		self.weights = []
		self.current_weight = np.float64(1)
		self.last_performances = []
		self.last_increments = []
		self.part = ['hinge', 'leg', 'foot']
		self.databuffer = {
			"hinge_low": [],
			"hinge_high": [],
			"leg_low": [],
			"leg_high": [],
			"foot_low": [],
			"foot_high": []
		}
		self.keys = list(self.databuffer.keys())
		print('******CHECK THAT THE PARAMETERS ARE ORDERED CORRECTLY******', self.keys, sep='\n') # TODO: remove this print
		self.current_bound = 0
		# self.update_counter = 0

	"""
	SETTERS AND GETTERS
	"""
	def set_delta(self, delta: float):
		self.delta = delta

	def get_bounds(self):
		return self.bounds

	"""
	CORE
	"""
	def eval_entropy(self) -> float:
		range_hinge = self.bounds["hinge_high"] - self.bounds["hinge_low"]
		range_leg = self.bounds["leg_high"] - self.bounds["leg_low"]
		range_foot = self.bounds["foot_high"] - self.bounds["foot_low"]

		entropy = np.log([range_hinge, range_leg, range_foot]).mean()

		return entropy

	def insert_ep_return(self, body_part: str, ep_return: float) -> None:
		# try:
		if self.keys[self.current_bound] == body_part:
			self.databuffer[body_part].append(ep_return)
		# except:
		# 	print("Parameter body_part is NOT correct!")

	# compute the mean performance and clear buffer
	def _evaluate_perfomance(self, body_part: str) -> float:
		try:
			performance = np.mean(np.array(self.databuffer[body_part]))
			self.databuffer[body_part].clear()
		except:
			print("Parameter body_part is NOT correct!")
		return performance

	# Check size of the ADR and in case increase or decrease the bounds
	def updateADR(self, body_part: str):
		if len(self.databuffer[body_part]) >= self.m:
			# low or high
			bp, extract_extreme = tuple(body_part.split("_"))
			performance = self._evaluate_perfomance(body_part)
			
			self.last_performances.append((body_part, performance))

			# if performance >= self._th('high'):
			if performance >= self.thresholds[1]:
				if extract_extreme == "high":
					self._increase_high_bounds(body_part, performance)
				else:
					self._decrease_low_bounds(body_part, performance)
			# elif performance <= self._th('low'):
			if performance <= self.thresholds[0]:
				if extract_extreme == "high":
					self._decrease_high_bounds(body_part, performance)
				else:
					self._increase_low_bounds(body_part, performance)

	def get_random_masses(self):
		# Set three random masses
		hinge_mass = np.random.uniform(
			self.bounds["hinge_low"], self.bounds["hinge_high"])
		leg_mass = np.random.uniform(
			self.bounds["leg_low"], self.bounds["leg_high"])
		foot_mass = np.random.uniform(
			self.bounds["foot_low"], self.bounds["foot_high"])

		d = {"hinge": hinge_mass, "leg": leg_mass, "foot": foot_mass}

		# prob of set masses to lower or upper bound
		u = np.random.uniform(0, 1)
		k_bounds = None

		# Set one random parameter to its lower or upper bound
		if u < self.p_b:
			k_bounds = self._select_random_parameter()
			body_part = k_bounds.split("_")[0]

			d[body_part] = self.bounds[k_bounds]

		return list(d.values()), k_bounds

	def evaluate(self, episode_return, key_bounds) -> None:
		self.insert_ep_return(body_part=key_bounds, ep_return=episode_return)
		self.updateADR(body_part=key_bounds)
	
	"""
	DELTA
	"""
	def _small_or_big(self, threshold: float, performance: float, jump_factor: float = 5):
		gain = (performance - threshold)/1000 # divide by 1000 to obtain a number in (0, 1)
		n_rand = np.random.random()
		step = self.delta*jump_factor if n_rand < gain else self.delta
		return step
			
	def _proportional(self, threshold: float, performance: float, alpha: float = 1):
		gain = alpha * (performance - threshold)/100 # divide by 100 to likely obtain a number in (0, 10) (mostly values < 1)
		step = self.delta * (1 + gain)
		return step

	def _random(self, *args):
		return np.random.random()

	def _constant(self, *args):
		return self.delta

	def _gaussian(self, *args):
		return max(0.1 * self.delta, self.delta + 0.02 * np.random.normal())

	"""
	THRESHOLDS
	"""
	# def _th(self, x): # DEPRECATED
	#     return 1600 if x=='high' else 1200
	#     q = 0.4 if x == 'high' else 0.2
	#     d = np.array(self.rewards)
	#     w = np.array(self.weights)
	#     return weighted.quantile(d, w, q) if d.size > 0 else 0

	# def add(self, r): # DEPRECATED
	#     self.rewards.append(r)
	#     self.weights.append(self.current_weight)
	#     self.current_weight /= .99

	"""
	UTILITIES AND DEBUG
	"""
	def print_distributions(self):
		for p in self.part:
			high = self.bounds[p+'_high']
			low = self.bounds[p+'_low']
			center = self.init_params[p]
			left = round((center - low) // self.delta)
			right = round((high - center) // self.delta)
			if low != center:
				print(f'\t[{round(low, 2)}]«{"-"*left}', end='')
			else:
				print('\t', end='')
			print(f'[{round(center, 2)}]', end='')
			if high != center:
				print(f'{"-"*right}»[{round(high, 2)}]')
			else:
				print()
	
	def _increase_high_bounds(self, body_part: str, performance):
		step = self.step(self.thresholds[1], performance)
		self.bounds[body_part] = self.bounds[body_part] + step 
		self.last_increments.append((body_part, 'high+', step))
		self.current_bound = (self.current_bound+1) % len(self.keys)
		# if self.update_counter == 0:
		# 	self.current_bound = (self.current_bound+1) % len(self.keys)
		# else:
		# 	self.update_counter += 1

	def _decrease_low_bounds(self, body_part: str, performance):
		step = self.step(self.thresholds[1], performance)
		new_low_bounds = self.bounds[body_part] - step
		self.bounds[body_part] = max(new_low_bounds, 0)
		self.last_increments.append((body_part, 'low+', step))
		self.current_bound = (self.current_bound+1) % len(self.keys)
		# if self.update_counter == 0:
		# 	self.current_bound = (self.current_bound+1) % len(self.keys)
		# else:
		# 	self.update_counter += 1
	
	def _decrease_high_bounds(self, body_part: str, performance):
		body = body_part.split('_')[0]
		if not np.isclose(self.init_params[body], self.bounds[body_part]):
			# self.update_counter -= 1
			self.bounds[body_part] = max(self.bounds[body_part] - self.delta, self.init_params[body])
		self.last_increments.append((body_part, 'high-', self.delta))
	
	def _increase_low_bounds(self, body_part: str, performance):
		body = body_part.split('_')[0]
		if not np.isclose(self.init_params[body], self.bounds[body_part]):
			# self.update_counter -= 1
			self.bounds[body_part] = min(self.bounds[body_part] + self.delta, self.init_params[body])
		self.last_increments.append((body_part, 'low-', self.delta))

	def _init_bounds(self):
		try:
			dict = {"hinge_low": self.init_params['hinge'],
					"hinge_high": self.init_params['hinge'],
					"leg_low": self.init_params['leg'],
					"leg_high": self.init_params['leg'],
					"foot_low": self.init_params['foot'],
					"foot_high": self.init_params['foot']
					}
		except:
			print("Bounds not initialized")
		return dict

	# Extract random key
	def _select_random_parameter(self) -> str:
		# keys = list(self.bounds.keys())
		# rand = np.random.choice(len(keys))
		rand = np.random.randint(2) # random 0 or 1, if 1 changes the bound of the parameter that we are testing
		part = self.keys[self.current_bound^rand]
		# print('selected bound:', part) # TODO: remove this print
		return part
		# return keys[rand]

# =============================================================================================================================================

class ADRCallback(BaseCallback):
	def __init__(self, handlerADR, vec_env, eval_callback, n_envs=1, verbose=0, log_path_tb=None, log_path=None, save_freq:int=int(5e5), save_path:str='./models', name_prefix:str='adr_model'):
		super(ADRCallback, self).__init__(verbose)
		self.adr = handlerADR
		self.n_envs = n_envs
		self.vec_env = vec_env
		self.eval_callback = eval_callback
		self.bounds_used = [None] * n_envs
		self.n_episodes = 0
		self.log_timesteps = [] # num_timesteps
		self.log_results = []
		self.log_length = []
		self.log_num_episode = []
		self.log_bounds = []
		self.save_freq = save_freq
		self.save_path = save_path
		self.name_prefix = name_prefix
		self.num_timesteps_offset = 0
		self.interrupted = False
		self.log_path = log_path
		self.log_path_tb = log_path_tb

	def _init_callback(self) -> None:
		self.interrupted = False
		# Create folder if needed
		log_path = self.log_path
		if log_path is not None:
			log_path = os.path.join(log_path, "train")
			os.makedirs(os.path.dirname(log_path), exist_ok=True)
		self.log_path = log_path
		
		log_path_tb = self.log_path_tb
		if log_path_tb is not None:
			log_path_tb = os.path.join(log_path_tb, "tensorboard")
			os.makedirs(os.path.dirname(log_path_tb), exist_ok=True)
		self.log_path_tb = log_path_tb
		
		if self.save_path is not None:
			os.makedirs(self.save_path, exist_ok=True)
	
	def reload(self, num_timesteps_offset:int, verbose:int=None, log_path_tb:str=None, log_path:str=None, save_freq:str=None, save_path:str=None) -> None:
		self.num_timesteps_offset = num_timesteps_offset
		if verbose is not None:
			self.verbose = verbose
		if log_path is not None:
			self.log_path = log_path
		if log_path_tb is not None:
			self.log_path_tb = log_path_tb
		if save_freq is not None:
			self.save_freq = save_freq
		if save_path is not None:
			self.save_path = save_path

	def dump(self) -> None:
		path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps+self.num_timesteps_offset}_steps.dill")
		dump_obj = {
			'model': self.model,
			'callback': self
		}
		if self.verbose > 1:
			print(f"Saving model checkpoint to {path}")
		try:
			with open(path, 'wb') as dumpf:
				dill.dump(obj=dump_obj, file=dumpf)
		except:
			try:
				self.model.save(path.replace('dill','zip'))
			except:
				print(f'error saving {path}')

	def _on_rollout_start(self):
		pass

	def _on_rollout_end(self) -> None:
		self._clear_output()
		print()
		print('NUM_TIMESTEPS:', self.num_timesteps + self.num_timesteps_offset)
		print('N_EPISODES:', self.n_episodes)
		print()
		self.adr.print_distributions()
		print()
		print('DATA BUFFER SIZE:')
		pprint({k:len(v) for (k, v) in self.adr.databuffer.items()})
		print()
		print('LAST PERFORMANCES:')
		pprint(self.adr.last_performances)
		print()
		print('LAST UPDATES:')
		pprint(self.adr.last_increments)
		print()
		print(f"LOW_TH = {round(self.adr.thresholds[0])}\nHIGH_TH = {round(self.adr.thresholds[1])}")
		print()
		self.adr.last_performances.clear()
		self.adr.last_increments.clear()
		
		if self.interrupted:
			self.dump()
			exit()

	def _on_training_end(self):
		pass

	def _on_step(self):
		num_timesteps = self.num_timesteps + self.num_timesteps_offset
		#print(self.locals['dones'], self.locals['infos'], range(self.vec_env.num_envs), self.bounds_used, sep="\n\n")
		#sd = input()
		for done, infos, nr_env, bound_used in zip(self.locals['dones'], self.locals['infos'], range(self.vec_env.num_envs), self.bounds_used):
			if(done):
				self.n_episodes += 1
				if bound_used is not None:
					self.adr.evaluate(infos['episode']['r'], bound_used)
				env_params, self.bounds_used[nr_env] = self.adr.get_random_masses()
				# check indices, it is VecEnvIndices (from stable_baselines3.common.vec_env.base_vec_env)
				self.vec_env.env_method('set_parameters', env_params, indices=nr_env)
				"""
				Logging parameters
				"""
				if self.log_path is not None:
					self.log_num_episode.append(self.n_episodes)
					self.log_results.append(infos['episode']['r'])
					self.log_length.append(infos['episode']['l'])
					self.log_timesteps.append(num_timesteps)
					self.log_bounds.append(list(self.adr.get_bounds().values()))

					np.savez(
						self.log_path,
						timesteps=self.log_timesteps,
						results=self.log_results,
						ep_lengths=self.log_length,
						ep_num=self.log_num_episode,
						bounds=self.log_bounds
					)

				if self.log_path_tb is not None:
					self.logger.record('n_episodes', self.n_episodes)
					self.logger.record('ep_return',infos['episode']['r'])
					self.logger.record('ep_length', infos['episode']['l'])
					self.logger.record('num_timesteps', self.num_timesteps)
				

		if num_timesteps % self.save_freq < self.n_envs:
			self.dump()

	def _clear_output(self):
		os.system( 'clear' )
