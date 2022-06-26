from collections import OrderedDict
from dis import dis
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class PolicyNN(torch.nn.Module):
	def __init__(self, state_space, action_space, randomized=True):
		super().__init__()
		self.state_space = state_space
		self.action_space = action_space
		self.hidden = 64
		self.tanh = torch.nn.Tanh()
		self.randomized = randomized

		"""
			Actor network
		"""
		self.fc1 = torch.nn.Linear(state_space, self.hidden)
		self.fc2 = torch.nn.Linear(self.hidden, self.hidden)
		self.fc3 = torch.nn.Linear(self.hidden, action_space)
		
		# Learned standard deviation for exploration at training time 
	
		if randomized:
			self.sigma_activation = F.softplus
			init_sigma = 0.5
			self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if type(m) is torch.nn.Linear:
				torch.nn.init.normal_(m.weight)
				torch.nn.init.zeros_(m.bias)

	def forward(self, x):
		"""
			Actor
		"""
		x_actor = self.tanh(self.fc1(x))
		x_actor = self.tanh(self.fc2(x_actor))
		action_mean = self.fc3(x_actor)

		if self.randomized:
			sigma = self.sigma_activation(self.sigma)
			normal_dist = Normal(action_mean, sigma)
			return normal_dist
		else:
			return action_mean

class Policy():
	def __init__(self, state_space, action_space):
		self.policies = OrderedDict()
		self.policies['actor'] = PolicyNN(state_space, action_space)
		self.policies['critic'] = PolicyNN(state_space, 1, randomized=False)

	def to(self, device):
		for k,v in self.policies.items():
			self.policies[k] = v.to(device)
		return self

	def state_dict(self):
		sd = OrderedDict()
		for k,nn in self.policies.items():
			sd.update([(k,nn.state_dict())])
		return sd
	
	def load_state_dict(self, states, strict):
		for k,sd in states.items():
			self.policies[k].load_state_dict(sd, strict=strict)

class Agent(object):
	def __init__(self, policy, device='cpu', nn_baseline = True):
		self.train_device = device

		self.policy = policy.to(self.train_device)
		self.policy = policy
		self.actor_optimizer = torch.optim.Adam(policy.policies['actor'].parameters(), lr=1e-3)
		self.critic_optimizer = torch.optim.Adam(policy.policies['critic'].parameters(), lr=1e-3)
		self.use_NN_state_value = nn_baseline

		self.gamma = 0.99
		self._reinit()

	def _reinit(self):
		self.states = []
		self.next_states = []
		self.action_log_probs = []
		self.rewards = []
		self.done = []

	def update_policy(self):
		action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
		# states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
		# next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
		# rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
		# done = torch.Tensor(self.done).to(self.train_device)
		rewards = self.rewards

		states_values = [self.estimate_return(s) for s in self.states]
		states_values = torch.stack(states_values, dim=0).to(self.train_device).squeeze(-1)

		# Compute the discounted returns
		discounted_returns = []
		dr = 0
		for r in rewards[::-1]:
			dr = dr*self.gamma + r
			discounted_returns.append(dr)
		discounted_returns.reverse()
		discounted_returns = torch.stack(discounted_returns, dim=0).to(self.train_device).squeeze(-1)

		#baseline = states_values
		self.actor_optimizer.zero_grad()
		
		if self.use_NN_state_value: 
			delta = (discounted_returns - states_values).detach()
			# print(states_values.grad_fn)

			gamma_t = torch.from_numpy(np.fromiter((self.gamma**i for i in range(len(rewards))), dtype=np.float64))
			loss = -torch.mul(torch.mul(gamma_t, delta), action_log_probs).mean()
			loss.backward()
			
			self.critic_optimizer.zero_grad()
			loss_critic = -torch.mul(delta, states_values).mean()
			loss_critic.backward()
			self.critic_optimizer.step()

		else:
			discounted_returns = (discounted_returns - discounted_returns.mean())/ discounted_returns.std()
			# loss = -torch.mul(torch.mul(gamma_t,discounted_returns), action_log_probs).mean()
			loss = -torch.mul(discounted_returns, action_log_probs).mean()
			loss.backward()

		self.actor_optimizer.step()

	def get_action(self, state, evaluation=False):
		x = torch.from_numpy(state).float().to(self.train_device)

		normal_dist = self.policy.policies['actor'](x)

		if evaluation:  # Return mean
			return normal_dist.mean, None

		else:   # Sample from the distribution
			action = normal_dist.sample()

			# Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
			action_log_prob = normal_dist.log_prob(action).sum()

			return action, action_log_prob
	
	def estimate_return(self, state):
		x = state
		state_value = self.policy.policies['critic'](x)
		return state_value

	def store_outcome(self, state, next_state, action_log_prob, reward, done):
		self.states.append(torch.from_numpy(state).float())
		# self.next_states.append(torch.from_numpy(next_state).float())
		self.action_log_probs.append(action_log_prob)
		self.rewards.append(torch.Tensor([reward]))
		self.done.append(done)
		
		if done:
			self.update_policy()
			self._reinit()


