import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from copy import copy
import gym
from collections import OrderedDict

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
    def __init__(self, policy, device='cpu', use_entropy=True, **kwargs):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.policy_optimizer = torch.optim.Adam(policy.policies['actor'].parameters(), lr=1e-3)
        self.state_value_optimizer = torch.optim.Adam(policy.policies['critic'].parameters(), lr=1e-3)
        self.gamma = 0.99
        self.use_entropy = use_entropy
        self.I = 1
        self.i = 0
        self.c_losses= []
        self.a_losses = []
        self.a = []
        self.c = []
        self.e = []
        self.BATCH_SIZE = 1

        
    def compute_losses(self, state, next_state, action_log_prob, R, done):
        v_hat = self.policy.policies['critic'](torch.from_numpy(state).float().to(self.train_device))
        v_hat_next = self.policy.policies['critic'](torch.from_numpy(next_state).float().to(self.train_device)) if not done \
                else torch.tensor([0], device=self.train_device)
        delta = R + self.gamma * v_hat_next - v_hat
        
        loss_entropy = - 0.01 * torch.exp(action_log_prob) * action_log_prob if self.use_entropy else torch.tensor([0]).to(self.train_device)
        loss_critic = F.mse_loss(R + self.gamma * v_hat_next, v_hat)
        loss_actor = - self.I * delta.detach() * action_log_prob

        self.a.append(loss_actor.item())
        self.c.append(loss_critic.item())
        self.e.append(loss_entropy.item())

        return loss_critic, loss_actor + loss_entropy

    def update_networks(self, state, next_state, action_log_prob, R, done):
        lc, la = self.compute_losses(state, next_state, action_log_prob, R, done)
        self.c_losses.append(lc)
        self.a_losses.append(la)
        self.I = self.I * self.gamma

        if done or (self.i+1) % self.BATCH_SIZE == 0:
            actor_loss = torch.stack(self.a_losses).mean()
            critic_loss = torch.stack(self.c_losses).mean()

            self.policy_optimizer.zero_grad()
            actor_loss.backward()
            self.policy_optimizer.step()

            self.state_value_optimizer.zero_grad()
            critic_loss.backward()
            self.state_value_optimizer.step()
        return

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

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.update_networks(state, next_state, action_log_prob, reward, done)
        if done or (self.i+1) % self.BATCH_SIZE == 0:
            self._clear()
        if done:
            self.i = 0

    def _clear(self):
        self.a_losses.clear()
        self.c_losses.clear()
        self.I = 1
