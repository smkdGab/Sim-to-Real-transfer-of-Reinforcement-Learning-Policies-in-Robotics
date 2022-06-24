import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class PolicyREINFORCE(torch.nn.Module):

    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)
        self.activation = F.softplus

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
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        return normal_dist
        



class AgentREINFORCE(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device

        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self._reinit()


    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        
        
        # Compute the discounted returns
        discounted_returns = np.fromiter((self.gamma**i * rewards[i] for i in range(len(rewards))), dtype= float)
        # Reverse the vector to compute cumsum() and reverse again to put in the right order 
        discounted_returns = discounted_returns[::-1].cumsum()[::-1]
        discounted_returns = torch.from_numpy(discounted_returns.copy()).to(self.train_device)

        # Compute the gradient
        self.optimizer.zero_grad()
        loss = -torch.mul(discounted_returns, action_log_probs).mean()
        loss.backward()
        self.optimizer.step()
        
    
        
    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()
    
            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()
            

            return action, action_log_prob


    def store_outcome(self, action_log_prob, reward, done):
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        
        if done:
            self.update_policy()
            self._reinit()


    def _reinit(self):
        self.action_log_probs = []
        self.rewards = []
    

