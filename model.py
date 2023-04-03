import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical as C


class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action, discrete=False):
        super(Critic, self).__init__()
        self.discrete = discrete
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        self.obs_dim = dim_observation * n_agent
        self.act_dim = self.dim_action * n_agent

        self.FC1 = nn.Linear(self.obs_dim, 1024)
        self.FC2 = nn.Linear(1024 + self.act_dim, 512)
        self.FC3 = nn.Linear(512, 300)
        self.FC4 = nn.Linear(300, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        if self.discrete and len(acts.shape) == 3:
            acts = th.reshape(acts, (1000, self.act_dim))
        result = F.relu(self.FC1(obs))
        combined = th.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action, discrete=False):
        super(Actor, self).__init__()
        self.discrete = discrete
        self.FC1 = nn.Linear(dim_observation, 500)
        self.FC2 = nn.Linear(500, 128)
        self.FC3 = nn.Linear(128, dim_action)
        self.softmax = nn.Softmax(dim=0)

    # action output between -2 and 2
    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = self.FC3(result)
        if self.discrete:
            action_prob = self.softmax(result)
            dist = C(probs=action_prob)
            action = dist.sample()
            return action
        else:
            return th.tanh(result)
