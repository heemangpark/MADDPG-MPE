from copy import deepcopy

import numpy as np
import torch as th
import torch.nn as nn
from torch.optim import Adam

from memory import ReplayMemory, Experience
from model import Critic, Actor
from params import scale_reward


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size, capacity, episodes_before_train, type):
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs, dim_act) for i in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(), lr=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(), lr=0.0001) for x in self.actors]

        self.type = type

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        ByteTensor = th.cuda.ByteTensor if self.use_cuda else th.ByteTensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = th.stack(batch.states).type(FloatTensor)
            action_batch = th.stack(batch.actions).type(FloatTensor)
            reward_batch = th.stack(batch.rewards).type(FloatTensor)
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = th.stack(
                [s for s in batch.next_states
                 if s is not None]).type(FloatTensor)

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](whole_state, whole_action)

            non_final_next_actions = [
                self.actors_target[i](non_final_next_states[:,
                                      i,
                                      :]) for i in range(
                    self.n_agents)]
            non_final_next_actions = th.stack(non_final_next_actions)
            non_final_next_actions = (
                non_final_next_actions.transpose(0,
                                                 1).contiguous())

            target_Q = th.zeros(
                self.batch_size).type(FloatTensor)

            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states),
                non_final_next_actions.view(-1,
                                            self.n_agents * self.n_actions)
            ).squeeze()
            # scale_reward: to scale reward in Q functions

            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
                    reward_batch[:, agent].unsqueeze(1) * scale_reward)

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, state_batch):
        # state_batch: n_agents x state_dim
        actions = th.zeros(
            self.n_agents,
            self.n_actions)
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_agents):
            sb = state_batch[i, :].detach()
            act = self.actors[i](sb.unsqueeze(0)).squeeze()

            act += th.from_numpy(
                np.random.randn(2) * self.var[i]).type(FloatTensor)

            if self.episode_done > self.episodes_before_train and \
                    self.var[i] > 0.05:
                self.var[i] *= 0.999998
            act = th.clamp(act, -1.0, 1.0)

            actions[i, :] = act
        self.steps_done += 1

        return actions

    def save(self, epi):
        if self.type == 'ag':
            th.save([actors.state_dict() for actors in self.actors], 'models/ag_actor_{}.pt'.format(epi))
            th.save([critics.state_dict() for critics in self.critics], 'models/ag_critic_{}.pt'.format(epi))
            th.save([a_optim.state_dict() for a_optim in self.actor_optimizer], 'models/ag_a_optim_{}.pt'.format(epi))
            th.save([c_optim.state_dict() for c_optim in self.critic_optimizer], 'models/ag_c_optim_{}.pt'.format(epi))
        elif self.type == 'adv':
            th.save([actors.state_dict() for actors in self.actors], 'models/adv_actor_{}.pt'.format(epi))
            th.save([critics.state_dict() for critics in self.critics], 'models/adv_critic_{}.pt'.format(epi))
            th.save([a_optim.state_dict() for a_optim in self.actor_optimizer], 'models/adv_a_optim_{}.pt'.format(epi))
            th.save([c_optim.state_dict() for c_optim in self.critic_optimizer], 'models/adv_c_optim_{}.pt'.format(epi))

    def load(self, epi):
        if self.type == 'ag':
            for ag_a, model in zip(self.actors, th.load('models/ag_actor_{}.pt'.format(epi))):
                ag_a.load_state_dict(model)
                ag_a.eval()
            for ag_c, model in zip(self.critics, th.load('models/ag_critic_{}.pt'.format(epi))):
                ag_c.load_state_dict(model)
                ag_c.eval()
            for ag_a_optim, model in zip(self.actor_optimizer, th.load('models/ag_a_optim_{}.pt'.format(epi))):
                ag_a_optim.load_state_dict(model)
            for ag_c_optim, model in zip(self.critic_optimizer, th.load('models/ag_c_optim_{}.pt'.format(epi))):
                ag_c_optim.load_state_dict(model)
        elif self.type == 'adv':
            for adv_a, model in zip(self.actors, th.load('models/adv_actor_{}.pt'.format(epi))):
                adv_a.load_state_dict(model)
                adv_a.eval()
            for adv_c, model in zip(self.critics, th.load('models/adv_critic_{}.pt'.format(epi))):
                adv_c.load_state_dict(model)
                adv_c.eval()
            for adv_a_optim, model in zip(self.actor_optimizer, th.load('models/adv_a_optim_{}.pt'.format(epi))):
                adv_a_optim.load_state_dict(model)
            for adv_c_optim, model in zip(self.critic_optimizer, th.load('models/adv_c_optim_{}.pt'.format(epi))):
                adv_c_optim.load_state_dict(model)