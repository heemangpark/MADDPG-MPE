import pickle

import numpy as np
import torch as th
import h5py

from MADDPG import MADDPG
from madrl_environments.multiagent.make_env import make_env

# import visdom

# do not render the scene
# wandb.init()
e_render = False

food_reward = 10.
poison_reward = -1.
encounter_reward = 0.01
n_coop = 2

world_mpe = make_env('simple_tag')
world_mpe.discrete_action_space = False
# vis = visdom.Visdom(port=5274)
reward_record = []

np.random.seed(1234)
th.manual_seed(1234)

n_ag = 2
n_adv = 4
n_states = 28
n_actions = 2
capacity = 1000000
batch_size = 1000

n_episode = 7
max_steps = 50
episodes_before_train = 100

win = None
param = None

maddpg_ag = MADDPG(n_ag, n_states, n_actions, batch_size, capacity, episodes_before_train)
maddpg_adv = MADDPG(n_adv, n_states, n_actions, batch_size, capacity, episodes_before_train)

FloatTensor = th.cuda.FloatTensor if maddpg_ag.use_cuda else th.FloatTensor

# Dataset saving
action_data = []
obs_data = []
reward_data = []
state_data = []
terminated_data = []

for i_episode in range(n_episode):
    # obs = world.reset()
    obs = world_mpe.reset()
    world_mpe.render(obs)
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()

    ag_rwd = 0
    adv_rwd = 0

    for t in range(max_steps):
        # render every 100 episodes to speed up training
        # if i_episode % 100 == 0 and e_render:
        #     world.render()
        obs = obs.type(FloatTensor)
        action_ag = maddpg_ag.select_action(obs[:n_ag]).data.cpu()
        action_adv = maddpg_adv.select_action(obs[n_ag:]).data.cpu()
        action_tot = th.cat([action_ag, action_adv], dim=0)
        obs_, reward, done, _ = world_mpe.step(action_tot.numpy())

        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()

        # saving
        action_data.append(action_tot.numpy())
        obs_data.append(obs)
        reward_data.append(reward)
        # state_data.append(world_mpe)
        terminated = t == max_steps - 1
        terminated_data.append([terminated] * (n_ag + n_adv))

        if t != max_steps - 1:
            next_obs = obs_
            next_obs_ag = next_obs[:n_ag]
            next_obs_adv = next_obs[n_ag:]
        else:
            next_obs = None
            next_obs_ag = None
            next_obs_adv = None

        ag_rwd += reward[:n_ag].sum()
        adv_rwd += reward[n_ag:].sum()

        maddpg_ag.memory.push(obs[:n_ag].data, action_ag, next_obs_ag, reward[:n_ag])
        maddpg_adv.memory.push(obs[n_ag:].data, action_adv, next_obs_adv, reward[n_ag:])
        obs = next_obs

        critic_loss_ag, actor_loss_ag = maddpg_ag.update_policy()
        critic_loss_adv, actor_loss_adv = maddpg_adv.update_policy()

    maddpg_ag.episode_done += 1
    maddpg_adv.episode_done += 1

    if (i_episode + 1) % 1000 == 0:
        with open('ag_{}.pkl'.format(i_episode + 1), 'wb') as f:
            pickle.dump(maddpg_ag, f)
        with open('adv_{}.pkl'.format(i_episode + 1), 'wb') as f:
            pickle.dump(maddpg_adv, f)

    # print('Episode: %d, reward = %f' % (i_episode, total_reward))
    # print('Episode: %d, reward_ag = %f, reward_adv = %f' % (i_episode, ag_rwd, adv_rwd))
    # reward_record.append(total_reward)

    # if maddpg_ag.episode_done == maddpg_ag.episodes_before_train:
    #     print('training now begins...')
    #     print('scale_reward=%f\n' % scale_reward +
    #           'ag_reward: {}, adv_rwd: {}'.format(ag_rwd, adv_rwd))

    # if win is None:
    #     # win = vis.line(X=np.arange(i_episode, i_episode+1),
    #     #                Y=np.array([
    #     #                    np.append(total_reward, rr)]),
    #     #                opts=dict(
    #     #                    ylabel='Reward',
    #     #                    xlabel='Episode',
    #     #                    title='MADDPG on WaterWorld_mod\n' +
    #     #                    'agent=%d' % n_agents +
    #     #                    ', coop=%d' % n_coop +
    #     #                    ', sensor_range=0.2\n' +
    #     #                    'food=%f, poison=%f, encounter=%f' % (
    #     #                        food_reward,
    #     #                        poison_reward,
    #     #                        encounter_reward),
    #     #                    legend=['Total'] +
    #     #                    ['Agent-%d' % i for i in range(n_agents)]))
    # else:
    #     vis.line(X=np.array(
    #         [np.array(i_episode).repeat(n_agents+1)]),
    #              Y=np.array([np.append(total_reward,
    #                                    rr)]),
    #              win=win,
    #              update='append')
    # if param is None:
    #     param = vis.line(X=np.arange(i_episode, i_episode+1),
    #                      Y=np.array([maddpg.var[0]]),
    #                      opts=dict(
    #                          ylabel='Var',
    #                          xlabel='Episode',
    #                          title='MADDPG on WaterWorld: Exploration',
    #                          legend=['Variance']))
    # else:
    #     vis.line(X=np.array([i_episode]),
    #              Y=np.array([maddpg.var[0]]),
    #              win=param,
    #              update='append')

world_mpe.close()

# saving
f = h5py.File(f'dataset_simpletag_{n_ag}ag_{n_adv}adv', 'w')
f['actions'] = np.stack(action_data)
f['obs'] = np.stack(obs_data)
f['reward'] = np.stack(reward_data)
f['terminated'] = np.stack(terminated_data)

f.close()
