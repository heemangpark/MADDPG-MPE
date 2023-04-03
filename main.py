import numpy as np
import torch as th
import wandb

from MADDPG import MADDPG
from madrl_environments.multiagent.make_env import make_env

np.random.seed(42)
th.manual_seed(42)

discrete = True

world_mpe = make_env('simple_tag')
world_mpe.discrete_action_input = False

if discrete:
    world_mpe.discrete_action_input = True

n_ag = 2
n_adv = 4
n_states = 28
n_actions = 4 if discrete else 2
capacity = 1_000_000
batch_size = 1000

n_episode = 100_000
max_steps = 50
episodes_before_train = 100

maddpg_ag = MADDPG(n_ag, n_states, n_actions, batch_size, capacity, episodes_before_train, 'ag', discrete=discrete)
maddpg_adv = MADDPG(n_adv, n_states, n_actions, batch_size, capacity, episodes_before_train, 'adv', discrete=discrete)

FloatTensor = th.cuda.FloatTensor if maddpg_ag.use_cuda else th.FloatTensor

w_plot = True
if w_plot:
    wandb.init()

for i_episode in range(n_episode):
    obs = world_mpe.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()

    ag_rwd = 0
    adv_rwd = 0

    for t in range(max_steps):
        obs = obs.type(FloatTensor)
        action_ag = maddpg_ag.select_action(obs[:n_ag]).data.cpu()
        action_adv = maddpg_adv.select_action(obs[n_ag:]).data.cpu()
        action_tot = th.cat([action_ag, action_adv], dim=0)
        obs_, reward, done, _ = world_mpe.step(action_tot.numpy())

        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
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

        _, _ = maddpg_ag.update_policy()
        _, _ = maddpg_adv.update_policy()

    maddpg_ag.episode_done += 1
    maddpg_adv.episode_done += 1

    if w_plot:
        wandb.log({'action_ag': ag_rwd, 'action_adv': adv_rwd})

    if (i_episode + 1) % 2000 == 0:
        maddpg_ag.save(i_episode + 1), maddpg_adv.save(i_episode + 1)

world_mpe.close()
