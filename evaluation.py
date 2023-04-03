import time

import numpy as np
import torch as th

from MADDPG import MADDPG
from madrl_environments.multiagent.make_env import make_env

world_mpe = make_env('simple_tag')
world_mpe.discrete_action_space = False

np.random.seed(42)
th.manual_seed(42)

n_ag = 2
n_adv = 4
n_states = 28
n_actions = 2

batch_size = 1000
capacity = 1_000_000
episodes_before_train = 100

maddpg_ag = MADDPG(n_ag, n_states, n_actions, batch_size, capacity, episodes_before_train, 'ag')
maddpg_adv = MADDPG(n_adv, n_states, n_actions, batch_size, capacity, episodes_before_train, 'adv')
FloatTensor = th.cuda.FloatTensor if maddpg_ag.use_cuda else th.FloatTensor
maddpg_ag.load(48000), maddpg_adv.load(48000)

eval_steps = 100
max_steps = 50
visualize = False

eval_rwds = [0, 0]
for _ in range(eval_steps):
    obs = world_mpe.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()

    ag_rwd, adv_rwd = 0, 0

    for t in range(max_steps):
        obs = obs.type(FloatTensor)
        action_ag = maddpg_ag.select_action(obs[:n_ag]).data.cpu()
        action_adv = maddpg_adv.select_action(obs[n_ag:]).data.cpu()
        action_tot = th.cat([action_ag, action_adv], dim=0)
        obs_, reward, done, _ = world_mpe.step(action_tot.numpy())

        ag_rwd += sum(reward[:n_ag])
        adv_rwd += sum(reward[n_ag:])

        if visualize:
            world_mpe.render(obs_, vis=True)
            time.sleep(.1)
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

        obs = next_obs

    eval_rwds[0] += ag_rwd
    eval_rwds[1] += adv_rwd
    # print('Good Rewards: {:.1f} || Adversary Rewards: {}'.format(ag_rwd, adv_rwd))

world_mpe.close()
print('Good Rewards: {:.1f} || Adversary Rewards: {:.1f}'.format(eval_rwds[0] / eval_steps, eval_rwds[1] / eval_steps))
