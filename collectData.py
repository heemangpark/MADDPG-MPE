import h5py
import numpy as np
import torch as th
from tqdm import trange

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
capacity = 1000000
batch_size = 1000

n_episode = 100_000
max_steps = 50
episodes_before_train = 100

maddpg_ag = MADDPG(n_ag, n_states, n_actions, batch_size, capacity, episodes_before_train, 'ag')
maddpg_adv = MADDPG(n_adv, n_states, n_actions, batch_size, capacity, episodes_before_train, 'adv')
FloatTensor = th.cuda.FloatTensor if maddpg_ag.use_cuda else th.FloatTensor
maddpg_ag.load(48000), maddpg_adv.load(48000)

# Dataset saving
action_data_tot = []
obs_data_tot = []
reward_data_tot = []
terminated_data_tot = []

for i_episode in trange(n_episode):
    action_data = []
    obs_data = []
    reward_data = []
    terminated_data = []
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

        # saving
        action_data.append(action_tot.numpy())
        obs_data.append(obs.numpy())
        reward_data.append(reward.numpy())
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

        obs = next_obs

    action_data_tot.append(action_data)
    reward_data_tot.append(reward_data)
    terminated_data_tot.append(terminated_data)
    obs_data_tot.append(obs_data)

world_mpe.close()

# saving
f = h5py.File(f'dataset_simpletag_{n_ag}ag_{n_adv}adv.h5', 'w')
f['actions'] = np.stack(action_data_tot)
f['obs'] = np.stack(obs_data_tot)
f['reward'] = np.stack(reward_data_tot)
f['terminated'] = np.stack(terminated_data_tot)

f.close()
