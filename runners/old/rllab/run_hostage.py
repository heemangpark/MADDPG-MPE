#!/usr/bin/env python
#
# File: run_pursuit.py
#
# Created: Wednesday, July  6 2016 by megorov <mail@megorov.com>
#
from __future__ import absolute_import, print_function

import argparse
import json
import uuid
import datetime
import dateutil.tz
import os.path as osp
import ast

import gym
import numpy as np
import tensorflow as tf
from gym import spaces

from madrl_environments.pursuit import MAWaterWorld
from madrl_environments import StandardizedEnv
from rllabwrapper import RLLabEnv

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
from rllab.sampler import parallel_sampler
import rllab.misc.logger as logger
from rllab import config

def main():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    rand_id = str(uuid.uuid4())[:5]
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')
    default_exp_name = 'experiment_%s_%s' % (timestamp, rand_id)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_name', type=str, default=default_exp_name, help='Name of the experiment.')

    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--gae_lambda', type=float, default=0.99)

    parser.add_argument('--n_iter', type=int, default=250)
    parser.add_argument('--sampler_workers', type=int, default=1)
    parser.add_argument('--max_traj_len', type=int, default=250)
    parser.add_argument('--update_curriculum', action='store_true', default=False)
    parser.add_argument('--n_timesteps', type=int, default=8000)
    parser.add_argument('--control', type=str, default='centralized')

    parser.add_argument('--control', type=str, default='centralized')
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--n_good', type=int, default=3)
    parser.add_argument('--n_hostage', type=int, default=5)
    parser.add_argument('--n_bad', type=int, default=5)
    parser.add_argument('--n_coop_save', type=int, default=2)
    parser.add_argument('--n_coop_avoid', type=int, default=2)
    parser.add_argument('--n_sensors', type=int, default=20)
    parser.add_argument('--sensor_range', type=float, default=0.2)
    parser.add_argument('--save_reward', type=float, default=3)
    parser.add_argument('--hit_reward', type=float, default=-1)
    parser.add_argument('--encounter_reward', type=float, default=0.01)
    parser.add_argument('--bomb_reward', type=float, default=-10.)

    parser.add_argument('--recurrent', action='store_true', default=False)
    parser.add_argument('--baseline_type', type=str, default='linear')
    parser.add_argument('--policy_hidden_sizes', type=str, default='128,128')
    parser.add_argument('--baselin_hidden_sizes', type=str, default='128,128')

    parser.add_argument('--max_kl', type=float, default=0.01)

    parser.add_argument('--log_dir', type=str, required=False)
    parser.add_argument('--tabular_log_file', type=str, default='progress.csv',
                        help='Name of the tabular log file (in csv).')
    parser.add_argument('--text_log_file', type=str, default='debug.log',
                        help='Name of the text log file (in pure text).')
    parser.add_argument('--params_log_file', type=str, default='params.json',
                        help='Name of the parameter log file (in json).')
    parser.add_argument('--seed', type=int,
                        help='Random seed for numpy')
    parser.add_argument('--args_data', type=str,
                        help='Pickled data for stub objects')
    parser.add_argument('--snapshot_mode', type=str, default='all',
                        help='Mode to save the snapshot. Can be either "all" '
                             '(all iterations will be saved), "last" (only '
                             'the last iteration will be saved), or "none" '
                             '(do not save snapshots)')
    parser.add_argument('--log_tabular_only', type=ast.literal_eval, default=False,
                        help='Whether to only print the tabular log information (in a horizontal format)')


    args = parser.parse_args()

    parallel_sampler.initialize(n_parallel=args.sampler_workers)

    if args.seed is not None:
        set_seed(args.seed)
        parallel_sampler.set_seed(args.seed)

    args.hidden_sizes = tuple(map(int, args.policy_hidden_sizes.split(',')))

    centralized = True if args.control == 'centralized' else False

    sensor_range = np.array(map(float, args.sensor_range.split(',')))
    assert sensor_range.shape == (args.n_pursuers,)

    env = ContinuousHostageWorld(args.n_good, args.n_hostage, args.n_bad, args.n_coop_save,
                                 args.n_coop_avoid, n_sensors=args.n_sensors,
                                 sensor_range=args.sensor_range, save_reward=args.save_reward,
                                 hit_reward=args.hit_reward, encounter_reward=args.encounter_reward,
                                 bomb_reward=args.bomb_reward)

    env = RLLabEnv(StandardizedEnv(env), mode=args.control)

    if args.buffer_size > 1:
        env = ObservationBuffer(env, args.buffer_size)

    if args.recurrent:
        policy = GaussianGRUPolicy(env_spec=env.spec, hidden_sizes=args.hidden_sizes)
    else:
        policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=args.hidden_sizes)

    if args.baseline_type == 'linear':
        baseline = LinearFeatureBaseline(env_spec=env.spec)
    else:
        baseline = ZeroBaseline(obsfeat_space)

    # logger
    default_log_dir = config.LOG_DIR
    if args.log_dir is None:
        log_dir = osp.join(default_log_dir, args.exp_name)
    else:
        log_dir = args.log_dir
    tabular_log_file = osp.join(log_dir, args.tabular_log_file)
    text_log_file = osp.join(log_dir, args.text_log_file)
    params_log_file = osp.join(log_dir, args.params_log_file)

    logger.log_parameters_lite(params_log_file, args)
    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)
    prev_snapshot_dir = logger.get_snapshot_dir()
    prev_mode = logger.get_snapshot_mode()
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(args.snapshot_mode)
    logger.set_log_tabular_only(args.log_tabular_only)
    logger.push_prefix("[%s] " % args.exp_name)

    algo = TRPO(env=env,
            policy=policy,
            baseline=baseline,
            batch_size=args.n_timesteps,
            max_path_length=args.max_traj_len,
            n_itr=args.n_iter,
            discount=args.discount,
            step_size=args.max_kl,
            mode=args.control,)

    algo.train()



if __name__ == '__main__':
    main()
