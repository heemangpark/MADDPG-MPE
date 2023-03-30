#!/usr/bin/env python
#
# File: run_multiwalker.py
#
# Created: Friday, September  2 2016 by rejuvyesh <mail@rejuvyesh.com>
#
from runners import RunnerParser
from runners.curriculum import Curriculum

from madrl_environments.walker.multi_walker import MultiWalkerEnv
from madrl_environments.mujoco.ant.multi_ant import MultiAnt
from madrl_environments import StandardizedEnv, ObservationBuffer

# yapf: disable
ENV_OPTIONS = [
    ('n_legs', int, 4, ''),
    ('ts', float, 0.02, ''),
    ('integrator', str, 'RK4', ''),
    ('leg_length', float, 0.282, ''),
    ('out_file', str, 'multi_ant.xml', ''),
    ('base_file', str, 'ant_og.xml', ''),
    ('reward_mech', str, 'local', ''),
    ('buffer_size', int, 1, ''),
    ('curriculum', str, None, ''),
]
# yapf: enable

def main(parser):
    mode = parser._mode
    args = parser.args

    env = MultiAnt(n_legs=args.n_legs, ts=args.ts, integrator=args.integrator,
                   leg_length=args.leg_length, out_file=args.out_file,
                   base_file=args.base_file, reward_mech=args.reward_mech)

    if args.buffer_size > 1:
        env = ObservationBuffer(env, args.buffer_size)

    if mode == 'rllab':
        from runners.rurllab import RLLabRunner
        run = RLLabRunner(env, args)
    elif mode == 'rltools':
        from runners.rurltools import RLToolsRunner
        run = RLToolsRunner(env, args)
    else:
        raise NotImplementedError()

    if args.curriculum:
        curr = Curriculum(args.curriculum)
        run(curr)
    else:
        run()

if __name__ == '__main__':
    main(RunnerParser(ENV_OPTIONS))
