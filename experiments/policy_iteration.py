import json
import os
import time

import numpy as np

import solvers
from .base import BaseExperiment, OUTPUT_DIRECTORY

if not os.path.exists(OUTPUT_DIRECTORY + '/PI'):
    os.makedirs(OUTPUT_DIRECTORY + '/PI')


class PolicyIterationExperiment(BaseExperiment):
    def __init__(self, details, verbose=False):
        super(PolicyIterationExperiment, self).__init__(details, verbose)

    def convergence_check_fn(self, solver, step_count):
        return solver.has_converged()

    def perform(self):
        # Policy iteration
        self._details.env.reset()

        grid_file_name = '{}/PI/{}_grid.csv'.format(OUTPUT_DIRECTORY, self._details.env_name)
        with open(grid_file_name, 'w') as f:
            f.write("params,time,steps,reward_mean,reward_median,reward_min,reward_max,reward_std\n")

        discount_factors = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99])
        dims = len(discount_factors)
        self.log("Searching PI in {} dimensions".format(dims))

        runs = 1
        for discount_factor in discount_factors:
            t = int(round(time.time() * 1000))
            self.log("{}/{} Processing PI with discount factor {}".format(runs, dims, discount_factor))

            p = solvers.PolicyIterationSolver(self._details.env, discount_factor=discount_factor,
                                              max_policy_eval_steps=3000, verbose=self._verbose)

            stats = self.run_solver_and_collect(p, self.convergence_check_fn)

            self.log("Took {} steps".format(len(stats.steps)))
            stats.to_csv('{}/PI/{}_{}_episodes.csv'.format(OUTPUT_DIRECTORY, self._details.env_name, discount_factor))
            optimal_policy_stats = self.run_policy_and_collect(p, stats.optimal_policy, times=100)
            with open(grid_file_name, 'a') as f:
                f.write('"{}",{},{},{},{},{},{},{}\n'.format(
                    json.dumps({'discount_factor': discount_factor}).replace('"', '""'),
                    int(round(time.time() * 1000)) - t,
                    len(optimal_policy_stats.rewards),
                    optimal_policy_stats.reward_mean,
                    optimal_policy_stats.reward_median,
                    optimal_policy_stats.reward_min,
                    optimal_policy_stats.reward_max,
                    optimal_policy_stats.reward_std,
                ))
            runs += 1
