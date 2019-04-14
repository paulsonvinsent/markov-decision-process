import argparse
import logging
import random as rand
from datetime import datetime

import numpy as np

import environments
import experiments
from experiments import qlearning_experiment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(experiment_detals, experiment, timing_key, verbose, timings):
    t = datetime.now()
    for details in experiment_detals:
        logger.info("Running {} experiment: {}".format(timing_key, details.env_readable_name))
        exp = experiment(details, verbose=verbose)
        exp.perform()
    t_d = datetime.now() - t
    timings[timing_key] = t_d.seconds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MDP experiments')
    parser.add_argument('--seed', type=int, help='A random seed to set, if desired')
    parser.add_argument('--policy', action='store_true', help='Run the policy iteration experiment')
    parser.add_argument('--value', action='store_true', help='Run the value iteration experiment')
    parser.add_argument('--q', action='store_true', help='Run the Q-Learner experiment')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--plot', action='store_true', help='Plot data results')
    parser.add_argument('--verbose', action='store_true', help='If true, provide verbose output')
    args = parser.parse_args()
    verbose = args.verbose

    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, (2 ** 32) - 1)
        logger.info("Using seed {}".format(seed))
        np.random.seed(seed)
        rand.seed(seed)

    logger.info("Creating MDPs")
    logger.info("----------")

    envs = [
        {
            'env': environments.get_frozen_lake_environment(),
            'name': 'frozen_lake',
            'readable_name': 'Frozen Lake (8x8)',
            'state_to_track': 0
        },
        {'env': environments.get_taxi_environment(),
         'name': 'taxi',
         'readable_name': 'Taxi problem',
         'state_to_track': 14
         }

    ]

    experiment_details = []
    for env in envs:
        env['env'].seed(seed)
        logger.info('{}: State space: {}, Action space: {}'.format(env['readable_name'], env['env'].unwrapped.nS,
                                                                   env['env'].unwrapped.nA))
        experiment_details.append(experiments.ExperimentDetails(
            env['env'], env['name'], env['readable_name'],
            seed=seed
        ))

    if verbose:
        logger.info("----------")
    logger.info("Running experiments")

    timings = {}

    if args.policy or args.all:
        run_experiment(experiment_details, experiments.PolicyIterationExperiment, 'PI', verbose, timings)

    if args.value or args.all:
        run_experiment(experiment_details, experiments.ValueIterationExperiment, 'VI', verbose, timings)

    if args.q or args.all:
        for env_desc in envs:
            env = env_desc['env']
            state_to_track = env_desc['state_to_track']
            qlearning_experiment.run_qlearning_experiment(env_desc['name'], env, state_to_track)

    logger.info(timings)
