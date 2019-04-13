import gym
from gym.envs.registration import register

from .modified_taxi_environment import *
from .modified_frozen_lake import *
from .modified_cliff_walking import *

__all__ = ['TaxiEnv', 'RewardingFrozenLakeEnv', 'CliffWalkingEnv']

register(
    id='TaxiEnv-v3',
    entry_point='environments:ModifiedTaxiEnvironment'
)

register(
    id='CliffWalkingEnv-v3',
    entry_point='environments:CliffWalkingEnv'
)

register(
    id='FrozenLakeEnv-v3',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8', 'is_slippery': False}
)


def get_frozen_lake_environment():
    return gym.make('FrozenLakeEnv-v3')


def get_taxi_environment():
    return gym.make('TaxiEnv-v3')


def get_cliffwalking_environment():
    return gym.make('CliffWalkingEnv-v3')
