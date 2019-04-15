import gym

from .modified_taxi_environment import *


def get_frozen_lake_environment():
    return gym.make('FrozenLake8x8-v0')


def get_taxi_environment():
    return gym.make('Taxi-v2')


