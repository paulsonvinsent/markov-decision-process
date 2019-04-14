import csv
import os
import time

import numpy as np

OUTPUT_DIRECTORY = './output/Q'

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)


# taken from https://github.com/Lodur03/Q-Learning-Taxi-v2/blob/master/Q-Learning-Taxi.ipynb
def random_q_learning(name, env, state_to_track, episodes=30000, max_steps=1000, lr=0.3,
                      decay_fac=0.00001, gamma=0.90,
                      play_solution=False):
    initial_lr = lr
    # Number of possible actions
    action_size = env.action_space.n

    # Number of possible states
    state_size = env.observation_space.n

    qtable = np.zeros((state_size, action_size))

    start_time = int(round(time.time() * 1000))

    stats = []

    for episode in range(episodes):

        state = env.reset()  # Reset the environment
        done = False  # Are we done with the environment
        lr -= decay_fac  # Decaying learning rate
        step = 0
        episode_reward = 0.0

        if lr <= 0:  # Nothing more to learn?
            break

        for step in range(max_steps):

            # Randomly Choose an Action
            action = env.action_space.sample()

            # Take the action -> observe new state and reward
            new_state, reward, done, info = env.step(action)
            episode_reward = episode_reward + reward

            # Update qtable values
            if done == True:  # If last, do not count future accumulated reward
                if (step < 199 | step > 201):
                    qtable[state, action] = qtable[state, action] + lr * (reward + gamma * 0 - qtable[state, action])
                break
            else:  # Consider accumulated reward of best decision stream
                qtable[state, action] = qtable[state, action] + lr * (
                        reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

            # if done.. jump to next episode
            if done == True:
                break

            # moving states
            state = new_state

        current_time = int(round(time.time() * 1000)) - start_time
        state_value = np.max(qtable[state_to_track, :])
        stats.append([episode, current_time, state_value, episode_reward])

        episode += 1

        if (episode % (episodes / 10) == 0):
            print('episode = ', episode)
            print('learning rate = ', lr)
            print('-----------')

    if (play_solution):
        # New environment
        state = env.reset()
        env.render()
        done = False
        total_reward = 0
        while (done == False):
            action = np.argmax(qtable[state, :])  # Choose best action (Q-table)
            state, reward, done, info = env.step(action)  # Take action
            total_reward += reward  # Summing rewards

            # Display it
            time.sleep(0.5)
            env.render()
            print('Episode Reward = ', total_reward)
    with open("{}/randomq_name_{}_lr_{}_gamma_{}.csv".format(OUTPUT_DIRECTORY, name, initial_lr, gamma), "w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(stats)
    return stats


def epsilon_greedy_q_learning(name, env, state_to_track, episodes=30000, max_steps=1000, lr=0.3,
                              decay_fac=0.00001, gamma=0.90,
                              play_solution=False):
    initial_lr=lr
    # Number of possible actions
    action_size = env.action_space.n

    # Number of possible states
    state_size = env.observation_space.n

    qtable = np.zeros((state_size, action_size))
    epsilon = 1.0
    epsilon_decay = (1.0 - 0.01) / episodes

    start_time = int(round(time.time() * 1000))

    stats = []

    for episode in range(episodes):

        state = env.reset()  # Reset the environment
        done = False  # Are we done with the environment
        lr -= decay_fac  # Decaying learning rate
        step = 0
        episode_reward = 0.0

        if lr <= 0:  # Nothing more to learn?
            break

        for step in range(max_steps):

            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])
            # Randomly Choose an Action

            # Take the action -> observe new state and reward
            new_state, reward, done, info = env.step(action)
            episode_reward = episode_reward + reward

            # Update qtable values
            if done == True:  # If last, do not count future accumulated reward
                if (step < 199 | step > 201):
                    qtable[state, action] = qtable[state, action] + lr * (reward + gamma * 0 - qtable[state, action])
                break
            else:  # Consider accumulated reward of best decision stream
                qtable[state, action] = qtable[state, action] + lr * (
                        reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

            # if done.. jump to next episode
            if done == True:
                break

            # moving states
            state = new_state
        epsilon = epsilon - epsilon_decay

        current_time = int(round(time.time() * 1000)) - start_time
        state_value = np.max(qtable[state_to_track, :])
        stats.append([episode, current_time, state_value, episode_reward])

        episode += 1

        if (episode % (episodes / 10) == 0):
            print('episode = ', episode)
            print('learning rate = ', lr)
            print('-----------')

    if (play_solution):
        # New environment
        state = env.reset()
        env.render()
        done = False
        total_reward = 0
        while (done == False):
            action = np.argmax(qtable[state, :])  # Choose best action (Q-table)
            state, reward, done, info = env.step(action)  # Take action
            total_reward += reward  # Summing rewards

            # Display it
            time.sleep(0.5)
            env.render()
            print('Episode Reward = ', total_reward)
    with open("{}/eps_greedy_name_{}_lr_{}_gamma_{}.csv".format(OUTPUT_DIRECTORY, name, initial_lr, gamma), "w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(stats)
    return stats


# env = gym.make('FrozenLake8x8-v0')
# # env = gym.make("Taxi-v2")  # Create environment
# state_to_track = 14
# env.render()
# print(env.reset())
# env.render()
#
# lr = 0.3  # Learning rate
# decay_fac = 0.000001  # Decay learning rate each iteration
# gamma = 0.90  # Discounting rate - later rewards impact less
# episodes = 30000  # Total episodes
# max_steps = 1000  # Max steps per episode
#
# epsilon_greedy_q_learning('frozen_lake', env, state_to_track, episodes, max_steps, lr, decay_fac, gamma, True)

episodes = 30000  # Total episodes
max_steps = 1000  # Max steps per episode
decay_fac = 0.000001  # Decay learning rate each iteration


def run_qlearning_experiment(name, env, state_to_track=0):
    for lr in [0.1, 0.3, 0.6, 0.9]:
        for gamma in [0.5, 0.7, 0.9, 0.99]:
            print("Running experiment {} for lr ={}  gamma ={}".format(name, lr, gamma))
            print("{} :Running Epsilon greedy q learning", name)
            epsilon_greedy_q_learning(name, env, state_to_track, episodes, max_steps, lr, decay_fac, gamma,
                                      False)
            print("{} : Running Random search q learning,name")
            random_q_learning(name, env, state_to_track, episodes, max_steps, lr, decay_fac, gamma,
                              False)
