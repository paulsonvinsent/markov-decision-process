import matplotlib.pyplot as plt
import pandas as pd

# Frozen lake specific stuff
columns = ['episode', 'current_time', 'state_value', 'episode_reward', 'learning_rate']

gamma_values = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
iterations_for_convergence = [2, 4, 4, 12, 4, 4, 3]
time_for_convergence = [7, 20, 24, 120, 92, 144, 251]

env_name = 'Frozen Lake'

plt.figure()
plt.title("{} - Gamma Vs Iterations and Time for Convergence".format('{} Policy Iteration'.format(env_name)))
plt.grid()
plt.xlabel("Gamma")
plt.ylabel("Iterations for convergence")
plt.plot(gamma_values, iterations_for_convergence, 'o-', color="r")
for i, txt in enumerate(time_for_convergence):
    plt.annotate(txt, (gamma_values[i], iterations_for_convergence[i]))
plt.legend(loc="best")
plt.show()

data = pd.read_csv('/Users/pvincent/Desktop/markov-decision-process/output/PI/frozen_lake_0.99_episodes.csv')

steps_array = data[['steps']]
statevalue_array = data[['value_of_state']]

plt.figure()
plt.title("{} - Iteration vs Value of Candidate State (Gamma 0.99)".format('{} Policy Iteration'.format(env_name)))
plt.grid()
plt.xlabel("Iterations")
plt.ylabel("State Value")
plt.plot(steps_array, statevalue_array, 'o-', color="r")
plt.legend(loc="best")
plt.show()

iterations_for_convergence = [5, 8, 11, 19, 52, 85, 206]
time_for_convergence = [6, 10, 14, 24, 64, 110, 255]

plt.figure()
plt.title("{} - Gamma Vs Iterations and Time for Convergence".format('{} Value Iteration'.format(env_name)))
plt.grid()
plt.xlabel("Gamma")
plt.ylabel("Iterations for convergence")
plt.plot(gamma_values, iterations_for_convergence, 'o-', color="r")
for i, txt in enumerate(time_for_convergence):
    plt.annotate(txt, (gamma_values[i], iterations_for_convergence[i]))
plt.legend(loc="best")
plt.show()

data = pd.read_csv('/Users/pvincent/Desktop/markov-decision-process/output/VI/frozen_lake_0.99.csv')

steps_array = data[['steps']]
statevalue_array = data[['value_of_state']]
delta_array = data[['delta']]

plt.figure()
plt.title("{} - Iteration vs Value of Candidate State".format('{} Value Iteration'.format(env_name)))
plt.grid()
plt.xlabel("Iterations")
plt.ylabel("State Value")
plt.plot(steps_array, statevalue_array, 'o-', color="r")
plt.legend(loc="best")
plt.show()

plt.figure()
plt.title("{} - Iteration vs Delta".format('{} Value Iteration'.format(env_name)))
plt.grid()
plt.xlabel("Iterations")
plt.ylabel("Delta")
plt.plot(steps_array, delta_array, 'o-', color="r")
plt.legend(loc="best")
plt.show()

# Effect of learning rate

data1 = pd.read_csv(
    '/Users/pvincent/Desktop/markov-decision-process/output/Q/eps_greedy_name_frozen_lake_lr_0.9_gamma_0.99.csv',
    names=columns, header=None)

data2 = pd.read_csv(
    '/Users/pvincent/Desktop/markov-decision-process/output/Q/eps_greedy_name_frozen_lake_lr_0.6_gamma_0.99.csv',
    names=columns, header=None)

data3 = pd.read_csv(
    '/Users/pvincent/Desktop/markov-decision-process/output/Q/eps_greedy_name_frozen_lake_lr_0.3_gamma_0.99.csv',
    names=columns, header=None)

data4 = pd.read_csv(
    '/Users/pvincent/Desktop/markov-decision-process/output/Q/eps_greedy_name_frozen_lake_lr_0.1_gamma_0.99.csv',
    names=columns, header=None)

episodes = data1['episode']
statevalue1 = data1['state_value']
statevalue2 = data2['state_value']
statevalue3 = data3['state_value']
statevalue4 = data4['state_value']

plt.figure()
plt.title("{} - Episodes Vs Value of a state (Gamma 0.99)".format('Frozen Lake Q Learning'))
plt.grid()
plt.xlabel("Episodes")
plt.ylabel("State Value")
plt.plot(episodes, statevalue1, 'o-', color="r", label="Learning rate 0.9")
plt.plot(episodes, statevalue2, 'o-', color="g", label="Learning rate 0.6")
plt.plot(episodes, statevalue3, 'o-', color="b", label="Learning rate 0.3")
plt.plot(episodes, statevalue4, 'o-', color="c", label="Learning rate 0.1")
plt.legend(loc="best")
plt.show()

time1 = data1['current_time']
time2 = data2['current_time']
time3 = data3['current_time']
time4 = data4['current_time']

plt.figure()
plt.title("{} - Episodes Vs Time".format('Frozen Lake Q Learning'))
plt.grid()
plt.xlabel("Episodes")
plt.ylabel("Time")
plt.plot(episodes, time1, 'o-', color="r", label="Learning rate 0.9")
plt.plot(episodes, time2, 'o-', color="g", label="Learning rate 0.6")
plt.plot(episodes, time3, 'o-', color="b", label="Learning rate 0.3")
plt.plot(episodes, time4, 'o-', color="c", label="Learning rate 0.1")
plt.legend(loc="best")
plt.show()

selected_learning_rate = 0.3

data1 = pd.read_csv(
    '/Users/pvincent/Desktop/markov-decision-process/output/Q/eps_greedy_name_frozen_lake_lr_0.3_gamma_0.99.csv',
    names=columns, header=None)

data2 = pd.read_csv(
    '/Users/pvincent/Desktop/markov-decision-process/output/Q/randomq_name_frozen_lake_lr_0.3_gamma_0.99.csv',
    names=columns, header=None)
episodes = data1['episode']
statevalue1 = data1['state_value']
statevalue2 = data2['state_value']

plt.figure()
plt.title("{} - Random Search Vs Epsilon Greedy (Gamma 0.99)".format('Frozen Lake Q Learning'))
plt.grid()
plt.xlabel("Episodes")
plt.ylabel("State Value")
plt.plot(episodes, statevalue1, 'o-', color="r", label="Epsilon Greedy")
plt.plot(episodes, statevalue2, 'o-', color="g", label="Random ")
plt.legend(loc="best")
plt.show()
