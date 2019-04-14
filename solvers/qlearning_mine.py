import numpy as np


class QLearningLearner:
    def __init__(self, iterations, epsilon_start,
                 epsilon_final, alpha_start,
                 alpha_final, gamma,
                 number_of_states,
                 number_of_actions,
                 start_state):
        self.Q = np.zeros([number_of_states, number_of_actions])
        self.epsilon = epsilon_start
        self.epsilon_decay = (epsilon_start - epsilon_final) / iterations
        self.alpha = alpha_start
        self.alpha_decay = (alpha_start - alpha_final) / iterations
        self.gamma = gamma
        self.number_of_actions = number_of_actions
        self.number_of_states = number_of_states
        self.start_state = start_state

    def plot_name(self):
        return "Q Learning"

    def predict_action(self, state):
        return np.argmax(self.Q[state])

    def next_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.number_of_actions)
        else:
            action = np.argmax(self.Q[state])
        self.epsilon = self.epsilon - self.epsilon_decay
        self.alpha = self.alpha - self.alpha_decay
        return action

    def learn(self, prev_state, action, reward0, state):
        prev_state_and_action = prev_state + (action,)
        self.Q[prev_state_and_action] = self.Q[prev_state_and_action] * (1 - self.alpha) + self.alpha * (
                (1 - self.gamma) * reward0 + self.gamma * np.max(self.Q[state]))






QLearningLearner(10000,1,0.01,1,0.01,0.9,)