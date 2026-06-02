import numpy as np
import random
import pickle
import os


class DualQAgent:
    def __init__(self, action_dim, lr=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Core dictionary mapping state tuples to arrays of Q-values
        self.q_table = {}

    def _get_state(self, obs):
        """
        Extracts the specific 6-field state tuple from the 8-feature observation array.
        Expected observation layout:
        [rx, ry, loaded, bx, by, dropoff_x, dropoff_y, direction]
        """
        rx = int(round(obs[0]))
        ry = int(round(obs[1]))
        loaded = int(round(obs[2]))

        # Target shelf coordinates
        bx = int(round(obs[3]))
        by = int(round(obs[4]))

        # Robot's last movement direction
        direction = int(round(obs[7]))

        # The Q-Table relies on this exact tuple signature
        return (rx, ry, loaded, direction, bx, by)

    def select_action(self, obs):
        state = self._get_state(obs)

        # Exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        # Exploitation
        if state not in self.q_table:
            # Pick a random action for unseen states
            return random.randint(0, self.action_dim - 1)

        return int(np.argmax(self.q_table[state]))

    def update(self, obs, action, reward, next_obs):
        state = self._get_state(obs)
        next_state = self._get_state(next_obs)

        # Lazily initialize unseen states with zeros
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_dim)

        # Standard Q-Learning Bellman equation
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]

        self.q_table[state][action] += self.lr * td_error

    def save_tables(self, folder, filename):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_tables(self, folder, filename):
        path = os.path.join(folder, filename)
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)

        # Turn off exploration once tables are loaded
        self.epsilon = 0.0