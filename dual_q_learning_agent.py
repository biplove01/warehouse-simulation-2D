import pickle
import random
import numpy as np
import os

class DualQAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.q_table = {}                   # ← single table now
        # self.lr = 0.75
        self.lr = 0.85
        self.gamma = 0.95
        # self.epsilon = 1.0
        # self.epsilon = 0.45
        self.epsilon = 0.25
        self.eps_decay = 0.99999
        # self.eps_min = 0.02
        self.eps_min = 0.005

    def _get_state(self, obs):
        robot_direction = int(obs[7])
        loaded = int(obs[2])                # 0 or 1
        rx, ry     = int(obs[0]), int(obs[1])
        bx, by     = int(obs[3]), int(obs[4])
        dx, dy     = int(obs[5]), int(obs[6])
        return (rx, ry, loaded, robot_direction, bx, by, dx, dy)

    def select_action(self, obs):
        state = self._get_state(obs)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim)
        return int(np.argmax(self.q_table[state]))

    def update(self, obs, action, reward, next_obs, done=False):
        state      = self._get_state(obs)
        next_state = self._get_state(next_obs)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_dim)

        old_value = self.q_table[state][action]
        next_max  = 0 if done else np.max(self.q_table[next_state])
        target    = reward + self.gamma * next_max
        self.q_table[state][action] = old_value + self.lr * (target - old_value)

        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

    def save_tables(self, folder, filename):
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, filename)
        data = {
            "q_table": self.q_table,
            "epsilon": self.epsilon
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved to {path}")

    def load_tables(self, folder, filename):
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.q_table  = data["q_table"]
            self.epsilon  = data.get("epsilon", 1.0)
            print(f"Loaded from {path}")
        else:
            print("No saved data found.")
