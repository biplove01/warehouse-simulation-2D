import numpy as np
import random
import pickle
import os

class DualQAgent:
  def __init__(self, action_dim):
    self.action_dim = action_dim
    # Table 1: State -> [RobotX, RobotY, BoxX, BoxY]
    self.q_pickup = {}
    # Table 2: State -> [RobotX, RobotY, DropX, DropY]
    self.q_dropoff = {}

    self.lr = 0.1
    self.gamma = 0.95
    self.epsilon = 1.0 # for training
    # self.epsilon = 0 # for testing
    self.eps_decay = 0.9995
    self.eps_min = 0.05


  def _get_table_and_state(self, obs):

    robot_direction = int(obs[7])

    loaded = bool(obs[2])
    rx, ry = int(obs[0]), int(obs[1])
    bx, by = int(obs[3]), int(obs[4])
    dx, dy = int(obs[5]), int(obs[6])
    if not loaded:
        return self.q_pickup, (rx, ry, robot_direction, bx, by)
    else:
        return self.q_dropoff, (rx, ry, robot_direction, dx, dy)


  def select_action(self, obs):
      table, state = self._get_table_and_state(obs)
      # print("State:", state)          # ← add this
      if random.random() < self.epsilon:
          action = random.randint(0, self.action_dim - 1)
      else:
          if state not in table:
              table[state] = np.zeros(self.action_dim)
          action = int(np.argmax(table[state]))
      # print("Chosen action:", action)  # ← add this
      return action


  def update(self, obs, action, reward, next_obs):
    table, state = self._get_table_and_state(obs)
    next_table, next_state = self._get_table_and_state(next_obs)

    if state not in table:
      table[state] = np.zeros(self.action_dim)
    if next_state not in next_table:
      next_table[next_state] = np.zeros(self.action_dim)

    # Q-update
    old_value = table[state][action]
    next_max = np.max(next_table[next_state])

    table[state][action] = old_value + self.lr * (reward + self.gamma * next_max - old_value)

    if self.epsilon > self.eps_min:
      self.epsilon *= self.eps_decay


  def save_tables(self, folder, filename):
    if not os.path.exists(folder):
      os.makedirs(folder)

    path = os.path.join(folder, filename)
    data = {
      "pickup": self.q_pickup,
      "dropoff": self.q_dropoff,
      "epsilon": self.epsilon
    }
    with open(path, "wb") as f:
      pickle.dump(data, f)
    print(f"Tables saved to {path}")


  def load_tables(self, folder, filename):
    path = os.path.join(folder, filename)
    if os.path.exists(path):
      with open(path, "rb") as f:
        data = pickle.load(f)
        self.q_pickup = data["pickup"]
        self.q_dropoff = data["dropoff"]
        self.epsilon = data.get("epsilon", 1.0) # commented for testing
      print(f"Tables loaded from {path}")
    else:
      print("No saved data found in folder. Starting fresh.")
