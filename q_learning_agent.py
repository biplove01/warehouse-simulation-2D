import numpy as np
import pickle
import random

class QLearningAgent:
  def __init__(self, action_dim):
    self.q_table = {}
    self.action_dim = action_dim
    self.lr = 0.1
    self.gamma = 0.95
    self.epsilon = 1
    self.eps_decay = 0.9995
    self.eps_min = 0.01


  def get_q_values(self, state):
    if state not in self.q_table:
      self.q_table[state] = np.zeros(self.action_dim)
    return self.q_table[state]


  def select_action(self, state):
    if random.random() < self.epsilon:
      return random.randint(0, self.action_dim - 1)
    return np.argmax(self.get_q_values(state))


  def update(self, state, action, reward, next_state):
    current_q = self.get_q_values(state)[action]
    max_next_q = np.max(self.get_q_values(next_state))

    # Q-Learning Formula: Q(s,a) = Q(s,a) + lr * [R + gamma * maxQ(s',a') - Q(s,a)]
    new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
    self.q_table[state][action] = new_q

    if self.epsilon > self.eps_min:
      self.epsilon *= self.eps_decay
