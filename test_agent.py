from warehouse_env import WarehouseEnv
from dual_q_learning_agent import DualQAgent
import time

DATA_FOLDER = "training_data"
FILE_NAME = "warehouse_data.pkl"

env = WarehouseEnv(render_mode="human")
agent = DualQAgent(env.action_space.n)

agent.load_tables(DATA_FOLDER, FILE_NAME)

# 🔒 disable exploration
agent.epsilon = 0.0

EPISODES = 20

for ep in range(EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(obs)
        print("Action:", action)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        time.sleep(0.05)

    print(f"Episode {ep+1} | Reward: {total_reward}")

env.close()
