from warehouse_env import WarehouseEnv
from dual_q_learning_agent import DualQAgent
import time

DATA_FOLDER = "training_data"
FILE_NAME = "warehouse_data.pkl"

env = WarehouseEnv(render_mode="human")
agent = DualQAgent(env.action_space.n)

agent.load_tables(DATA_FOLDER, FILE_NAME)

agent.epsilon = 0.01
# agent.epsilon = 0.1

EPISODES = 20

for ep in range(EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(obs)
        print("Action:", action)

        obs, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += env.reward
        time.sleep(0.05)

    print(f"Episode {ep+1} | Reward: {total_reward}")

env.close()
