import time
import copy
from warehouse_env import WarehouseEnv
from dual_q_learning_agent import DualQAgent

DATA_FOLDER = "training_data"
FILE_NAME = "warehouse_data.pkl"

env = WarehouseEnv(render_mode="human")

# Primary agent (loads the trained table)
agent = DualQAgent(env.action_space.n)
agent.load_tables(DATA_FOLDER, FILE_NAME)
agent.epsilon = 0.95  # Low exploration for testing

# Second robot shares the exact same policy
agent2 = DualQAgent(env.action_space.n)
agent2.q_table = copy.deepcopy(agent.q_table)
agent2.epsilon = 0.01

EPISODES = 20

for ep in range(EPISODES):
    obs_list, _ = env.reset()
    obs_robot1, obs_robot2 = obs_list  # Order matches robot index (robot0 first)

    done = False
    total_reward = 0.0

    print(f"Episode {ep+1} started. Click shelves to add boxes (manual tasks).")

    while not done:
        action1 = agent.select_action(obs_robot1)
        action2 = agent2.select_action(obs_robot2)

        print(f"Actions - Robot1: {action1}, Robot2: {action2}")

        next_obs_list, reward, terminated, truncated, _ = env.step([action1, action2])

        obs_robot1, obs_robot2 = next_obs_list

        total_reward += reward
        done = terminated or truncated

        time.sleep(0.05)  # Smooth rendering

    print(f"Episode {ep+1} finished | Total Reward: {total_reward:.2f} | Deliveries: {env.score}")

env.close()