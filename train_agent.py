from warehouse_env import WarehouseEnv
from dual_q_learning_agent import DualQAgent
import copy

env = WarehouseEnv(render_mode=None)
agent0 = DualQAgent(env.action_space.n)
agent1 = DualQAgent(env.action_space.n)

SAVE_INTERVAL = 20
EPISODES = 2_000
DATA_FOLDER = "training_data"
FILE_NAME = "warehouse_data.pkl"

# Load shared policy if exists
agent0.load_tables(DATA_FOLDER, FILE_NAME)
agent1.q_table = copy.deepcopy(agent0.q_table)
agent1.epsilon = agent0.epsilon

for ep in range(1, EPISODES + 1):
    print(f"episode: {ep}")
    obs_list, _ = env.reset()
    obs0, obs1 = obs_list
    steps = 0
    done = False
    total_reward = 0.0

    while not done:
        action0 = agent0.select_action(obs0)
        action1 = agent1.select_action(obs1)
        next_obs_list, reward, terminated, truncated, _ = env.step([action0, action1])
        next_obs0, next_obs1 = next_obs_list

        agent0.update(obs0, action0, reward, next_obs0, done=(terminated or truncated))
        agent1.update(obs1, action1, reward, next_obs1, done=(terminated or truncated))

        obs0, obs1 = next_obs0, next_obs1
        total_reward += reward
        done = terminated or truncated
        steps += 1

    if ep % 2 == 0:
        print(f"Ep {ep} | total_reward: {total_reward:.4f} | steps: {steps} | "
              f"states0: {len(agent0.q_table)} states1: {len(agent1.q_table)} Score: {env.score}")

    if ep % SAVE_INTERVAL == 0:
        print(f"Episode {ep}: Saving checkpoint...")
        agent0.save_tables(DATA_FOLDER, FILE_NAME)  # Save primary agent

print(f"Final states0: {len(agent0.q_table)}")
agent0.save_tables(DATA_FOLDER, FILE_NAME)