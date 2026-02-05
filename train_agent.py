from warehouse_env import WarehouseEnv
from dual_q_learning_agent import DualQAgent
from collections import deque

env = WarehouseEnv(render_mode=None)
agent = DualQAgent(env.action_space.n)


SAVE_INTERVAL = 500
EPISODES = 10_000

DATA_FOLDER = "training_data"
FILE_NAME = "warehouse_data.pkl"
agent.load_tables(DATA_FOLDER, FILE_NAME)


robot_prev_state = env.reset()


for ep in range(1, EPISODES + 1):

    print(f"episode: {ep}")
    obs, _ = env.reset()
    steps = 0
    done = False

    total_reward = 0.0
    successfully_dropped = False


    while not done:
      action = agent.select_action(obs)

      next_obs, term, trunc, _ = env.step(action)

      total_reward += env.reward

      agent.update(obs, action, env.reward, next_obs)

      obs = next_obs
      done = term or trunc
      successfully_dropped = term     # just for debugging
      steps += 1


    if ep % 2 == 0:
      print(f"Ep {ep} | total_reward: {total_reward:.4f} | steps: {steps} | "
      f"states: {len(agent.q_table)} Sucess: {successfully_dropped}")

    if ep % SAVE_INTERVAL == 0:
      print(f"Episode {ep}: Saving checkpoint...")
      agent.save_tables(DATA_FOLDER, FILE_NAME)

      print(f"Current States {len(agent.q_table)}")

# Final save
agent.save_tables(DATA_FOLDER, FILE_NAME)

# Things stable for now
