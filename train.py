from warehouse_env import WarehouseEnv
from dual_q_learning_agent import DualQAgent

env = WarehouseEnv(render_mode=None)
agent = DualQAgent(env.action_space.n)


SAVE_INTERVAL = 2000
EPISODES = 6000

DATA_FOLDER = "training_data"
FILE_NAME = "warehouse_data.pkl"

# load existing knowledge if available
agent.load_tables(DATA_FOLDER, FILE_NAME)


for ep in range(1, EPISODES + 1):

    print(f"episode: {ep}")
    obs, _ = env.reset()
    steps = 0
    done = False

    total_reward = 0.0

    while not done:
      action = agent.select_action(obs)
      next_obs, reward, term, trunc, _ = env.step(action)

      total_reward += reward

      agent.update(obs, action, reward, next_obs)
      obs = next_obs
      done = term or trunc
      steps += 1

      if steps > 300:
        done = True
        reward -= 300


    if ep % 100 == 0:
      print(f"Ep {ep} | total_reward: {total_reward:.4f} | steps: {steps} | "
      f"states: pickup {len(agent.q_pickup)}, drop {len(agent.q_dropoff)}")

    if ep % SAVE_INTERVAL == 0:
      print(f"Episode {ep}: Saving checkpoint...")
      agent.save_tables(DATA_FOLDER, FILE_NAME)
      print(f"Current States - Pickup: {len(agent.q_pickup)}, Dropoff: {len(agent.q_dropoff)}")

# Final save
agent.save_tables(DATA_FOLDER, FILE_NAME)
