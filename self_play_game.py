from warehouse_env import WarehouseEnv

env = WarehouseEnv(render_mode="human")
obs, info = env.reset()

for _ in range(1000):
  action = env.action_space.sample()  # Agent policy goes here
  obs, reward, terminated, truncated, info = env.step(action)

  if terminated or truncated:
    obs, info = env.reset()

env.close()
