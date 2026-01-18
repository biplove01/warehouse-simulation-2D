import pickle
import pandas as pd

with open("./training_data/warehouse_data.pkl", "rb") as f:
    data = pickle.load(f)

print("Keys:", data.keys())
print("\nEpsilon:", data["epsilon"])
print("\nNumber of states:", len(data["q_table"]))



# import pickle
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the data
# with open("./training_data/warehouse_data.pkl", "rb") as f:
#     data = pickle.load(f)

# q_table = data['q_table']

# # If q_table is a dictionary, convert to a 2D array
# if isinstance(q_table, dict):
#     # Assuming states are keys and values are arrays of actions
#     states = list(q_table.keys())
#     q_values = np.array([q_table[s] for s in states])
# else:
#     q_values = q_table

# # Plotting
# plt.figure(figsize=(12, 8))
# sns.heatmap(q_values, annot=False, cmap='viridis')
# plt.xlabel('Actions')
# plt.ylabel('States')
# plt.title(f"Q-Table Heatmap (Epsilon: {data['epsilon']:.4f})")
# plt.savefig('q_table_viz.png')
