import pickle
import pandas as pd

with open("./training_data/warehouse_data.pkl", "rb") as f:
    data = pickle.load(f)

print("Keys:", data.keys())
print("\nEpsilon:", data["epsilon"])
print("\nNumber of states:", len(data["q_table"]))

