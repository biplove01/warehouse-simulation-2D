import pickle
import pandas as pd

with open("./training_data/warehouse_data.pkl", "rb") as f:
    data = pickle.load(f)

print("Keys:", data.keys())
print("\nEpsilon:", data["epsilon"])
print("\nNumber of states:", len(data["q_table"]))
# df = pd.read_pickle("./training_data/warehouse_data.pkl")
# print(df.head())


# Optional: first few pickup states
# for state, q_values in list(data["pickup"].items())[:25]:
#     print(state, "/n", q_values)
