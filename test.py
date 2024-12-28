import pandas as pd
import pickle

# Load transactions
transactions = pd.read_pickle("data/transactions.pkl")

# Find active customers (with at least 5 transactions)
customer_counts = transactions['customer_id'].value_counts()
active_customers = customer_counts[customer_counts >= 5].index

# Sample 1000 users (or less if there aren't that many active users)
n_demo_users = 1000
demo_users = active_customers[:n_demo_users].tolist()

# Save to pickle file
with open('data/demo_user_ids.pkl', 'wb') as f:
    pickle.dump(demo_users, f)

print(f"Created demo_user_ids.pkl with {len(demo_users)} users")