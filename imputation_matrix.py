#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.sparse import lil_matrix, coo_matrix


# In[ ]:


# -------------------------
# Configuration
# -------------------------

# Set your data folder here
data_folder = "/path/to/your/data"  # <- Replace with actual path
k = 5  # Number of top neighbors to consider


# In[ ]:


# File paths
train_file = os.path.join(data_folder, "train.txt")
output_file = os.path.join(data_folder, f"rate_add_{k}nn.txt")


# In[ ]:


# -------------------------
# Data Loading and Mapping
# -------------------------

class DataLoader:
    """
    Custom data loader that maps user/item IDs to 0-based indices
    and loads the ratings data from a file.
    """
    def __init__(self):
        self.user_mapper = {}
        self.item_mapper = {}
        self.user_counter = 0
        self.item_counter = 0

    def load_ratings_train(self, file_path):
        # Load ratings file with userId, itemId, rating
        df = pd.read_csv(file_path, sep=" ", header=None)
        df.columns = ['userId', 'itemId', 'rating']
        # Map user/item IDs to indices
        df, self.user_mapper, self.item_mapper, self.user_counter, self.item_counter = self._map_entities(df)
        return df, self.user_counter, self.item_counter

    def _map_entities(self, df):
        for idx, row in df.iterrows():
            user_id, item_id = row['userId'], row['itemId']
            if user_id not in self.user_mapper:
                self.user_mapper[user_id] = self.user_counter
                self.user_counter += 1
            if item_id not in self.item_mapper:
                self.item_mapper[item_id] = self.item_counter
                self.item_counter += 1

            df.at[idx, 'userId'] = self.user_mapper[user_id]
            df.at[idx, 'itemId'] = self.item_mapper[item_id]
            df.at[idx, 'rating'] = abs(row['rating'])  # Ensure ratings are positive
        return df, self.user_mapper, self.item_mapper, self.user_counter, self.item_counter


# In[ ]:


# Initialize and load data
data_loader = DataLoader()
df, userNum, itemNum = data_loader.load_ratings_train(file_path=train_file)

# -------------------------
# Build User Mean Dictionary
# -------------------------

r_mean = defaultdict(float)
r_count = defaultdict(int)

for _, row in df.iterrows():
    user = row['userId']
    r_mean[user] += row['rating']
    r_count[user] += 1

# Compute average rating per user
for user in r_mean:
    r_mean[user] /= r_count[user]

# -------------------------
# Build User-Item List
# -------------------------

user_itemlist = defaultdict(list)
item_set = set()

for _, row in df.iterrows():
    user, item = int(row['userId']), int(row['itemId'])
    user_itemlist[user].append(item)
    item_set.add(item)

item_list = list(item_set)

# -------------------------
# Create User-Item Rating Matrix (Sparse)
# -------------------------

rate = lil_matrix((userNum, itemNum))

for _, row in df.iterrows():
    user, item, rating = int(row['userId']), int(row['itemId']), row['rating']
    rate[user, item] = rating

# -------------------------
# Compute User-User Pearson Similarity
# -------------------------

Sim = lil_matrix((userNum, userNum))

for u1 in user_itemlist:
    for u2 in user_itemlist:
        if u1 < u2:
            common_items = set(user_itemlist[u1]) & set(user_itemlist[u2])
            if not common_items:
                continue

            numerator = sum((rate[u1, i] - r_mean[u1]) * (rate[u2, i] - r_mean[u2]) for i in common_items)
            denom_u1 = sum((rate[u1, i] - r_mean[u1]) ** 2 for i in user_itemlist[u1])
            denom_u2 = sum((rate[u2, i] - r_mean[u2]) ** 2 for i in user_itemlist[u2])

            if numerator != 0 and denom_u1 != 0 and denom_u2 != 0:
                sim_score = numerator / (np.sqrt(denom_u1) * np.sqrt(denom_u2))
                Sim[u1, u2] = Sim[u2, u1] = sim_score

# Convert sparse similarity matrix to dense
Sim_dense = coo_matrix(Sim).toarray()


# In[ ]:


# -------------------------
# Sort Similar Users for Each User
# -------------------------

def sort_sparse_matrix(sim_matrix, top_k, only_indices=True):
    """
    Sort rows of a similarity matrix by value.
    Returns top_k neighbors for each user.
    """
    sorted_dict = {}
    for i in range(sim_matrix.shape[0]):
        row = list(zip(sim_matrix[i].nonzero()[0], sim_matrix[i][sim_matrix[i] != 0]))
        sorted_row = sorted(row, key=lambda x: x[1], reverse=True)
        sorted_dict[i] = [idx for idx, _ in sorted_row[:top_k]] if only_indices else sorted_row[:top_k]
    return sorted_dict

# Get top-k similar neighbors for each user
top_k_neighbors = sort_sparse_matrix(Sim_dense, top_k=k)


# In[ ]:


# -------------------------
# Impute Missing Ratings Using Neighbors
# -------------------------

rate_add = lil_matrix((userNum, itemNum))

for user in range(userNum):
    for item in range(itemNum):
        if rate[user, item] != 0:
            continue  # Skip if already rated

        weighted_sum = 0.0
        sim_sum = 0.0
        has_neighbor = False

        for neighbor in top_k_neighbors.get(user, []):
            if Sim_dense[user, neighbor] > 0 and rate[neighbor, item] != 0:
                weighted_sum += Sim_dense[user, neighbor] * (rate[neighbor, item] - r_mean[neighbor])
                sim_sum += Sim_dense[user, neighbor]
                has_neighbor = True

        if has_neighbor and sim_sum != 0:
            imputed_rating = r_mean[user] + weighted_sum / sim_sum
            if imputed_rating > 0:
                rate_add[user, item] = imputed_rating

# -------------------------
# Save Imputed Matrix to File
# -------------------------

with open(output_file, 'w') as f:
    for user in range(userNum):
        for item in range(itemNum):
            if rate_add[user, item] != 0:
                f.write(f"{user} {item} {rate_add[user, item]:.6f}\n")

