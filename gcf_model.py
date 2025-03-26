#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss
from torch.nn import Module
from torch.nn import init

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, vstack
from scipy import sparse
from scipy.spatial.distance import cdist, cosine
from sklearn.metrics import pairwise_distances
import torch
from torch import nn as nn

from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
from numpy import diag

from torch.utils.data import DataLoader

from torch.utils.data import random_split

from torch.optim import Adam
from torch.nn import MSELoss

from os import path
from datetime import datetime


# In[ ]:


import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self):
        # Maps original user and item IDs to internal indices
        self.user_mapper = {}
        self.item_mapper = {}
        self.user_counter = 0
        self.item_counter = 0

    def load_ratings_train(self, file_path):
        # Loads training ratings from a file
        df = pd.read_csv(file_path, sep=" ", header=None)
        df.columns = ['userId', 'itemId', 'rating']
        # Maps raw IDs to internal indices
        df, self.user_mapper, self.item_mapper, self.user_counter, self.item_counter = self._map_entities(df)
        
        return df, self.user_counter, self.item_counter

    def load_ratings_test(self, file_path):
        # Loads test ratings and maps IDs using training mappings
        df = pd.read_csv(file_path, sep=" ", header=None)
        df.columns = ['userId', 'itemId', 'rating']
        # Apply mappings
        df['userId'] = df['userId'].map(self.user_mapper)
        df['itemId'] = df['itemId'].map(self.item_mapper)
        # Filter out unknown users/items
        df = df[df['userId'] < self.user_counter]
        df = df[df['itemId'] < self.item_counter]
        
        df = df.reset_index(drop=True)
        
        return df

    def load_trust(self, file_path):
        # Loads trust data between users
        df = pd.read_csv(file_path, sep=" ", header=None)
        df.columns = ['userId1', 'userId2', 'trust']
        
        # Map user IDs using the mapper from training data
        df['userId1'] = df['userId1'].map(self.user_mapper)
        df['userId2'] = df['userId2'].map(self.user_mapper)
        
        # Filter out any unmapped/unknown users
        df = df[df['userId1'] < self.user_counter]
        df = df[df['userId2'] < self.user_counter]
        df = df.reset_index(drop=True)
        
        return df

    def load_rate_add(self, file_path):
        # Loads additional rating data (raw, no mapping)
        df = pd.read_csv(file_path, sep=" ", header=None)
        df.columns = ['userId', 'itemId', 'rating']        
        return df

    def _map_entities(self, df):
        # Maps original user/item IDs to internal indices
        for idx, row in df.iterrows():
            user_id = row['userId']
            item_id = row['itemId']
            # Assign unique internal index to each user ID
            if user_id not in self.user_mapper:
                self.user_mapper[user_id] = self.user_counter
                self.user_counter += 1
            # Assign unique internal index to each item ID
            if item_id not in self.item_mapper:
                self.item_mapper[item_id] = self.item_counter
                self.item_counter += 1
            # Replace raw IDs with internal indices
            df.at[idx, 'userId'] = self.user_mapper[user_id]
            df.at[idx, 'itemId'] = self.item_mapper[item_id]
            # Ensure rating is positive
            df.at[idx, 'rating'] = abs(row['rating'])
        
        return df, self.user_mapper, self.item_mapper, self.user_counter, self.item_counter


# In[ ]:


class GNNLayer(Module):
    def __init__(self, inF, outF):
        super(GNNLayer, self).__init__()
        # Basic linear transformation
        self.linear = nn.Linear(inF, outF)
        # Interaction transformation: element-wise multiplication
        self.interActTransform = nn.Linear(inF, outF)

        # Attention mechanisms for each graph type
        self.interAttention_main = nn.Linear(inF, outF)
        self.interAttention_add = nn.Linear(inF, outF)
        self.interAttention_trust = nn.Linear(inF, outF)

        # Learnable attention weights (one per branch)
        self.a_main = nn.Parameter(torch.empty(size=(outF, 1)))
        self.a_add = nn.Parameter(torch.empty(size=(outF, 1)))
        self.a_trust = nn.Parameter(torch.empty(size=(outF, 1)))

        # Xavier initialization for attention weights
        nn.init.xavier_uniform_(self.a_main.data)
        nn.init.xavier_uniform_(self.a_add.data)
        nn.init.xavier_uniform_(self.a_trust.data)

    def forward(self, laplacianMat, selfLoop, Trust_LaplacianMat, Addrate_LaplacianMat, features):
        device = features.device

        # Prepare graph Laplacians
        L1 = (laplacianMat + selfLoop).to(device)  # main graph + self-loop
        L2 = laplacianMat.to(device)
        L3 = (Trust_LaplacianMat + selfLoop).to(device)
        L4 = Trust_LaplacianMat.to(device)
        L5 = (Addrate_LaplacianMat + selfLoop).to(device)
        L6 = Addrate_LaplacianMat.to(device)

        # Element-wise feature interaction
        inter_feature = features * features

        # Main graph interaction
        main_1 = self.linear(torch.sparse.mm(L1, features))
        main_2 = self.interActTransform(torch.sparse.mm(L2, inter_feature))
        main_att = torch.mm(torch.tanh(self.interAttention_main(main_1 + main_2)), self.a_main)
        w_main = main_att.mean()

        # Trust graph interaction
        trust_1 = self.linear(torch.sparse.mm(L3, features))
        trust_2 = self.interActTransform(torch.sparse.mm(L4, inter_feature))
        trust_att = torch.mm(torch.tanh(self.interAttention_trust(trust_1 + trust_2)), self.a_trust)
        w_trust = trust_att.mean()

        # Additional ratings graph interaction
        add_1 = self.linear(torch.sparse.mm(L5, features))
        add_2 = self.interActTransform(torch.sparse.mm(L6, inter_feature))
        add_att = torch.mm(torch.tanh(self.interAttention_add(add_1 + add_2)), self.a_add)
        w_add = add_att.mean()

        # Attention fusion using softmax over graph branches
        W = torch.stack([w_main, w_add, w_trust]).unsqueeze(0)  # shape: (1, 3)
        Beta = nn.Softmax(dim=1)(W)

        # Weighted sum of all interaction features
        return (
            Beta[0, 0] * main_1 + Beta[0, 1] * add_1 + Beta[0, 2] * trust_1 +
            Beta[0, 0] * main_2 + Beta[0, 1] * add_2 + Beta[0, 2] * trust_2
        )


# In[ ]:


class GCF(Module):
    def __init__(self, userNum, itemNum, rt, trust, rt_add, embedSize=100, layers=[100, 80, 50], useCuda=False):
        super(GCF, self).__init__()
        self.useCuda = useCuda
        self.userNum = userNum
        self.itemNum = itemNum

        # Embedding layers for users and items
        self.uEmbd = nn.Embedding(userNum, embedSize)
        self.iEmbd = nn.Embedding(itemNum, embedSize)

        # GNN layers list
        self.GNNlayers = nn.ModuleList()

        # Build adjacency matrices for the three types of graphs
        self.LaplacianMat, self.Trust_LaplacianMat, self.Addrate_LaplacianMat = self.buildLaplacianMat(rt, trust, rt_add)

        # Sparse identity matrix for self-loop
        self.selfLoop = self.getSparseEye(userNum + itemNum)

        # Feedforward prediction layers
        self.transForm1 = nn.Linear(layers[-1]*2, 64)
        self.transForm2 = nn.Linear(64, 32)
        self.transForm3 = nn.Linear(32, 1)

        # Stack GNN layers as defined in config
        for From, To in zip(layers[:-1], layers[1:]):
            self.GNNlayers.append(GNNLayer(From, To))

    def getSparseEye(self, num):
        # Create a sparse identity matrix for self-loops
        i = torch.LongTensor([[k for k in range(num)], [k for k in range(num)]])
        val = torch.FloatTensor([1] * num)
        return torch.sparse.FloatTensor(i, val)

    def buildLaplacianMat(self, rt, trust, rt_add):
        # Helper to build bi-partite rating or trust matrix
        def build_sparse_matrix(data, userNum, itemNum):
            item_shifted = data['itemId'] + userNum
            upper = coo_matrix((data['rating'], (data['userId'], item_shifted)), shape=(userNum, userNum + itemNum))
            lower = coo_matrix((data['rating'], (data['itemId'], data['userId'])),
                               shape=(itemNum, userNum))
            lower = lower.transpose()
            lower.resize((itemNum, userNum + itemNum))
            return sparse.vstack([upper, lower])

        # Normalize adjacency matrix to get Laplacian: L = D^(-1/2) * A * D^(-1/2)
        def normalize_adj(A):
            sumArr = (A > 0).sum(axis=1).A1
            sumArr[sumArr == 0] = 1
            D = sparse.diags(np.power(sumArr, -0.5))
            L = D @ A @ D
            L = sparse.coo_matrix(L)
            i = torch.LongTensor([L.row, L.col])
            data = torch.FloatTensor(L.data)
            return torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

        # Rating interaction
        A = build_sparse_matrix(rt, self.userNum, self.itemNum)

        # Additional rating interaction
        A_add = build_sparse_matrix(rt_add, self.userNum, self.itemNum)

        # Trust matrix (user-user only)
        trust_mat = sparse.dok_matrix((self.userNum, self.userNum + self.itemNum), dtype=np.float32)
        for i in range(len(trust)):
            u1, u2, val = trust.iloc[i, 0], trust.iloc[i, 1], trust.iloc[i, 2]
            if u1 < self.userNum and u2 < self.userNum:
                trust_mat[u1, u2] = val
        trust_full = sparse.vstack([trust_mat, sparse.dok_matrix((self.itemNum, self.userNum + self.itemNum))])

        # Return normalized Laplacians
        return normalize_adj(A), normalize_adj(trust_full), normalize_adj(A_add)

    def getFeatureMat(self):
        # Generate concatenated user + item embeddings
        uidx = torch.arange(self.userNum)
        iidx = torch.arange(self.itemNum)
        if self.useCuda:
            uidx = uidx.cuda()
            iidx = iidx.cuda()
        userEmbd = self.uEmbd(uidx)
        itemEmbd = self.iEmbd(iidx)
        return torch.cat([userEmbd, itemEmbd], dim=0)

    def forward(self, userIdx, itemIdx):
        if self.useCuda:
            userIdx = userIdx.cuda()
            itemIdx = itemIdx.cuda()

        # Shift item indices to match embedding matrix layout
        itemIdx_shifted = itemIdx + self.userNum
        user_list = userIdx.cpu().tolist()
        item_list = itemIdx_shifted.cpu().tolist()

        # Initial embeddings
        features = self.getFeatureMat()
        Embeddings = features.clone()
        all_embeddings = [Embeddings]

        # Apply GNN layers
        for gnn in self.GNNlayers:
            features = gnn(self.LaplacianMat, self.selfLoop, self.Trust_LaplacianMat, self.Addrate_LaplacianMat, features)
            all_embeddings.append(features.clone())

        # Average GNN outputs from all layers
        all_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(all_embeddings, dim=1)

        # Get embeddings for target users/items
        userEmbd = final_embeddings[user_list]
        itemEmbd = final_embeddings[item_list]

        # MLP for prediction
        embd = torch.cat([userEmbd, itemEmbd], dim=1)
        embd = nn.ReLU()(self.transForm1(embd))
        embd = self.transForm2(embd)
        embd = self.transForm3(embd)
        prediction = embd.flatten()

        return prediction, userEmbd, itemEmbd, final_embeddings


# In[ ]:


from torch.utils.data import Dataset

class ML1K(Dataset):
    def __init__(self, rt):
        # Initialize dataset with userId, itemId, and rating from dataframe
        self.uId = list(rt['userId'])
        self.iId = list(rt['itemId'])
        self.rt = list(rt['rating'])

    def __len__(self):
        # Returns total number of samples
        return len(self.uId)

    def __getitem__(self, idx):
        # Returns a single sample at the given index
        return self.uId[idx], self.iId[idx], self.rt[idx]


# In[ ]:


import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam

def main():
    from your_module import DataLoader as CustomLoader, GCF, ML1K  # adjust if needed

    # --- Folder for data ---
    data_folder = "/path/to/your/dataset"  

    # --- Load data ---
    data_loader = CustomLoader()

    train_data, userNum, itemNum = data_loader.load_ratings_train(
        file_path=os.path.join(data_folder, "train.txt")
    )

    test_data = data_loader.load_ratings_test(
        file_path=os.path.join(data_folder, "test.txt")
    )

    trust_df = data_loader.load_trust(
        file_path=os.path.join(data_folder, "trust.txt")
    )

    rate_add_df = data_loader.load_rate_add(
        file_path=os.path.join(data_folder, "rate_add.txt")
    )

    # --- Hyperparameters ---
    reg_lambda = 0.001
    para = {
        'epoch': 300,
        'lr': 0.001,
        'batch_size': 2048,
    }

    # --- Prepare datasets and loaders ---
    train_dataset = ML1K(train_data)
    test_dataset = ML1K(test_data)

    train_loader = DataLoader(train_dataset, batch_size=para['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # --- Model ---
    model = GCF(userNum, itemNum, train_data, trust_df, rate_add_df, embedSize=80, layers=[80, 80, 80], useCuda=True)
    model = model.cuda()

    optimizer = Adam(model.parameters(), lr=para['lr'])
    criterion = MSELoss()

    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)

    # --- Logging setup ---
    log_path = os.path.join("results", "results.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "a") as f:
        f.write("\n" + "="*60 + "\n")
        f.write(f"Run started at {datetime.now()}\n")

    # --- Training ---
    best_rmse = float('inf')
    best_mae = float('inf')
    rmse_epoch = 0
    mae_epoch = 0
    start_time = datetime.now()

    for epoch in range(para['epoch']):
        model.train()
        for user, item, rating in train_loader:
            user, item, rating = user.cuda(), item.cuda(), rating.float().cuda()
            optimizer.zero_grad()
            preds, _, _, _ = model(user, item)

            loss = criterion(preds, rating)
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss += reg_lambda * l2_norm

            loss.backward()
            optimizer.step()

        # --- Evaluation ---
        model.eval()
        with torch.no_grad():
            for user, item, rating in test_loader:
                user, item, rating = user.cuda(), item.cuda(), rating.float().cuda()
                preds, _, _, _ = model(user, item)

            test_rmse = torch.sqrt(criterion(preds, rating)).item()
            test_mae = torch.mean(torch.abs(rating - preds)).item()

            if test_rmse < best_rmse:
                best_rmse = test_rmse
                rmse_epoch = epoch
            if test_mae < best_mae:
                best_mae = test_mae
                mae_epoch = epoch

        log_line = (
            f"Epoch {epoch+1:03d} | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | "
            f"Best RMSE: {best_rmse:.4f} (Epoch {rmse_epoch}) | Best MAE: {best_mae:.4f} (Epoch {mae_epoch})"
        )
        print(log_line)
        with open(log_path, "a") as f:
            f.write(log_line + "\n")

    # --- Wrap-up ---
    end_time = datetime.now()
    duration = end_time - start_time

    summary = (
        f"\nTraining Complete\n"
        f"Time: {duration}\n"
        f"Best RMSE: {best_rmse:.4f} @ Epoch {rmse_epoch}\n"
        f"Best MAE : {best_mae:.4f} @ Epoch {mae_epoch}\n"
    )
    print(summary)
    with open(log_path, "a") as f:
        f.write(summary)


if __name__ == "__main__":
    main()

