

## 🧠 Project Overview

This repository contains the official implementation of the core components used in our paper:

**_Enhancing Recommender Systems through Imputation and Social-Aware Graph Convolutional Neural Network_**

Recommender systems rely heavily on user–item interactions, which are typically sparse. Our work introduces a hybrid graph-based solution that combines **user trust relationships** and an **imputation-based graph** to address the sparsity problem. This implementation includes two major components.

---

## 📁 Code Structure

### 📌 Imputation Matrix Construction — `imputation_matrix.py`

This module builds an **imputed rating matrix** to enrich the sparse user–item interaction graph by estimating missing ratings using **user-based collaborative filtering with Pearson similarity**.

#### Key Steps
- Maps raw user/item IDs to internal indices
- Builds a sparse user–item rating matrix
- Calculates pairwise user similarity
- Uses top-*k* most similar users to impute unrated entries
- Outputs the imputed matrix to a text file

#### Purpose
Constructs the **imputation graph**, one of the auxiliary graphs in our GCN model, simulating missing rating behavior to reduce sparsity and improve recommendation quality.

---

### 📌 Graph Convolutional Recommender — `gcf_model.py`

This module implements our **Graph-based Collaborative Filtering (GCF)** model, enhanced with:
- A **rating graph**
- A **trust graph** (social connections/trust statements)
- The **imputation graph** (constructed above)

#### Key Features
- Learns user and item embeddings via stacked GNN layers
- Incorporates all three graphs with an **attention mechanism** to weigh their importance
- Includes a complete training loop with:
  - RMSE, MAE, MSE, and R² metrics
  - Logging to `results.txt`
  - Optional GPU support

#### Purpose
Trains a graph neural network that attends to multiple interaction signals (ratings, trust, and imputed similarities) for **more accurate, robust, and personalized recommendations**.

---

## 🧪 Datasets

The code is designed to work with standard recommendation datasets such as **Epinions** or **FilmTrust**.

Each file should follow this format:

```txt
userId itemId rating
```


## 🚀 Reproducibility
To reproduce results:

Run imputation_matrix.py to generate the imputed rating file (e.g., rate_add_5nn.txt).

Train the GCF model with gcf_model.py using the original and imputed data.

## 📜 Citation & Reference

If you find this work helpful, please consider citing our paper:

**Enhancing Recommender Systems through Imputation and Social-Aware Graph Convolutional Neural Network**  
*Neural Networks*, **Volume 184**, April 2025, Article 107071  
[🔗 Read on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608024010001)

### BibTeX

```bibtex
@article{your2025imputationgcn,
  title     = {Enhancing Recommender Systems through Imputation and Social-Aware Graph Convolutional Neural Network},
  author    = {Azadeh Faroughi, Parham Moradi, Mahdi Jalili},
  journal   = {Neural Networks},
  volume    = {184},
  pages     = {107071},
  year      = {2025},
  issn      = {0893-6080},
  doi       = {10.1016/j.neunet.2024.12.010},
  url       = {https://www.sciencedirect.com/science/article/pii/S0893608024010001}
}
```

## 🛠️ Requirements
Python 3.8+

PyTorch

NumPy

SciPy

scikit-learn

pandas
