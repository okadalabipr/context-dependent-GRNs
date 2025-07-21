"""
Diseases of which reports exist more than 10 in DiSignAtlas
"""
import os
import re
import gc
import math
import copy
import time
import datetime
import argparse
from functools import partial
from tqdm import tqdm
import random
from joblib import Parallel, delayed
from argparse import Namespace
from itertools import product
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

####################
### Functions
####################
def get_elapsed(start, end):
    elapsed = end - start
    elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    return elapsed

## DPWalker
# Ref: Representation Leanring for Scale-Free Networks (AAAI-18)
def get_penalized_transition_probs(graph, W):
    DP_matrix = W.toarray() * nx.adjacency_matrix(graph, weight="weight").toarray()
    DP_matrix /= DP_matrix.sum(axis=1, keepdims=True)

    node_to_index = {node: idx for idx, node in enumerate(list(graph.nodes))}
    DP_dict = {node: probs[[node_to_index[nei] for nei in graph.neighbors(node)]] 
            for node, probs in zip(graph.nodes, DP_matrix)}
    return DP_dict

def get_penalized_transition_probs_multisteps(graph, W, steps=3):
    DP_matrix = W.toarray() * nx.adjacency_matrix(graph, weight="weight").toarray()
    DP_matrix /= DP_matrix.sum(axis=1, keepdims=True)

    multistep_probs = np.zeros_like(DP_matrix)
    for i in range(1, steps+1):
        multistep_probs += np.linalg.matrix_power(DP_matrix, i)
    
    return multistep_probs

# Compute degree penalty matrix W
def get_degree_penalty_matrix(graph, beta):
    # D = sp.diags([degree for _, degree in graph.degree(weight="weight")]) # sparse
    D = sp.diags([degree for _, degree in graph.degree()]) # sparse
    C = get_common_neighbor_matrix(graph) + nx.adjacency_matrix(graph)  # C' = C + A
    D_inv_beta = sp.diags(np.power(np.array(D.diagonal()), -beta))  # D^-beta
    W = D_inv_beta.T @ C @ D_inv_beta  # W = (D^-beta)^T C' (D^-beta)
    return W

def get_common_neighbor_matrix(graph):
    A = nx.adjacency_matrix(graph) # sparse
    C = A.T @ A - sp.diags((A.T @ A).diagonal())  # C = A^T A - diag(A^T A)
    return C

# Random walk with degree penalty
def random_walk(graph, start_node, walk_length, DP_dict):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        neighbors = list(graph.neighbors(cur))
        if len(neighbors) > 0:
            # weights = [W[cur_idx, nodes_list.index(neighbor)] for neighbor in neighbors]
            # probabilities = np.array(weights) / sum(weights)
            probabilities = DP_dict[cur]
            next_node = random.choices(neighbors, probabilities)[0]
            walk.append(next_node)
        else:
            break
    return walk

# Generate random walks
def generate_walks(graph, num_walks, walk_length, DP_dict, n_jobs=1):
    nodes = list(graph.nodes())
    
    shuffled_nodes = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        shuffled_nodes.append(nodes.copy())

    walks = Parallel(n_jobs=n_jobs)(delayed(lambda nodes: [random_walk(graph, node, walk_length, DP_dict) for node in nodes])(nodes) for nodes in tqdm(shuffled_nodes, desc=f"[Generating walks]"))
    walks = [walk for sublist in walks for walk in sublist]
    return walks


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# CosineAnnealingWarmupが最後min_lrの収束するように書き換え
def _get_cosine_schedule_with_warmup_min_lr(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr: float, initial_lr: float
):
    if current_step < num_warmup_steps:
        # Warmup phase: linearly increase from 0 to initial_lr
        return float(current_step) / float(max(1, num_warmup_steps))
    
    # Cosine phase
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_lr = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
    
    # Scale cosine_lr to range [min_lr / initial_lr, 1.0]
    min_lr_ratio = min_lr / initial_lr
    return max(min_lr_ratio, cosine_lr * (1.0 - min_lr_ratio) + min_lr_ratio)

def get_cosine_schedule_with_warmup_min_lr(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, min_lr: float = 1e-7, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to `min_lr`, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        min_lr (`float`):
            The minimum learning rate to which the cosine schedule will decay.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the default is to just decrease from the max value to `min_lr`
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # Get the initial learning rate from the optimizer
    initial_lr = optimizer.defaults['lr']

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_min_lr,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr=min_lr,
        initial_lr=initial_lr,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

if __name__ == "__main__":
    ####################
    ### Configuration
    ####################
    config = Namespace(
        exp = "GeneRelNet",
        seed=42,
        # DeepWalk parameters
        embedding_dim=256,
        num_negative_samples=5,
        # Degree Penalty
        beta = 1.0,
        # Training parameters
        num_epochs=2000,
        steps = 3,
        batch_size = 128,
        learning_rate = 0.001,
        min_lr = 1e-6,
        num_warmup_epochs = 0,
        n_patience = 100,
        pos_weight = 1.0
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", type=str, required=True)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--num_negative_samples", type=int, default=5)
    args = parser.parse_args()
    config.beta = args.beta
    config.steps = args.steps
    config.pos_weight = args.pos_weight
    config.num_negative_samples = args.num_negative_samples

    mesh_id = args.disease
    
    ####################
    ### Load disease name
    ####################
    disease_name_df = pd.read_csv("../data/fig4/DiSignAtlas_human_diseases_more_than_10_reports.csv")
    disease_name = disease_name_df.set_index("mesh_id").at[mesh_id, "disease"]
    if isinstance(disease_name, pd.Series):
        disease_name = disease_name.values[0]

    def sanitize_directory_name(name):
        forbidden_chars = r'[\\/:*?"<>|\s\']'
        sanitized_name = re.sub(forbidden_chars, '_', name)
        return sanitized_name.strip('_')
    disease_name = sanitize_directory_name(disease_name)

    os.makedirs(f"../data/fig4/{config.exp}_{disease_name}", exist_ok=True)

    ####################
    ### Load network
    ####################
    # Load gene-gene interactions from BioREX database
    relations_df = pd.read_csv("../data/fig2-3/all_human_gene_interactions_2024-12-18.csv")

    # Calculate context-dependent weights
    relations_df["pair"] = relations_df.apply(lambda x: "--".join(sorted([str(x["from_gene"]), str(x["to_gene"])])), axis=1)
    relations_df = relations_df.drop_duplicates(subset=["pmid", "pair"]).reset_index(drop=True)

    # Load corretion between PMIDs and BERT embeddings
    # You have to calculate correlation in advance and ensure that "condition" is included in the column names.
    threshold = 0.1

    # find the directory where the correlation files are stored
    disease_file_df = pd.read_csv("../data/fig2-3/MeSH/mesh_disease_leaves_w_annotation.csv")
    chunk_number = disease_file_df[disease_file_df["mesh_id"] == mesh_id]["chunk_number"].values[0]

    pmid_corrs = pd.read_parquet(f"../data/fig2-3/All_MeSH_diseases_pmid_bert_corrs_chunk{chunk_number}.parquet", columns=["pmid", mesh_id])

    relations_df["corr"] = relations_df["pmid"].map(pmid_corrs.set_index("pmid")[mesh_id])
    relations_df = relations_df.dropna(subset=["corr"]).reset_index(drop=True)
    # Thresholding
    relations_df = relations_df[relations_df["corr"] > threshold].reset_index(drop=True)
    print("Number of interactions after thresholding:", len(relations_df))

    relations_df = relations_df[["pair", "corr"]].groupby("pair").agg({"corr": "sum"}).reset_index()
    # relations_df["corr"] = np.log10(relations_df["corr"].values + 10)
    relations_df["corr"] = relations_df["corr"] / relations_df["corr"].sum() * 1e6
    # Log transformation to make the distribution more normal
    relations_df["corr"] = np.log1p(relations_df["corr"].values)
    # rename
    relations_df.columns = ["pair", "weight"]

    # Create a graph
    edges = [(pair.split("--")[0], pair.split("--")[1], count) for pair, count in zip(relations_df["pair"], relations_df["weight"])]
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    n_nodes = len(G.nodes)
    n_edges = len(G.edges)
    density = nx.density(G)
    print("--- Step 1 ---")
    print(f"Number of nodes: {n_nodes}")
    print(f"Number of edges: {n_edges}")
    print(f"Density: {density}")

    # Extract connected components
    connected_components = list(nx.connected_components(G))
    size_connected_components = [len(c) for c in connected_components]
    # Extract the largest connected component
    main_subgraph = G.subgraph(connected_components[np.argmax(size_connected_components)]).copy()
    gene_ids = np.array(list(main_subgraph.nodes))
    n_nodes = len(main_subgraph.nodes)
    n_edges = len(main_subgraph.edges)
    density = nx.density(main_subgraph)
    print("--- Step 2 (After extracting largest connected component) ---")
    print(f"Number of nodes: {n_nodes}")
    print(f"Number of edges: {n_edges}")
    print(f"Density: {density}")

    ####################
    # Degree penalied random walk
    ####################
    W = get_degree_penalty_matrix(main_subgraph, config.beta)
    DP_probs = get_penalized_transition_probs_multisteps(main_subgraph, W, steps=config.steps)

    ###########################
    # Dataset
    ###########################
    class GeneDataset(Dataset):
        def __init__(self, gene_ids, DP_probs, neg_samples):
            self.gene_ids = gene_ids
            self.DP_probs = DP_probs
            self.neg_samples = neg_samples
            self.pos_samples = 1
        def __len__(self):
            return len(self.gene_ids)
        def __getitem__(self, idx):
            p = self.DP_probs[idx].copy()
            if p[idx] != 1:
                p[idx] = 0
            p = p / np.sum(p)
            posid = np.random.choice(len(self.gene_ids), self.pos_samples, p=p)

            diffusion = self.DP_probs[idx].copy()
            diffusion = np.clip(diffusion, 1e-3, None)
            negp = (1. / diffusion)
            negp[idx] = 0
            negp = negp / np.sum(negp)
            negid = np.random.choice(len(self.gene_ids), self.neg_samples, p=negp)

            left_idxes = [idx] * (len(posid) + len(negid))
            right_idxes = np.concatenate([posid, negid])
            labels = [1.] * len(posid) + [0.] * len(negid)
            return torch.tensor(left_idxes), torch.tensor(right_idxes), torch.tensor(labels)
    
    def collate_fn(batch):
        left_idxes, right_idxes, labels = zip(*batch)
        return torch.cat(left_idxes), torch.cat(right_idxes), torch.cat(labels)

    
    ###########################
    # Model
    ###########################
    class GeneEmbedding(nn.Module):
        def __init__(self, vocab_size, embedding_dim):
            super(GeneEmbedding, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            nn.init.xavier_uniform_(self.embedding.weight)
        def forward_embedding(self, x):
            return self.embedding(x)
        def forward(self, left_idx, right_idx):
            left_emb = self.embedding(left_idx)
            right_emb = self.embedding(right_idx)
            return (left_emb * right_emb).sum(dim=1)

    ###########################
    # Train embedding model
    ###########################
    ds = GeneDataset(gene_ids, DP_probs, config.num_negative_samples)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn,
                    num_workers=10, pin_memory=True, drop_last=False)

    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeneEmbedding(len(gene_ids), config.embedding_dim).to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = get_cosine_schedule_with_warmup_min_lr(optimizer, config.num_warmup_epochs, config.num_epochs, config.min_lr)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([config.pos_weight]).to(device))
    
    start = datetime.datetime.now()
    print("Start training...", start.strftime("%Y-%m-%d %H:%M:%S"))

    best_loss = np.inf
    patience = 0
    for e in range(config.num_epochs):
        losses_epoch = []
        labels_epoch = []
        for left_idx, right_idx, labels in dl:
            left_idx, right_idx, labels = left_idx.to(device), right_idx.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(left_idx, right_idx)
            loss = criterion(output, labels)

            losses_epoch.extend(loss.detach().cpu().tolist())
            labels_epoch.extend(labels.detach().cpu().tolist())

            loss.mean().backward()
            optimizer.step()
        
        # To compare the loss across other settings, escecially pos_weight, we calculate the loss after removing the effect of pos_weight
        labels_epoch = np.array(labels_epoch)
        losses_epoch = np.array(losses_epoch)
        losses_epoch = np.where(labels_epoch == 1, losses_epoch / config.pos_weight, losses_epoch)

        loss_mean = np.mean(losses_epoch)
        loss_pos_mean = np.mean(losses_epoch[labels_epoch == 1])
        loss_neg_mean = np.mean(losses_epoch[labels_epoch == 0])
        elapsed = datetime.datetime.now() - start
        elapsed = str(elapsed).split('.')[0]
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {e+1}/{config.num_epochs} loss={loss_mean:.4f} pos_loss={loss_pos_mean:.4f} neg_loss={loss_neg_mean:.4f} elapsed={elapsed} | lr={current_lr:.6f}")

        if loss_mean < best_loss:
            best_loss = loss_mean
            best_model = copy.deepcopy(model)
            patience = 0
        else:
            patience += 1
            if patience >= config.n_patience:
                print(f"Early stopping: patience limit reached.")
                break
        
        # update learning rate per epoch
        scheduler.step()
        
    print("Training finished.")

    # Save the embedding
    encoded_genes = model.embedding.weight.detach().cpu().numpy()
    np.save(f"../data/fig4/{config.exp}_{disease_name}/gene_vec_{config.embedding_dim}.npy", encoded_genes)
    np.save(f"../data/fig4/{config.exp}_{disease_name}/gene_ids.npy", gene_ids)