import sys
sys.path.append("..")

import random
import numpy as np
import torch

import pandas as pd
from txgnn import TxData, TxGNN, TxEval
from txgnn.utils import evaluate_graph_construct
# Download/load knowledge graph dataset
TxData = TxData(data_folder_path = './data')
TxData.prepare_split(split = 'complex_disease', seed = 42)

# save
# TxData.df_train.to_csv('./complex_disease_train.csv', index = False)
# TxData.df_valid.to_csv('./complex_disease_valid.csv', index = False)
# TxData.df_test.to_csv('./complex_disease_test.csv', index = False)


TxGNN = TxGNN(data = TxData, 
              weight_bias_track = False,
              proj_name = 'TxGNN', # wandb project name
              exp_name = 'TxGNN', # wandb experiment name
              device = 'cuda:0' # define your cuda device
              )

# Initialize a new model
TxGNN.model_initialize(n_hid = 100, # number of hidden dimensions
                      n_inp = 100, # number of input dimensions
                      n_out = 100, # number of output dimensions
                      proto = True, # whether to use metric learning module
                      proto_num = 3, # number of similar diseases to retrieve for augmentation
                      attention = False, # use attention layer (if use graph XAI, we turn this to false)
                      sim_measure = 'all_nodes_profile', # disease signature, choose from ['all_nodes_profile', 'protein_profile', 'protein_random_walk']
                      agg_measure = 'rarity', # how to aggregate sim disease emb with target disease emb, choose from ['rarity', 'avg']
                      num_walks = 200, # for protein_random_walk sim_measure, define number of sampled walks
                      path_length = 2 # for protein_random_walk sim_measure, define path length
                      )

################
# Generate the negative graph for training
################
def generage_edge_list_df(G):
    edge_list = []

    for canonical_etype in G.canonical_etypes:
        src_type, relation, dst_type = canonical_etype
        
        # get the source and destination node ids
        src, dst = G.edges(etype=canonical_etype)
        src = src.cpu().numpy()
        dst = dst.cpu().numpy()
        
        # create a list of dictionaries
        for s, d in zip(src, dst):
            edge_list.append({
                'src_type': src_type,   # ソースノードのタイプ
                'relation': relation,   # エッジのリレーション名
                'dst_type': dst_type,   # デスティネーションノードのタイプ
                'x_idx': s,             # ソースノードのインデックス
                'y_idx': d              # デスティネーションノードのインデックス
            })
        
    edge_list_df = pd.DataFrame(edge_list)

    return edge_list_df


seeds = [0, 1, 6, 10]
ks = [1, 3, 5, 10, 20]



for seed in seeds:
    for k in ks:
        # Set the random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        # generate negative graph according to the original TxGNN paper
        _, g_neg = evaluate_graph_construct(TxGNN.df_train, 
                                            TxGNN.G,
                                            "fix_dst",
                                            k,
                                            TxGNN.device)
        # get the negative graph
        g_neg = generage_edge_list_df(g_neg)
        
        g_neg.to_csv(f'./complex_disease_train_neg_k{k}_seed{seed}.csv', index = False)

