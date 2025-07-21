import os
import re
import pickle
import random
import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

#############
# Utils
#############
def calc_stat_inter_sim(embs):
    sim = embs @ embs.T
    upper_triangle_indices = np.triu_indices(sim.shape[0], k=1)
    upper_triangle_values = sim[upper_triangle_indices]
    return np.mean(upper_triangle_values), np.std(upper_triangle_values)

def calc_inter_sims(embs):
    sim = embs @ embs.T
    upper_triangle_indices = np.triu_indices(sim.shape[0], k=1)
    upper_triangle_values = sim[upper_triangle_indices]
    return upper_triangle_values

################
# Load DEG list
################
# 1. DEG
degs_data = []
with open("../data/fig4/Disease_information_DEGs.gmt", "r") as f:
    for line in f:
        line = line.strip().split("\t")
        degs_data.append([line[0], line[1], ",".join(line[2:])])
degs_data = pd.DataFrame(degs_data, columns=["DatasetID", "dataset", "genes"])
degs_data["species"] = degs_data["dataset"].apply(lambda x: x.split("|")[-1])

# 2. Use only DEGs of human
degs_data = degs_data[degs_data["species"] == "Homo sapiens"].reset_index(drop=True)
# display(degs_data.head())
# display(degs_data.shape)

# 2. Disese-datasetID
DatasetID_disease = pd.read_csv("../data/fig4/Disease_information_Datasets.csv", encoding="ISO-8859-1")
# display(DatasetID_disease.head())
# display(DatasetID_disease.shape)

# 3. Load Disease to be used
disease2use = pd.read_csv("../data/fig4/DiSignAtlas_human_diseases_more_than_10_reports.csv")
# display(disease2use.head())
# display(disease2use.shape)

# 4. DatasetID to be used
def sanitize_directory_name(name):
    forbidden_chars = r'[\\/:*?"<>|\s\']'
    sanitized_name = re.sub(forbidden_chars, '_', name)
    return sanitized_name.strip('_')

datasets2use = []
dataset2disease = {}
dirnames = []
for disease in disease2use["disease"]:
    if disease == "Clear Cell Renal Cell Carcinoma":
          continue
    dirnames.append(sanitize_directory_name(disease))
    datasets = DatasetID_disease[DatasetID_disease["disease"] == disease]["dsaid"].values.tolist()
    # Only use human datasets
    datasets = sorted(list(set(datasets) & set(degs_data["DatasetID"])))
    datasets2use.append(datasets)

    for dataset in datasets:
        dataset2disease[dataset] = sanitize_directory_name(disease)

degs_data2use = degs_data[degs_data["DatasetID"].isin(list(itertools.chain.from_iterable(datasets2use)))]
degs_data2use["disease"] = degs_data2use["DatasetID"].map(dataset2disease)
print(degs_data2use["disease"].value_counts()[:20])
# display(degs_data2use.head())
# display(degs_data2use.shape)

value_counts = degs_data2use["disease"].value_counts()
fig, ax = plt.subplots(figsize=(20, 10))
value_counts.plot(kind='bar', ax=ax, cmap="viridis")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_xlabel("")
ax.set_ylabel("Count")
ax.set_title("DiSignAtlas Disease Datasets Counts")
plt.tight_layout()
fig.savefig("../data/fig4/fig4.supple.DiSignAtlas_disease_counts.png", dpi=300, bbox_inches='tight')
plt.close(fig)

#########################################
# Inter-gene similarity of DEGs compared with unweighted graph
#########################################
print("-"*25)
print("Inter-gene similarity of DEGs compared with unweighted graph")
print("-"*25)
# 0. Prepare the dictionary to convert gene symbols to gene ID
df = pd.read_csv("../data/fig2-3/all_human_gene_interactions_2024-12-18.csv")
symbol2id = {symbol: gene_id for symbol, gene_id in zip(df["from_gene"], df["from_entrez"])}
symbol2id.update({symbol: gene_id for symbol, gene_id in zip(df["to_gene"], df["to_entrez"])})

# 1. Load embeddings of unweighted graph
embs_unweighted = np.load("../data/output/GeneRelNet_unweighted_graph/gene_vec_256.npy")
embs_unweighted = normalize(embs_unweighted, axis=1)
gene_symbols_unweighted = np.load("../data/output/GeneRelNet_unweighted_graph/gene_ids.npy", allow_pickle=True)
gene_ids_unweighted = np.array([symbol2id[symbol] for symbol in gene_symbols_unweighted])
gene_id_to_idx_unweighted = {gene_id: idx for idx, gene_id in enumerate(gene_ids_unweighted)}

# Prepare the dictionary to store the results
results_unweighted_graph = {}
min_degs = 5
for disease, datasets in zip(dirnames, datasets2use):
    # 2. Load embeddings and gene_ids
    path = f"../data/output/GeneRelNet_{disease}_unweighted_graph"
    embs = np.load(os.path.join(path, "gene_vec_256.npy"))
    embs = normalize(embs, axis=1)
    gene_symbols = np.load(os.path.join(path, "gene_ids.npy"), allow_pickle=True)
    gene_ids = np.array([symbol2id[symbol] for symbol in gene_symbols])
    gene_id_to_idx = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}

    # 3. Calculate z-score of inter-gene similarity of DEGs
    inter_sim_z_scores = []
    for dataset in datasets:
        degs = degs_data2use[degs_data2use["DatasetID"] == dataset]["genes"].values[0].split(",")
        degs2use = [int(gene_id) for gene_id in degs if int(gene_id) in gene_id_to_idx]
        if not len(degs2use) >= min_degs:
            continue
        degs_idxes = np.array([gene_id_to_idx[gene_id] for gene_id in degs2use])
        degs_idxes_unweighted = np.array([gene_id_to_idx_unweighted[gene_id] for gene_id in degs2use])

        # 4. Calculate mean and std of inter-gene similarity of DEGs of unweighted graph
        mean, std = calc_stat_inter_sim(embs_unweighted[degs_idxes_unweighted])

        # 5. Calculate z-score of inter-gene similarity of DEGs
        sims = calc_inter_sims(embs[degs_idxes])
        z_scores = (sims - mean) / std

        inter_sim_z_scores.append(np.mean(z_scores))
    results_unweighted_graph[disease] = inter_sim_z_scores
    print(f"{disease} ({len(inter_sim_z_scores)}): average z-score = {np.mean(inter_sim_z_scores):.4f}")
print("-"*25)
total_values_unweighted = list(itertools.chain.from_iterable(results_unweighted_graph.values()))
print(f"Total average z-score = {np.mean(total_values_unweighted):.4f}")
pickle.dump(results_unweighted_graph, open("../data/output/inter_sim_z_scores_unweighted_graph.pkl", "wb"))


#########################################
# Inter-gene similarity of DEGs compared with other conditioned graphs
#########################################
print("-"*25)
print("Inter-gene similarity of DEGs compared with other conditioned graphs")
print("-"*25)
# 0. Prepare the dictionary to convert gene symbols to gene ID
df = pd.read_csv("/share/pubtator3/data/input/all_human_gene_interactions_2024-12-18.csv")
symbol2id = {symbol: gene_id for symbol, gene_id in zip(df["from_gene"], df["from_entrez"])}
symbol2id.update({symbol: gene_id for symbol, gene_id in zip(df["to_gene"], df["to_entrez"])})

# Prepare the dictionary to store the results
results_other_diseases = {}
min_degs = 5
np.random.seed(0) # for reproducibility
random.seed(0) # for reproducibility
for disease, datasets in zip(dirnames, datasets2use):
    # 1. Load embeddings and gene_ids
    path = f"../data/output/GeneRelNet_{disease}_unweighted_graph"
    embs = np.load(os.path.join(path, "gene_vec_256.npy"))
    embs = normalize(embs, axis=1)
    gene_symbols = np.load(os.path.join(path, "gene_ids.npy"), allow_pickle=True)
    gene_ids = np.array([symbol2id[symbol] for symbol in gene_symbols])
    gene_id_to_idx = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}

    other_diseases = [d for d in dirnames if d != disease]

    # 2. Calculate z-score of inter-gene similarity of DEGs
    inter_sim_z_scores = []
    for dataset in datasets:
        degs = degs_data2use[degs_data2use["DatasetID"] == dataset]["genes"].values[0].split(",")

        # 3. other diseases
        # Randomly select other diseases (not efficiently)
        other_diseases2use = random.choice(other_diseases)
        path = f"../data/output/GeneRelNet_{other_diseases2use}_unweighted_graph"
        embs_other_diseases = np.load(os.path.join(path, "gene_vec_256.npy"))
        embs_other_diseases = normalize(embs_other_diseases, axis=1)
        gene_symbols_other_diseases = np.load(os.path.join(path, "gene_ids.npy"), allow_pickle=True)
        gene_ids_other_diseases = np.array([symbol2id[symbol] for symbol in gene_symbols_other_diseases])
        gene_id_to_idx_other_diseases = {gene_id: idx for idx, gene_id in enumerate(gene_ids_other_diseases)}
        
        degs2use = [int(gene_id) for gene_id in degs if (int(gene_id) in gene_id_to_idx) and (int(gene_id) in gene_id_to_idx_other_diseases)]

        if not len(degs2use) >= min_degs:
            continue
        degs_idxes = np.array([gene_id_to_idx[gene_id] for gene_id in degs2use])
        degs_idxes_other_diseases = np.array([gene_id_to_idx_other_diseases[gene_id] for gene_id in degs2use])

        # 4. Calculate mean and std of inter-gene similarity of DEGs of other disease
        mean, std = calc_stat_inter_sim(embs_other_diseases[degs_idxes_other_diseases])

        # 5. Calculate z-score of inter-gene similarity of DEGs
        sims = calc_inter_sims(embs[degs_idxes])
        z_scores = (sims - mean) / std

        inter_sim_z_scores.append(np.mean(z_scores))
    results_other_diseases[disease] = inter_sim_z_scores
    print(f"{disease} ({len(inter_sim_z_scores)}): average z-score = {np.mean(inter_sim_z_scores):.4f}")
print("-"*25)
total_values_other_diseases = list(itertools.chain.from_iterable(results_other_diseases.values()))
print(f"Total average z-score = {np.mean(total_values_other_diseases):.4f}")
pickle.dump(results_other_diseases, open("../data/output/inter_sim_z_scores_other_disease_graph.pkl", "wb"))