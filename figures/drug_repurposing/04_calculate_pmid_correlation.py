# MAKE SURE!! import faiss before importing pandas
import sys
sys.path.append("/share/vector-databases/code")
# from vector_database_update import UniprotVD
from vector_database_update import PubmedVD, UniprotVD

import os
import gc
import ast
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import torch
import transformers
from sentence_transformers import SentenceTransformer
print("torch version:", torch.__version__)
print("transformers version:", transformers.__version__)

base_pubtator3 = "/xxx/pubtator3"

################################################
### Load pmid list
################################################
pmids_biorex = pd.read_csv(os.path.join(base_pubtator3, "data", "output", "Pubtator3_BioREX_pmid_mesh.csv"))
pmids_biorex["from_mesh"] = pmids_biorex["from_mesh"].apply(ast.literal_eval)
print(f"Loaded {pmids_biorex.shape[0]} pmids from BioREX")


################################################
### Load Sentence transformers
################################################
model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
vd = PubmedVD(model_name=model_name,
              window_size=2, slide_size=1, nprobe=1)

model = SentenceTransformer(model_name)

################################################
# Load disease discriptions
################################################
disease_df = pd.read_csv("03_disease_descriptions.csv")
print(f"Loaded {disease_df.shape[0]} disease descriptions")

################################################
### Calculate correlation
################################################
def calc_cosine_similarity(X, Y, mean=None, std=None):
    """
    X, Y: n_samples x n_features
    mean, std: n_features
    """
    if mean is not None and std is not None:
        X = (X- mean) / std
        Y = (Y - mean) / std
    X = normalize(X.copy())
    Y = normalize(Y.copy())
    # sim = (X * Y).sum(axis=1)
    sim = X @ Y.T
    return sim

means = np.load("../../data/fig2-3/pubmed24n_mean.npy")
stds = np.load("../../data/fig2-3/pubmed24n_std.npy")

disease_text = disease_df["description"].values
disease_embeddings = model.encode(disease_text)

# Split results to prevent memory errors
pmid_corrs_chunk1 = []
pmid_corrs_chunk2 = []
resulting_pmids =[]

bs = 2000
for i in tqdm(range(0, len(pmids_biorex), bs)):
    pmids = pmids_biorex["pmid"].values[i:i+bs]

    ids = []
    batch_sentences = []
    for pmid in pmids:
        try:
            window_idxs = vd.get_window_idxs(pmid, include_title=True)
            sentences = vd.get_windows(window_idxs)
            sentences = [s[2] for s in sentences]
            ids.extend([pmid] * len(sentences))
            batch_sentences.extend(sentences)
        except:
            pass
    embeddings = model.encode(batch_sentences)
    corrs = calc_cosine_similarity(embeddings, disease_embeddings, mean=means, std=stds)

    corrs_df = pd.DataFrame(corrs, index=ids, columns=disease_df["disease"].values).reset_index()
    corrs_max_df = corrs_df.groupby("index").max().reset_index(names="pmid")
    # pmid_corrs.append(corrs_max_df)
    resulting_pmids.append(corrs_max_df["pmid"].values)
    pmid_corrs_chunk1.append(corrs_max_df.iloc[:, :1051])
    pmid_corrs_chunk2.append(corrs_max_df.iloc[:, 1051:])

resulting_pmids = np.concatenate(resulting_pmids)

pmid_corrs_chunk1 = pd.concat(pmid_corrs_chunk1, axis=0)
pmid_corrs_chunk1.to_parquet("04_Pubtator3_BioREX_pmid_bert_corrs_primeKG_chunk1.parquet", index=False)
del pmid_corrs_chunk1
gc.collect()

pmid_corrs_chunk2 = pd.concat(pmid_corrs_chunk2, axis=0)
pmid_corrs_chunk2.insert(0, "pmid", resulting_pmids)
pmid_corrs_chunk2.to_parquet("04_Pubtator3_BioREX_pmid_bert_corrs_primeKG_chunk2.parquet", index=False)
del pmid_corrs_chunk2
gc.collect()