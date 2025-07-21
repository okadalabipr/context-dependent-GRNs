# Figure 1: Similarity computation between PubMed abstracts and user queries.
# 
# This script computes the similarity between PubMed abstracts and input queries.
# The PubMed abstracts are stored internally as a FAISS vector database.
# Due to the large size of the full PubMed dataset, it is not included in this repository.
# Instead, we provide the output similarity matrix as a precomputed result.

# MAKE SURE!! import faiss before importing pandas
import sys
sys.path.append("/share/vector-databases/code")
# from vector_database_update import UniprotVD
from vector_database_update import PubmedVD, UniprotVD

import os
import gc
import ast
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

################################################
### Utils
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

means = np.load("../data/fig2-3/pubmed24n_mean.npy")
stds = np.load("../data/fig2-3/pubmed24n_std.npy")

################################################
### Load pmid with mesh
################################################
pmids_mesh = pd.read_csv("../data/fig2-3/PubTator3/Pubtator3_BioREX_pmid_mesh.csv")
pmids_mesh["from_mesh"] = pmids_mesh["from_mesh"].apply(ast.literal_eval)
print("Number of pmids with mesh:", len(pmids_mesh))

################################################
### Load Sentence transformers
################################################
model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
vd = PubmedVD(model_name=model_name,
              window_size=2, slide_size=1, nprobe=1)

model = SentenceTransformer(model_name)


################################################
### Representative disease queries
################################################
disease_names = ["Breast cancer", "Triple-negative breast cancer",
                 "Lung cancer", "Colorectal cancer", "Lymphoma", "Type 2 diabetes"]
disease_embeddings = model.encode(disease_names)

pmid_corrs = []
bs = 2000
for i in tqdm(range(0, len(pmids_mesh), bs)):
    pmids = pmids_mesh["pmid"].values[i:i+bs]

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

    corrs_df = pd.DataFrame(corrs, index=ids, columns=disease_names).reset_index()
    corrs_max_df = corrs_df.groupby("index").max().reset_index()
    pmid_corrs.append(corrs_max_df)
pmid_corrs = pd.concat(pmid_corrs).reset_index(drop=True)
pmid_corrs = pmid_corrs.rename(columns={"index": "pmid"})
pmid_corrs.to_parquet("../data/fig2-3/Repr_diseases_pmid_bert_corrs.parquet", index=False, engine="pyarrow")


################################################
### All MeSH diseases
################################################
all_mesh_diseases_df = pd.read_csv("../data/fig2-3/MeSH/mesh_disease_leaves_w_annotation.csv")

disease_text = all_mesh_diseases_df["label"].str.cat(all_mesh_diseases_df["scope_note"], sep=" ")
disease_embeddings = model.encode(disease_text)

# Split results to prevent memory errors
pmid_corrs_chunk1 = []
pmid_corrs_chunk2 = []
pmid_corrs_chunk3 = []
resulting_pmids =[]

bs = 2000
for i in tqdm(range(0, len(pmids_mesh), bs)):
    pmids = pmids_mesh["pmid"].values[i:i+bs]

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

    corrs_df = pd.DataFrame(corrs, index=ids, columns=all_mesh_diseases_df["mesh_id"].values).reset_index()
    corrs_max_df = corrs_df.groupby("index").max().reset_index(names="pmid")
    resulting_pmids.append(corrs_max_df["pmid"].values)
    pmid_corrs_chunk1.append(corrs_max_df.iloc[:, :1001])
    pmid_corrs_chunk2.append(corrs_max_df.iloc[:, 1001:2001])
    pmid_corrs_chunk3.append(corrs_max_df.iloc[:, 2001:])

resulting_pmids = np.concatenate(resulting_pmids)

pmid_corrs_chunk1 = pd.concat(pmid_corrs_chunk1, axis=0)
pmid_corrs_chunk1.to_parquet("../data/fig2-3/All_MeSH_diseases_pmid_bert_corrs_chunk1.parquet", index=False, engine="pyarrow")
del pmid_corrs_chunk1
gc.collect()

pmid_corrs_chunk2 = pd.concat(pmid_corrs_chunk2, axis=0)
pmid_corrs_chunk2.insert(0, "pmid", resulting_pmids)
pmid_corrs_chunk2.to_parquet("../data/fig2-3/All_MeSH_diseases_pmid_bert_corrs_chunk2.parquet", index=False, engine="pyarrow")
del pmid_corrs_chunk2
gc.collect()

pmid_corrs_chunk3 = pd.concat(pmid_corrs_chunk3, axis=0)
pmid_corrs_chunk3.insert(0, "pmid", resulting_pmids)
pmid_corrs_chunk3.to_parquet("../data/fig2-3/All_MeSH_diseases_pmid_bert_corrs_chunk3.parquet", index=False, engine="pyarrow")
del pmid_corrs_chunk3
gc.collect()