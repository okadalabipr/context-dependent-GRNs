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
### Load Cell-line-related queries
################################################
# This originally derived from GSE92742_Broad_LINCS_cell_info.txt (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742)
cell_info_df = pd.read_csv("../data/fig5/cell_info.csv")

cellline_descs = []
for _, row in cell_info_df.iterrows():
    desc = row["description"][:-1] + "," if row["description"].endswith(".") else row["description"] + ","
    desc = desc + f' referred as {row["cell_id"]}.'
    cellline_descs.append(desc)
cellline_descs_embeddings = model.encode(cellline_descs)


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
    corrs = calc_cosine_similarity(embeddings, cellline_descs_embeddings, mean=means, std=stds)

    corrs_df = pd.DataFrame(corrs, index=ids, columns=cell_info_df["cell_id"].values).reset_index()
    corrs_max_df = corrs_df.groupby("index").max().reset_index()
    pmid_corrs.append(corrs_max_df)
pmid_corrs = pd.concat(pmid_corrs).reset_index(drop=True)
pmid_corrs = pmid_corrs.rename(columns={"index": "pmid"})
pmid_corrs.to_parquet("../data/fig5/cellline_pmid_bert_corrs.parquet", index=False, engine="pyarrow")