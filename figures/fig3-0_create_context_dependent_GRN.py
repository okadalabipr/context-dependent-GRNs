import time
import gc
import pandas as pd
import numpy as np

####################
### Load network (Context-independent)
####################
# Load gene-gene interactions from BioREX database
relations_df = pd.read_csv("/share/pubtator3/data/input/all_human_gene_interactions_2024-12-18.csv")
all_pairs = np.array(sorted(list(set(relations_df.apply(lambda x: "--".join(sorted([x["from_gene"], x["to_gene"]])), axis=1)))))
print(relations_df.head())
print(relations_df.shape)

#####################
### context-dependent GRNs for all MeSH diseases
#####################
start = time.time()

threshold = 0.2

diseases = []
n_relations = []

for chunk in range(1, 4):
    pmid_corrs = pd.read_parquet(f"../data/fig2-3/All_MeSH_diseases_pmid_bert_corrs_chunk{chunk}.parquet")

    # initialize
    chunk_results = {"pair": all_pairs}

    for col in pmid_corrs.columns:
        if col == "pmid":
            continue
        disease_relations_df = relations_df.copy()
        disease_relations_df["corr"] = disease_relations_df["pmid"].map(pmid_corrs.set_index("pmid")[col])
        disease_relations_df = disease_relations_df.dropna(subset=["corr"]).reset_index(drop=True)
        # Thresholding
        disease_relations_df = disease_relations_df[disease_relations_df["corr"] > threshold].reset_index(drop=True)
        print(f"[{col}] Number of interactions after thresholding:", len(disease_relations_df))
        diseases.append(col)
        n_relations.append(len(disease_relations_df))

        disease_relations_df["pair"] = disease_relations_df.apply(lambda x: "--".join(sorted([x["from_gene"], x["to_gene"]])), axis=1)
        disease_relations_df = disease_relations_df.drop_duplicates(subset=["pmid", "pair"]).reset_index(drop=True)

        disease_relations_df = disease_relations_df[["pair", "corr"]].groupby("pair").agg({"corr": "sum"}).reset_index()
        # normalize counts per million
        disease_relations_df["corr"] = disease_relations_df["corr"] / disease_relations_df["corr"].sum() * 1e6
        # Log transformation to make the distribution more normal
        disease_relations_df["corr"] = np.log1p(disease_relations_df["corr"].values)
        disease_relations_df.columns = ["pair", "weight"]
        pair2weight = dict(zip(disease_relations_df["pair"], disease_relations_df["weight"]))
        weights = np.frompyfunc(lambda x: pair2weight.get(x, 0), 1, 1)(all_pairs).astype(np.float32)
        chunk_results[col] = weights
    
    chunk_results_df = pd.DataFrame(chunk_results)
    chunk_results_df.to_parquet(f"../data/fig2-3/All_MeSH_diseases_log1p_CPM_chunk{chunk}.parquet")

    del pmid_corrs, chunk_results_df
    gc.collect()

elapsed = time.time() - start
print("Elapsed time:", time.strftime("%H:%M:%S", time.gmtime(elapsed)))