import os
import numpy as np
import pandas as pd

def rank2logit(rank):
    rank = np.clip(rank, 1e-4, 1-1e-4)
    logit = np.log10((1-rank)/rank)
    return logit

#####################
# Load Target Genes
#####################
target_file = "FRoGS/data/cpd_gene_pairs.csv"
t_target=pd.read_csv(target_file)
cpd2target={}
target_gene_id = t_target.Broad_target_gene_id.tolist()

for k,v in zip(t_target.term_name.tolist(), target_gene_id):
    if str(v) == 'nan':
        continue
    ck = k.split('Cpd:')[1].split(':')[0] + '@'
    for t in v.split(':'):
        t = t+'@'
        if ck not in cpd2target:
            cpd2target[ck] = [t]
        else:
            if t not in cpd2target[ck]:
                cpd2target[ck].append(t)
    cpd2target[ck] = list(set(cpd2target[ck]))

delk = []
for k in cpd2target:
    if len(cpd2target[k]) > 5:
        delk.append(k)
for k in delk:
    del cpd2target[k]

target_num = []
for c in cpd2target:
    target_num.append(len(cpd2target[c]))
print('Average tragets per compound:', np.mean(target_num))

#####################
# Load Results
#####################
frogs_results_base = "results/FRoGS_original"
dpwalk_results_base = "results/context_dependent_GRN"
dpwalk_celltypes_base = "results/context_independent_GRN"
omnipath_results_base = "results/omnipath"
string_results_base = "results/STRING"


frogs_results_out = frogs_results_base + "-consensus_gene_rank"
dpwalk_results_out = dpwalk_results_base + "-consensus_gene_rank"
dpwalk_celltypes_out = dpwalk_celltypes_base + "-consensus_gene_rank"
omnipath_results_out = omnipath_results_base + "-consensus_gene_rank"
string_results_out = string_results_base + "-consensus_gene_rank"
os.makedirs(frogs_results_out, exist_ok=True)
os.makedirs(dpwalk_results_out, exist_ok=True)
os.makedirs(dpwalk_celltypes_out, exist_ok=True)
os.makedirs(omnipath_results_out, exist_ok=True)
os.makedirs(string_results_out, exist_ok=True)

for base in [frogs_results_base, dpwalk_results_base, dpwalk_celltypes_base,
             omnipath_results_base, string_results_base]:
    outdir = base + "-consensus_gene_rank"

    files = os.listdir("FRoGS/" + base)
    path_df = pd.DataFrame({"path": files})
    path_df["cpd"] = path_df["path"].str.split("@").str[0]
    
    rank_labels = []
    for cpd, df in path_df.groupby("cpd"):
        rank_dfs = []
        for i, path in enumerate(df["path"]):
            rank_df = pd.read_table(os.path.join(base, path)).set_index("gene")
            rank_df["rank"] = rank_df["rank"] / rank_df["rank"].max()  # Normalize
            rank_df.columns = [f"rank{i}", f"score{i}"]
            rank_dfs.append(rank_df)
        rank_dfs = pd.concat(rank_dfs, axis=1)

        rank_cols = [col for col in rank_dfs.columns if "rank" in col]
        rank_dfs["consensus_rank"] = rank_dfs[rank_cols].min(axis=1)
        rank_dfs = rank_dfs.sort_values("consensus_rank").reset_index()
        rank_dfs["logit"] = rank2logit(rank_dfs["consensus_rank"])

        targets = cpd2target.get(cpd+"@", [])
        if len(targets) > 0:
            targets = [int(t.replace("@", "")) for t in targets]
            rank_dfs["label"] = rank_dfs["gene"].isin(targets).astype(int)
        else:
            rank_dfs["label"] = 0

        rank_dfs.to_csv(os.path.join(outdir, f"{cpd}.tsv"), sep="\t", index=False)
        
        rank_label = rank_dfs[rank_dfs["label"]==1].index.tolist()
        rank_label = [r + 1 for r in rank_label]
        rank_labels.extend(rank_label)
    
    print("-----")
    print(base)
    print("Mean rank of target genes:", np.mean(rank_labels))
    print("-----")



