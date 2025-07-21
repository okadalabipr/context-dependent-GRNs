import argparse
import random
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score


def load_features():
    # Load the features of diseases and compounds
    disease_features = pd.read_csv('./06_disease_features.csv').set_index("disease_name")
    # rename the columns so that they start with "disease_"
    disease_features.columns = ["disease_" + col for col in disease_features.columns]

    cpd_features = pd.read_csv("./05_cpd_features.csv").set_index("drug_name").drop(columns=["drug_id"])
    # rename the columns so that they start with "drug_"
    cpd_features.columns = ["drug_" + col for col in cpd_features.columns]
    
    return disease_features, cpd_features

def test_inference(seed, disease_features, cpd_features):
    # Inference with LightGBM model
    # Use the same negative samples as in the TxGNN
    txgnn_df = pd.read_csv(f"TxGNN/reproduce/test_output/disease_centric_eval_{seed}_all_preds.csv")

    # Note: Drugs with no known associations to any diseases were excluded from the evaluation set.
    # Including such drugs would result in all model predictions being treated as false positives,
    # despite the possibility of uncovering true but previously unreported associations.
    # Therefore, these drugs are reserved for inference only and not used in performance benchmarking.
    txgnn_df = txgnn_df[txgnn_df["drug_name"].isin(cpd_features.index)].reset_index(drop=True)
    txgnn_df["relation"] = txgnn_df["relation"].apply(lambda x: x.replace("rev_", ""))
    txgnn_df = pd.concat([txgnn_df, pd.get_dummies(txgnn_df["relation"])], axis=1)

    # Create features
    test_feat = pd.concat([disease_features.loc[txgnn_df["disease_name"]].reset_index(drop=True),
                          cpd_features.loc[txgnn_df["drug_name"]].reset_index(drop=True),
                          txgnn_df[["contraindication", "indication", "off-label use"]]], axis=1).values.astype("float")


    # Inference
    preds = []
    for i in [0, 1, 6, 10]:
        # Load the LightGBM model
        model = pickle.load(open(f"09_lgbm_model_split{i}_seed{seed}_k5_lr0.03.pkl", "rb"))

        y_pred_proba = model.predict_proba(test_feat)[:, 1]
        preds.append(y_pred_proba)
    preds = np.stack(preds).mean(axis=0)

    txgnn_df["y_pred_1"] = preds

    return txgnn_df
        

def calc_metrics(df, seed):
    metrics_results = []
    for rel in ["contraindication", "indication"]:
        rel_df = df[df[rel]].reset_index(drop=True)
        pos_preds_txgnn = rel_df[rel_df["truth"] == 1]["y_pred"].values
        neg_preds_txgnn = rel_df[rel_df["truth"] == 0]["y_pred"].values

        pos_preds_1 = rel_df[rel_df["truth"] == 1]["y_pred_1"].values
        neg_preds_1 = rel_df[rel_df["truth"] == 0]["y_pred_1"].values

        # sampling
        random.seed(42)
        np.random.seed(42)
        neg_idxes = np.random.choice(len(neg_preds_txgnn), len(pos_preds_txgnn), replace=False)

        neg_preds_txgnn = neg_preds_txgnn[neg_idxes]
        neg_preds_1 = neg_preds_1[neg_idxes]

        labels = np.array([1] * len(pos_preds_txgnn) + [0] * len(neg_preds_txgnn))

        for metric in ["AUROC", "AUPRC"]:
            if metric == "AUROC":
                score_txgnn = roc_auc_score(labels, np.concatenate([pos_preds_txgnn, neg_preds_txgnn]))
                score_1 = roc_auc_score(labels, np.concatenate([pos_preds_1, neg_preds_1]))
            elif metric == "AUPRC":
                score_txgnn = average_precision_score(labels, np.concatenate([pos_preds_txgnn, neg_preds_txgnn]))
                score_1 = average_precision_score(labels, np.concatenate([pos_preds_1, neg_preds_1]))
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            print(f"[seed{seed}]{rel} {metric}: TxGNN {score_txgnn:.4f} Ours {score_1:.4f}")

            metrics_results.append([rel, metric, score_txgnn, score_1])
    metrics_df = pd.DataFrame(metrics_results, columns=["relation", "metric", "score_txgnn", "score_1"])
    metrics_df["seed"] = seed
    return metrics_df

def recall_at_k(df, score_col, k=100):
    # Sort by predicted score in descending order
    df_sorted = df.sort_values(by=score_col, ascending=False)

    # Top-k predictions
    top_k = df_sorted.head(k)

    # Number of true positives in the whole dataset
    total_positives = df['truth'].sum()

    # If there are no positives, return None or 0.0
    if total_positives == 0:
        return None  # or return 0.0 depending on how you want to handle this case

    # Count how many true positives are in the top-k
    true_positives_in_top_k = top_k['truth'].sum()

    # Compute recall@k
    recall_at_k_score = true_positives_in_top_k / total_positives

    return recall_at_k_score

def calc_disease_centric_recall_at_k(df, seed, k=100):
    metrics_results = []
    for rel in ["contraindication", "indication"]:
        rel_df = df[df[rel]].reset_index(drop=True)

        for d in rel_df["disease_name"].unique():
            disease_df = rel_df[rel_df["disease_name"] == d].reset_index(drop=True)
            if len(disease_df) == 0:
                continue
            
            # Calculate recall@k for TxGNN
            recall_txgnn = recall_at_k(disease_df, "y_pred", k)
            # Calculate recall@k for our method
            recall_1 = recall_at_k(disease_df, "y_pred_1", k)

            metrics_results.append([d, rel, "recall_at_k", k, recall_txgnn, recall_1])
    metrics_df = pd.DataFrame(metrics_results, columns=["disease_name", "relation", "metric", "k", "recall_txgnn", "recall_1"])
    metrics_df["seed"] = seed
    return metrics_df


def main():
    # Load features
    disease_features, cpd_features = load_features()

    # Loop through seeds and calculate metrics
    all_metrics = []
    all_metrics_recall = []
    for seed in range(1, 6):
        print(f"Processing seed {seed}...")

        # Test inference
        txgnn_df = test_inference(seed, disease_features, cpd_features)

        # Calculate metrics
        metrics_df = calc_metrics(txgnn_df, seed)
        recall_df = calc_disease_centric_recall_at_k(txgnn_df, seed)
        all_metrics.append(metrics_df)
        all_metrics_recall.append(recall_df)
    
    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_metrics_recall_df = pd.concat(all_metrics_recall, ignore_index=True)

    # Save results
    all_metrics_df.to_csv("09_metrics_results.csv", index=False)
    all_metrics_recall_df.to_csv("09_metrics_recall_results.csv", index=False)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

    main()