"""
- 09: lightgbm with default parameters except for learning rate
- Consider relations [indication, contraindication, off-label use] and concatenate them as features
"""
import argparse
import random
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.metrics import roc_auc_score, average_precision_score

def parse_args():
    parser = argparse.ArgumentParser(description="Train a LightGBM model")
    parser.add_argument("--learning_rate", type=float, default=0.91, help="Learning rate for LightGBM")
    parser.add_argument("--k", type=int, required=True, help="Number of negative samples")
    parser.add_argument("--split_seed", type=int, required=True, help="Seed used for splitting the dataset")
    parser.add_argument("--learning_seed", type=int, default=42, help="Seed for random number generation")
    return parser.parse_args()

def get_dict_node_index_to_name(i):
    # all nodes
    nodes_df = pd.read_csv("data/node.csv", delimiter="\t")
    nodes_df = nodes_df.set_index("node_index")
    node_id2name = nodes_df.set_index(["node_id", "node_type"])["node_name"].to_dict()


    # Relations to be used
    relations = ["contraindication", "indication", "off-label use", "rev_contraindication", "rev_indication", "rev_off-label use"]

    def validate_id(s):
        if s.endswith(".0"):
            s = s[:-2]
        return s
    
    # Load all nodes and define a dictionary to map the node indices to their names
    node_idx2name = {}
    
    train_df = pd.read_csv(f"complex_disease_train_{i}.csv")
    train_df = train_df[train_df["relation"].isin(relations)].reset_index(drop=True)
    train_df["x_id"] = train_df["x_id"].apply(validate_id)
    train_df["y_id"] = train_df["y_id"].apply(validate_id)
    train_df["x_idx"] = train_df["x_idx"].astype(int)
    train_df["y_idx"] = train_df["y_idx"].astype(int)
    train_df["x_name"] = train_df.apply(lambda row: node_id2name[(row["x_id"], row["x_type"])], axis=1)
    train_df["y_name"] = train_df.apply(lambda row: node_id2name[(row["y_id"], row["y_type"])], axis=1)
    node_idx2name.update(train_df.set_index(["x_idx", "x_type"])["x_name"].to_dict())
    node_idx2name.update(train_df.set_index(["y_idx", "y_type"])["y_name"].to_dict())
    
    valid_df = pd.read_csv(f"complex_disease_valid_{i}.csv")
    valid_df = valid_df[valid_df["relation"].isin(relations)].reset_index(drop=True)
    valid_df["x_id"] = valid_df["x_id"].apply(validate_id)
    valid_df["y_id"] = valid_df["y_id"].apply(validate_id)
    valid_df["x_idx"] = valid_df["x_idx"].astype(int)
    valid_df["y_idx"] = valid_df["y_idx"].astype(int)
    valid_df["x_name"] = valid_df.apply(lambda row: node_id2name[(row["x_id"], row["x_type"])], axis=1)
    valid_df["y_name"] = valid_df.apply(lambda row: node_id2name[(row["y_id"], row["y_type"])], axis=1)
    node_idx2name.update(valid_df.set_index(["x_idx", "x_type"])["x_name"].to_dict())
    node_idx2name.update(valid_df.set_index(["y_idx", "y_type"])["y_name"].to_dict())
    
    test_df = pd.read_csv(f"complex_disease_test_{i}.csv")
    test_df = test_df[test_df["relation"].isin(relations)].reset_index(drop=True)
    test_df["x_id"] = test_df["x_id"].apply(validate_id)
    test_df["y_id"] = test_df["y_id"].apply(validate_id)
    test_df["x_idx"] = test_df["x_idx"].astype(int)
    test_df["y_idx"] = test_df["y_idx"].astype(int)
    test_df["x_name"] = test_df.apply(lambda row: node_id2name[(row["x_id"], row["x_type"])], axis=1)
    test_df["y_name"] = test_df.apply(lambda row: node_id2name[(row["y_id"], row["y_type"])], axis=1)
    node_idx2name.update(test_df.set_index(["x_idx", "x_type"])["x_name"].to_dict())
    node_idx2name.update(test_df.set_index(["y_idx", "y_type"])["y_name"].to_dict())
    
    return node_idx2name

def load_train_dataset(i, split_seed, k, node_idx2name):
    # Relations to be used
    relations = ["contraindication", "indication", "off-label use", "rev_contraindication", "rev_indication", "rev_off-label use"]

    
    # Load the training dataset
    train_pos = pd.read_csv(f"./complex_disease_train_{i}.csv")
    train_pos = train_pos[train_pos["relation"].isin(relations)].reset_index(drop=True)
    train_pos["x_idx"] = train_pos["x_idx"].astype(int)
    train_pos["y_idx"] = train_pos["y_idx"].astype(int)
    train_pos["x_name"] = train_pos.apply(lambda row: node_idx2name[(row["x_idx"], row["x_type"])], axis=1)
    train_pos["y_name"] = train_pos.apply(lambda row: node_idx2name[(row["y_idx"], row["y_type"])], axis=1)
    # Create the relation one-hot encoding
    train_pos["relation"] = train_pos["relation"].str.replace("^rev_", "", regex=True)
    one_hot = pd.get_dummies(train_pos["relation"])
    # Create Disease-Compound pairs
    train_pos = pd.DataFrame(np.stack(train_pos.apply(lambda row: row[["x_name", "y_name"]].values if row["x_type"] == "disease" else row[["y_name", "x_name"]].values, axis=1)),
                             columns=["disease_name", "drug_name"])
    # Concatenate the relation one-hot encoding
    train_pos = pd.concat([train_pos, one_hot], axis=1)
    # Drop duplicate rows
    train_pos = train_pos.drop_duplicates(keep="first").reset_index(drop=True)
    
    # Load the negative samples
    train_neg = pd.read_csv(f"./complex_disease_train_neg_{i}_k{k}_seed{split_seed}.csv")
    train_neg = train_neg[train_neg["relation"].isin(relations)].reset_index(drop=True)
    train_neg["x_idx"] = train_neg["x_idx"].astype(int)
    train_neg["y_idx"] = train_neg["y_idx"].astype(int)
    train_neg["x_name"] = train_neg.apply(lambda row: node_idx2name[(row["x_idx"], row["src_type"])], axis=1)
    train_neg["y_name"] = train_neg.apply(lambda row: node_idx2name[(row["y_idx"], row["dst_type"])], axis=1)
    # Create the relation one-hot encoding
    train_neg["relation"] = train_neg["relation"].str.replace("^rev_", "", regex=True)
    one_hot = pd.get_dummies(train_neg["relation"])
    # Create Disease-Compound pairs
    train_neg = pd.DataFrame(np.stack(train_neg.apply(lambda row: row[["x_name", "y_name"]].values if row["src_type"] == "disease" else row[["y_name", "x_name"]].values, axis=1)),
                             columns=["disease_name", "drug_name"])
    # Concatenate the relation one-hot encoding
    train_neg = pd.concat([train_neg, one_hot], axis=1)
    # Drop duplicate rows
    train_neg = train_neg.drop_duplicates(keep="first").reset_index(drop=True)
    
    return train_pos, train_neg

def load_valid_dataset(i, node_idx2name):
    # Relations to be used
    relations = ["contraindication", "indication", "off-label use", "rev_contraindication", "rev_indication", "rev_off-label use"]
    
    # Load the validation dataset
    df_pos = pd.read_csv(f"./complex_disease_valid_pos_{i}.csv")
    df_pos = df_pos[df_pos["relation"].isin(relations)].reset_index(drop=True)
    df_pos["x_idx"] = df_pos["x_idx"].astype(int)
    df_pos["y_idx"] = df_pos["y_idx"].astype(int)
    df_pos["x_name"] = df_pos.apply(lambda row: node_idx2name[(row["x_idx"], row["src_type"])], axis=1)
    df_pos["y_name"] = df_pos.apply(lambda row: node_idx2name[(row["y_idx"], row["dst_type"])], axis=1)
    # Create the relation one-hot encoding
    df_pos["relation"] = df_pos["relation"].str.replace("^rev_", "", regex=True)
    one_hot = pd.get_dummies(df_pos["relation"])
    # Create Disease-Compound pairs
    df_pos = pd.DataFrame(np.stack(df_pos.apply(lambda row: row[["x_name", "y_name"]].values if row["src_type"] == "disease" else row[["y_name", "x_name"]].values, axis=1)),
                          columns=["disease_name", "drug_name"])
    # Concatenate the relation one-hot encoding
    df_pos = pd.concat([df_pos, one_hot], axis=1)
    # Drop duplicate rows
    df_pos = df_pos.drop_duplicates(keep="first").reset_index(drop=True)
    
    # Load the negative samples
    df_neg = pd.read_csv(f"./complex_disease_valid_neg_{i}.csv")
    df_neg = df_neg[df_neg["relation"].isin(relations)].reset_index(drop=True)
    df_neg["x_idx"] = df_neg["x_idx"].astype(int)
    df_neg["y_idx"] = df_neg["y_idx"].astype(int)
    df_neg["x_name"] = df_neg.apply(lambda row: node_idx2name[(row["x_idx"], row["src_type"])], axis=1)
    df_neg["y_name"] = df_neg.apply(lambda row: node_idx2name[(row["y_idx"], row["dst_type"])], axis=1)
    # Create the relation one-hot encoding
    df_neg["relation"] = df_neg["relation"].str.replace("^rev_", "", regex=True)
    one_hot = pd.get_dummies(df_neg["relation"])
    # Create Disease-Compound pairs
    df_neg = pd.DataFrame(np.stack(df_neg.apply(lambda row: row[["x_name", "y_name"]].values if row["src_type"] == "disease" else row[["y_name", "x_name"]].values, axis=1)),
                          columns=["disease_name", "drug_name"])
    # Concatenate the relation one-hot encoding
    df_neg = pd.concat([df_neg, one_hot], axis=1)
    # Drop duplicate rows
    df_neg = df_neg.drop_duplicates(keep="first").reset_index(drop=True)
    
    return df_pos, df_neg

def load_test_dataset(i, node_idx2name, mode="disease_centric"):
    # Relations to be used
    if mode == "disease_centric":
        relations = ["rev_contraindication", "rev_indication", "rev_off-label use"]
    else:
        relations = ["contraindication", "indication", "off-label use"]
    
    # Load the validation dataset
    df_pos = pd.read_csv(f"./complex_disease_test_pos_{i}.csv")
    df_pos = df_pos[df_pos["relation"].isin(relations)].reset_index(drop=True)
    df_pos["x_idx"] = df_pos["x_idx"].astype(int)
    df_pos["y_idx"] = df_pos["y_idx"].astype(int)
    df_pos["x_name"] = df_pos.apply(lambda row: node_idx2name[(row["x_idx"], row["src_type"])], axis=1)
    df_pos["y_name"] = df_pos.apply(lambda row: node_idx2name[(row["y_idx"], row["dst_type"])], axis=1)
    # Create the relation one-hot encoding
    df_pos["relation"] = df_pos["relation"].str.replace("^rev_", "", regex=True)
    one_hot = pd.get_dummies(df_pos["relation"])
    # Create Disease-Compound pairs
    df_pos = pd.DataFrame(np.stack(df_pos.apply(lambda row: row[["x_name", "y_name"]].values if row["src_type"] == "disease" else row[["y_name", "x_name"]].values, axis=1)),
                          columns=["disease_name", "drug_name"])
    # Concatenate the relation one-hot encoding
    df_pos = pd.concat([df_pos, one_hot], axis=1)
    # Drop duplicate rows
    df_pos = df_pos.drop_duplicates(keep="first").reset_index(drop=True)
    
    # Load the negative samples
    df_neg = pd.read_csv(f"./complex_disease_test_neg_{i}.csv")
    df_neg = df_neg[df_neg["relation"].isin(relations)].reset_index(drop=True)
    df_neg["x_idx"] = df_neg["x_idx"].astype(int)
    df_neg["y_idx"] = df_neg["y_idx"].astype(int)
    df_neg["x_name"] = df_neg.apply(lambda row: node_idx2name[(row["x_idx"], row["src_type"])], axis=1)
    df_neg["y_name"] = df_neg.apply(lambda row: node_idx2name[(row["y_idx"], row["dst_type"])], axis=1)
    # Create the relation one-hot encoding
    df_neg["relation"] = df_neg["relation"].str.replace("^rev_", "", regex=True)
    one_hot = pd.get_dummies(df_neg["relation"])
    # Create Disease-Compound pairs
    df_neg = pd.DataFrame(np.stack(df_neg.apply(lambda row: row[["x_name", "y_name"]].values if row["src_type"] == "disease" else row[["y_name", "x_name"]].values, axis=1)),
                          columns=["disease_name", "drug_name"])
    # Concatenate the relation one-hot encoding
    df_neg = pd.concat([df_neg, one_hot], axis=1)
    # Drop duplicate rows
    df_neg = df_neg.drop_duplicates(keep="first").reset_index(drop=True)
    
    return df_pos, df_neg

def load_features():
    # Load the features of diseases and compounds
    disease_features = pd.read_csv('./06_disease_features.csv').set_index("disease_name")
    # rename the columns so that they start with "disease_"
    disease_features.columns = ["disease_" + col for col in disease_features.columns]

    cpd_features = pd.read_csv("./05_cpd_features.csv").set_index("drug_name").drop(columns=["drug_id"])
    # rename the columns so that they start with "drug_"
    cpd_features.columns = ["drug_" + col for col in cpd_features.columns]
    
    return disease_features, cpd_features

def train_lgbm(X_train, y_train, X_valid, y_valid, learning_rate, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    model = lgb.LGBMClassifier(learning_rate=learning_rate, n_estimators=100000, random_state=seed)
    
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=['average_precision', 'logloss', 'auc'],
              callbacks=[early_stopping(stopping_rounds=1000, verbose=True, first_metric_only=True), lgb.log_evaluation(10)])

    return model
    

def main():
    args = parse_args()
    learning_rate = args.learning_rate
    k = args.k
    split_seed = args.split_seed
    learning_seed = args.learning_seed
    print(f"Learning rate: {learning_rate}, k: {k}, split_seed: {split_seed}, learning_seed: {learning_seed}")
    
    # Load the node index to name mapping
    node_idx2name = get_dict_node_index_to_name(learning_seed)
    
    # Load the features
    disease_features, cpd_features = load_features()
    
    # Load the training dataset
    print("Loading the training dataset...")
    train_pos, train_neg = load_train_dataset(learning_seed, split_seed, k, node_idx2name)
    # Create features
    train_pos = pd.concat([disease_features.loc[train_pos["disease_name"]].reset_index(drop=True),
                           cpd_features.loc[train_pos["drug_name"]].reset_index(drop=True),
                           train_pos[["contraindication", "indication", "off-label use"]]], axis=1)
    train_neg = pd.concat([disease_features.loc[train_neg["disease_name"]].reset_index(drop=True),
                           cpd_features.loc[train_neg["drug_name"]].reset_index(drop=True),
                           train_neg[["contraindication", "indication", "off-label use"]]], axis=1)
    
    train_X = pd.concat([train_pos, train_neg], axis=0).reset_index(drop=True).values.astype("float")
    train_y = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
    
    # Load the validation dataset
    print("Loading the validation dataset...")
    valid_pos, valid_neg = load_valid_dataset(learning_seed, node_idx2name)
    # Create features
    valid_pos = pd.concat([disease_features.loc[valid_pos["disease_name"]].reset_index(drop=True),
                           cpd_features.loc[valid_pos["drug_name"]].reset_index(drop=True),
                           valid_pos[["contraindication", "indication", "off-label use"]]], axis=1)
    valid_neg = pd.concat([disease_features.loc[valid_neg["disease_name"]].reset_index(drop=True),
                           cpd_features.loc[valid_neg["drug_name"]].reset_index(drop=True),
                           valid_neg[["contraindication", "indication", "off-label use"]]], axis=1)
    valid_X = pd.concat([valid_pos, valid_neg], axis=0).reset_index(drop=True).values.astype("float")
    valid_y = np.concatenate([np.ones(len(valid_pos)), np.zeros(len(valid_neg))])

    # Load the test dataset
    print("Loading the test dataset...")
    test_pos, test_neg = load_test_dataset(learning_seed, node_idx2name, mode="disease_centric")
    # Create features
    test_pos_feat = pd.concat([disease_features.loc[test_pos["disease_name"]].reset_index(drop=True),
                          cpd_features.loc[test_pos["drug_name"]].reset_index(drop=True),
                          test_pos[["contraindication", "indication", "off-label use"]]], axis=1)
    test_neg_feat = pd.concat([disease_features.loc[test_neg["disease_name"]].reset_index(drop=True),
                          cpd_features.loc[test_neg["drug_name"]].reset_index(drop=True),
                          test_neg[["contraindication", "indication", "off-label use"]]], axis=1)
    test_X = pd.concat([test_pos_feat, test_neg_feat], axis=0).reset_index(drop=True).values.astype("float")
    test_y = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])
    
    # Train the model
    print("Training the model...")
    model = train_lgbm(train_X, train_y, valid_X, valid_y, learning_rate, seed=learning_seed)

    # Save the model
    with open(f"09_lgbm_model_split{split_seed}_seed{learning_seed}_k{k}_lr{learning_rate}.pkl", 'wb') as f:
        pickle.dump(model, f)

    # Test inference
    print("Testing the model...")
    y_pred_proba = model.predict_proba(test_X)[:, 1]
    # Save the predictions
    test_df = pd.concat([test_pos, test_neg], axis=0).reset_index(drop=True)
    test_df["truth"] = test_y
    test_df["y_pred"] = y_pred_proba
    test_df.to_csv(f"09_lgbm_test_pred_split{split_seed}_seed{learning_seed}_k{k}_lr{learning_rate}.csv", index=False)

    

if __name__ == "__main__":
    main()