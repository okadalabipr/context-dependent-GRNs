import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

def main():
    # TxGNN paper results
    txgnn_df = pd.read_csv("TxGNN/reproduce/result_more_metrics.csv")
    txgnn_df = txgnn_df[(txgnn_df["Split"] == "complex_disease") & (txgnn_df["Metric Name"] == "AUPRC")].reset_index(drop=True)

    # Rename methods
    rename_dict = {"DSD-min": "DSD",
                   "KL-min": "KL",
                   "JS-min": "JS",}
    txgnn_df["Method"] = txgnn_df["Method"].replace(rename_dict)

    # context-dependent gene relation network - lightgbm (cGeneRelNet-LGB)
    df = pd.read_csv("09_metrics_results.csv")
    df = df[df["metric"] == "AUPRC"].reset_index(drop=True)
    df["Method"] = "cGeneRelNet-LGB"
    df["Split"] = "complex_disease"
    df = df.rename(columns={"relation": "Task", "metric": "Metric Name",
                            "score_1": "Metric", "seed": "Seed"})

    txgnn_df = pd.concat([
        txgnn_df,
        df[["Method", "Metric", "Seed", "Split", "Task", "Metric Name"]]
    ], axis=0).reset_index(drop=True)

    # Plot 
    for rel in ["indication", "contraindication"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.stripplot(data = txgnn_df[txgnn_df["Task"] == rel], x = 'Method', y = 'Metric', hue = 'Method', 
                  order = ['KL', 'JS', 'DSD', 'Proximity', 'RGCN', 'HGT', 'HAN', 'BioBERT', 'TxGNN', 'cGeneRelNet-LGB'], alpha = 0.3, ax=ax)
        sns.pointplot(data = txgnn_df[txgnn_df["Task"] == rel], x = 'Method', y = 'Metric', hue = 'Method', 
                    order = ['KL', 'JS', 'DSD', 'Proximity', 'RGCN', 'HGT', 'HAN', 'BioBERT', 'TxGNN', 'cGeneRelNet-LGB'], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUPRC")
        ax.set_xlabel("")
        ax.set_title(f"{rel} - Zero-shot Disease Split")
        fig.savefig(f"09_plot_{rel}.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
    print("Plotting completed.")