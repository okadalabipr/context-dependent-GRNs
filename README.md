# Reproducible Code for:  
**Literature-derived, context-aware gene regulatory networks improve biological predictions and mathematical modeling**

This repository provides reproducible code for the study by Tsutsui M. et al. (2025), focused on constructing and analyzing context-aware gene regulatory networks (GRNs) derived from biomedical literature.

<img width="4051" height="2094" alt="fig1" src="https://github.com/user-attachments/assets/784955ac-5f96-400a-8777-f63340b5ed24" />

---

## 📁 figures/

This directory contains scripts and notebooks corresponding to each main figure of the paper:

- **fig2**  
  Detection of disease-relevant PubMed literature using BERT-based embedding similarity, and evaluation of retrieval metrics (e.g., AUROC).

- **fig3**  
  Construction and visualization of context-dependent gene regulatory networks (GRNs).

- **fig4**  
  Quantitative analysis of the relationship between context-dependent GRNs and differentially expressed genes (DEGs).

- **fig5**  
  Generation of cell-type-specific gene embeddings and benchmarking their predictive performance for drug target prediction.

- **fig7**  
  Automation of mathematical model construction using context-dependent GRNs and large language models (LLMs).

---

## 📂 figures/drug_repurposing/

- Supplementary analyses related to **fig3**.  
- Evaluates the predictive performance of drug repurposing for unseen diseases using features derived from context-dependent GRNs.  
- To reproduce results, refer to the **TxGNN** repository for training data, test splits, and environment setup:  
  👉 https://github.com/mims-harvard/TxGNN

---

## 📂 figures/drug_target_prediction/

- Code for training deep learning models using **fig5**’s cell-type-specific gene embeddings.  
- To reproduce results, refer to the **FRoGS** repository for dataset splits, model architecture, and environment setup:  
  👉 https://github.com/chenhcs/FRoGS

---

## 📦 Large Datasets

Due to storage constraints, large datasets are hosted on [Zenodo](https://zenodo.org/record/xxxx) and must be downloaded as needed.

---

## 🤖 Mathematical Modeling with LLMs

The automation of mathematical model generation (fig7) using LLMs is handled in a separate project:  
👉 [BioMathForge](https://github.com/okada-lab/BioMathForge)

Please refer to that repository for installation instructions and usage details.

---

## 📄 License

This code is provided for academic use only under the MIT License.  
Please cite our paper if you find this work useful.

