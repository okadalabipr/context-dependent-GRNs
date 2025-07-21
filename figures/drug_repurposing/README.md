### Drug Repurposing

- This repository contains code related to the **Supplementary Figures**.
- The data splits strictly follow those used in the paper:  
  *TxGNN: Zero-shot prediction of therapeutic use with geometric deep learning and human centered design*  
  *([Huang K. et al., Nature Medicine, 2024](https://www.nature.com/articles/s41591-024-03233-x))*
- For environment setup and dependencies, **please refer to the official repository**:  
  [https://github.com/mims-harvard/TxGNN](https://github.com/mims-harvard/TxGNN)

- To reproduce the data split used in drug repurposing, clone the TxGNN repository:

  ```bash
  git clone https://github.com/mims-harvard/TxGNN.git
  ```

### Feature Generation

- **Disease-related features**
    - Because the disease descriptions used here differ from MeSH terms, we obtain textual descriptions of diseases using Google Search and an LLM (`03_generate_disease_descriptions_google.py`).
    - This step requires both a `GOOGLE_API_KEY` and an `OPENAI_API_KEY`.
    - We provide the precomputed results as `03_disease_descriptions.csv`, so running the script is not mandatory.
    - Subsequently, disease features based on context-dependent GRNs are generated using `04_calculate_pmid_correlation.py` and `06_vectorize_diseases.py`. However, we also provide `06_disease_features.csv`, so these steps are optional.

- **Compound-related features**
    - Using the PubTator3 database, a co-occurrence matrix of compounds and genes is constructed. Dimensionality reduction is then applied to generate compound features (`05_vecterize_compounds.ipynb`).
    - We provide the resulting file as `05_cpd_features.csv`, so running the corresponding scripts is not required.

- Intermediate files are not shared due to their large size. If needed, please contact the authors.
