#!/bin/bash

CSV_FILE="../data/fig4/DiSignAtlas_human_diseases_more_than_10_reports.csv"

while IFS=, read -r col1 col2 mesh_id other_columns; do
    if [[ "$mesh_id" == "mesh_id" ]]; then
        continue
    fi

    python -u fig4-0_DiSignAtlas_embedding.py --disease "$mesh_id" | tee logs/fig4-0_DiSignAtlas_embedding_"$mesh_id".log
done < <(tail -n +2 "$CSV_FILE")
