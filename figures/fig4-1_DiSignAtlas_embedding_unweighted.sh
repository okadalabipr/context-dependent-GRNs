#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

CSV_FILE="../data/fig4/DiSignAtlas_human_diseases_more_than_10_reports.csv"

while IFS=, read -r col1 col2 mesh_id other_columns; do
    if [[ "$mesh_id" == "mesh_id" ]]; then
        continue
    fi

    python -u fig4-1_DiSignAtlas_embedding_unweighted.py --disease "$mesh_id" | tee logs/fig4-1_DiSignAtlas_embedding_unweighted_"$mesh_id".log
done < <(tail -n +2 "$CSV_FILE")
