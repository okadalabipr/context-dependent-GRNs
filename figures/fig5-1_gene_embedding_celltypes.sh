#!/bin/bash

while IFS= read -r celltype; do
    if [[ -n "$celltype" ]]; then
        python fig5-1_gene_embedding_celltypes.py --celltype "$celltype" 2>&1 | tee logs/fig5-1_gene_embedding_celltypes-${celltype}.log
    fi
done < ../data/fig5/FRoGS_celltypes.txt