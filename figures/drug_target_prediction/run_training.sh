#!/bin/bash

mkdir -p logs

###################
# Run Training
###################

## FRoGS (prior study)
# shRNA
python -u l1000_model.py \
    --outdir 'FRoGS/results/FRoGS_original/' \
    --modeldir 'FRoGS/saved_model/FRoGS_original/' 2>&1 | tee logs/l1000_model-FRoGS_original.log
# cDNA
python -u l1000_model.py \
    --perttype cDNA \
    --outdir 'FRoGS/results/FRoGS_original/' \
    --modeldir 'FRoGS/saved_model/FRoGS_original/' 2>&1 | tee logs/l1000_model-FRoGS_original_cDNA.log

## context-dependent GRN
# shRNA
python -u l1000_model_celltypes.py \
    --outdir 'FRoGS/results/context_dependent_GRN/' \
    --modeldir 'FRoGS/saved_model/context_dependent_GRN/' 2>&1 | tee logs/l1000_model_celltypes.log
# cDNA
python -u l1000_model_celltypes.py \
    --perttype cDNA \
    --outdir 'FRoGS/results/context_dependent_GRN/' \
    --modeldir 'FRoGS/saved_model/context_dependent_GRN/' 2>&1 | tee logs/l1000_model_celltypes_cDNA.log

## context-independent GRN
# shRNA
python -u l1000_model.py \
    --emb_go '../../data/GeneRelNet-context-independent/gene_vec_DPWalk_256.csv' \
    --outdir 'FRoGS/results/context_independent_GRN/' \
    --modeldir 'FRoGS/saved_model/context_independent_GRN/' 2>&1 | tee logs/l1000_model-context_independent.log
# cDNA
python -u l1000_model.py \
    --perttype cDNA \
    --emb_go '../../data/GeneRelNet-context-independent/gene_vec_DPWalk_256.csv' \
    --outdir 'FRoGS/results/context_independent_GRN/' \
    --modeldir 'FRoGS/saved_model/context_independent_GRN/' 2>&1 | tee logs/l1000_model-context_independent_cDNA.log


## Omnipath
# shRNA
python -u l1000_model.py \
    --emb_go '../../data/GeneRelNet-omnipath/gene_vec_omnipath_256.csv' \
    --outdir 'FRoGS/results/omnipath/' \
    --modeldir 'FRoGS/saved_model/omnipath/' 2>&1 | tee logs/l1000_model-omnipath.log
# cDNA
python -u l1000_model.py \
    --perttype cDNA \
    --emb_go '../../data/GeneRelNet-omnipath/gene_vec_omnipath_256.csv' \
    --outdir 'FRoGS/results/omnipath/' \
    --modeldir 'FRoGS/saved_model/omnipath/' 2>&1 | tee logs/l1000_model-omnipath_cDNA.log
## STRING
# shRNA
python -u l1000_model.py \
    --emb_go '../../data/GeneRelNet-STRING/gene_vec_STRING_256.csv' \
    --outdir 'FRoGS/results/STRING/' \
    --modeldir 'FRoGS/saved_model/STRING/' 2>&1 | tee logs/l1000_model-STRING.log
# cDNA
python -u l1000_model.py \
    --perttype cDNA \
    --emb_go '../../data/GeneRelNet-STRING/gene_vec_STRING_256.csv' \
    --outdir 'FRoGS/results/STRING/' \
    --modeldir 'FRoGS/saved_model/STRING/' 2>&1 | tee logs/l1000_model-STRING_cDNA.log


###################
# Consensus Gene Rank Calculation
###################
python -u calculate_consensus_gene_rank.py 2>&1 | tee logs/calculate_consensus_gene_rank.log