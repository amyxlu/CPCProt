#!/bin/bash
cd ../scripts

MODEL="unirep"            # change this 
DATA="50_families.clf_all" # change this 

/h/amyxlu/.conda/envs/gim/bin/python validate_embeddings_by_families.py \
    --train_npz $CPC_PROT_DATA_DIR/pfam/embeddings/${DATA}_${MODEL}.train.npz \
    --val_npz $CPC_PROT_DATA_DIR/pfam/embeddings/${DATA}_${MODEL}.val.npz \
    --train_fasta $CPC_PROT_DATA_DIR/pfam/${DATA}.train.fasta \
    --val_fasta $CPC_PROT_DATA_DIR/pfam/${DATA}.val.fasta \
    --embed_method pooled  \
    --target family \
    > ../results/pfam_validation_${DATA}_${MODEL}.txt

cd ../launch_scripts
