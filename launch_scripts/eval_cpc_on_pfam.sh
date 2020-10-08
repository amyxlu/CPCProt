#!/bin/bash
cd ../scripts

SACRED_ID=$1                             # change this 
DATA="toy_100_families_50_per_fam"       # change this
EMBED=$2                                 # "z_mean" / "c_mean" / "c_final"
EXPERIMENT="debug_training"

RESULTS_DIR=$CPC_PROT_DATA_DIR/logs/CPCProt/${EXPERIMENT}/${SACRED_ID}/artifacts
mkdir -vp $RESULTS_DIR

/h/amyxlu/.conda/envs/gim/bin/python cpc_eval_on_pfam.py \
    --train_fasta $CPC_PROT_DATA_DIR/pfam/${DATA}.train.fasta \
    --val_fasta $CPC_PROT_DATA_DIR/pfam/${DATA}.val.fasta \
    --model_ckpt_dir $CPC_PROT_DATA_DIR/model_ckpts/${EXPERIMENT}/${SACRED_ID} \
    --figure_prefix $RESULTS_DIR/${EMBED}_${DATA} \
    --result_prefix $RESULTS_DIR/${EMBED}_${DATA}_clf \
    --embed_method $EMBED \

cd ../launch_scripts
