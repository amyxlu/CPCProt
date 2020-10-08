#!/bin/bash

set -e

git_dir=$(dirname `pwd`)
cpc_models_folder="/scratch/hdd001/home/haoran/ProteinGIMData/model_ckpts/vremote/patched2/"
output_folder="/scratch/hdd001/home/haoran/ProteinGIMData/model_ckpts/finetuning_simple/"
data_root='/scratch/hdd001/home/haoran/ProteinGIMData/'

slurm_pre="--partition t4v2 --gres gpu:1 --mem 50gb -c 4"

current=$(pwd -P)
cd "${git_dir}/src" 

# for model in bert unirep cpc; do
for model in cpc; do
    for task in fluorescence stability remote_homology secondary_structure; do
        for head in kNN LR; do
            if [[ ${model} == "LR" ]]; then
                options=("--LR_C 1.0" "--LR_C 1e-2" "--LR_C 1e-4")
            else
                options=("--knn_k 1" "--knn_k 5" "--knn_k 10")
            fi
            
            i=0
            for opt in "${options[@]}"; do            
                if [[ ${model} == "cpc" ]]; then
                    for cpc_model in 1066743 1071985 1071953; do
                        sbatch ${slurm_pre} --output "${current}/logs/${cpc_model}_${task}_${head}_${i}.out" finetune_simple.py \
                            --model_type cpc \
                            --model_folder "${cpc_models_folder}/${cpc_model}/" \
                            --head_type ${head} \
                            --task ${task} \
                            --output_folder "${output_folder}/patched_${cpc_model}/${task}/${head}/${i}/"  \
                            --data_root ${data_root} \
                            ${opt}
                    done
                else
                    sbatch ${slurm_pre} --output "${current}/logs/${model}_${task}_${head}_${i}.out" finetune_simple.py \
                        --model_type ${model} \
                        --head_type ${head} \
                        --task ${task} \
                        --output_folder "${output_folder}/${model}/${task}/${head}/${i}" \
                        --data_root ${data_root} \
                        ${opt}
                fi
                i=$((i+1))
            done
        done
    done
done

cd "${current}"
