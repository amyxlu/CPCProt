#!/bin/bash

set -e

git_dir=$(dirname `pwd`)
output_folder="/scratch/hdd001/home/haoran/ProteinGIMData/model_ckpts/finetuning/"
data_root='/scratch/hdd001/home/haoran/ProteinGIMData/'

task=$1
batch_size=$3
lr=$4
num_epochs=$5

if [ -z "$6" ]
then 
	folder_suffix=""
else
	folder_suffix=$6
fi
results_base_dir="${output_folder}/$2/$1/$6/"

configs=$(ls ${task}/$2 | grep .json)

case $task in 
	secondary_structure)
		splits=(valid cb513 casp12 ts115)
		model_name="CPCSeqToSeq"
		metric="accuracy"
		;;
	
	fluorescence | stability)
		splits=("valid" "test")
		model_name="CPCValueClf"
		metric="spearmanr mse"
		;;
	
	remote_homology)
		splits=(valid test_fold_holdout test_family_holdout test_superfamily_holdout)
		model_name="CPCSeqClf"
		metric="accuracy"
		;;
	
	*)
		echo "Bad task!"
		exit 1
		;;
esac

current=$(pwd -P)
cd "${git_dir}/src" 

for config in ${configs[@]}; do
    python finetune.py ${model_name} ${task} \
        --model_config_file "${git_dir}/launch_scripts/${task}/$2/${config}" \
        --learning_rate ${lr} \
        --output_dir "${results_base_dir}" \
        --data_dir ${data_root} \
        --batch_size ${batch_size} \
        --num_train_epochs ${num_epochs} \
        --eval_freq 5 \
        --save_freq 10000 \

    recent=$(ls "${results_base_dir}" | sort | tail -n 1) # most recent model folder

    for split in ${splits[@]}; do
        python evaluate.py ${model_name} ${task} "${results_base_dir}/${recent}" \
            --metrics ${metric} \
            --data_dir ${data_root} \
            --model_config_file "${results_base_dir}/${recent}/config.json" \
            --split ${split} >> "${results_base_dir}/${recent}/${split}_results.log"
    done
done
    
cd "${current}"
