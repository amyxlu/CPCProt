#!/bin/bash

for batch_size in 64 128 256 512 1024 2048; do
    sbatch pretrain.vr1.slrm ${batch_size} 
    sbatch pretrain.vr1.slrm ${batch_size}
done
