from generate_finetune_jsons import write_json
import subprocess
from sklearn.model_selection import ParameterGrid
import os

cpc_models_folder="/scratch/hdd001/home/haoran/ProteinGIMData/model_ckpts/vremote/patched2/"
slurm_pre = '--partition t4v2 --gres gpu:1 --mem 30gb -c 5'

freeze_CPC = False

models = {} #output_name: [model_dir, emb_type]

for i in [1071953, 1071985]: 
    models['patched_%s'%i] = [os.path.join(cpc_models_folder, str(i)), 'patched_cpc']

tasks = ['fluorescence','stability','remote_homology', 'secondary_structure']

grid = ParameterGrid({
'batch_size' : [16, 32, 128, 256],
'lr' : [5e-3, 1e-4, 1e-5],
'num_epochs' : [10, 20, 30]
})

for n in models:
    try:
        write_json(model_dir = models[n][0], emb_type = models[n][1], inner_folder = n, freeze_CPC = freeze_CPC)
    except Exception as e:
        print("Error writing json for %s"%n, e)
        continue
    for t in tasks:
        print('Starting', n, t)
        for c, params in enumerate(grid):
            batch_size = params['batch_size']
            lr = params['lr']
            num_epochs = params['num_epochs']
            subprocess.check_call(f'sbatch {slurm_pre} --output ./logs/{n}_{t}_{c}.out finetune.sh {t} {n} {batch_size} {lr} {num_epochs} {c}', shell = True)
