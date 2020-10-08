import json
import argparse
import os
import copy
from pathlib import Path
import re

def write_json(model_dir = None, emb_type = 'patched_cpc', inner_folder = '', freeze_CPC = False):
    base_config = {
                "CPC_args_path": os.path.join(model_dir, 'config.json'),
                "freeze_CPC": freeze_CPC,
                'emb_type': emb_type}

    ckpts = [str(Path(model_dir)/'best.ckpt')]
    
    tasks = {
        'fluorescence': 'emb',
        'stability': 'emb',
        'remote_homology' : 'emb',
        'secondary_structure': 'seq',
        'contact_prediction' : 'seq'
    }

    grid = {}
    grid['emb'] = ['get_z_mean', 'get_c_final'] 
    grid['seq'] = ['get_z_patched_seq'] 

    for task in tasks:
        for model in ckpts:
            for c, i in enumerate(grid[tasks[task]]):
                Path(os.path.join(task, inner_folder)).mkdir(parents = True, exist_ok = True)
                out = os.path.join(task, inner_folder, 'config%s_%s.json'%(c+1, os.path.basename(model)[:-5]))
                config = copy.deepcopy(base_config)
                config['CPC_model_path'] = model
                config['emb_method'] = i
                with open(out, 'w') as outfile:
                    json.dump(config, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cpt_model_dir', type = str)
    parser.add_argument('model_type', type = str)
    parser.add_argument('inner_folder', type = str, default = '')
    # cpc_model_dir must contain "best.ckpt" and "config.json"
    args = parser.parse_args()
    # if using main interface, always just write it to task root folder
    write_json(args.cpt_model_dir, args.model_type, args.inner_folder)

