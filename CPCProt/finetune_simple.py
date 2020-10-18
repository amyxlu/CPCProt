import os
import sys
sys.path.append(os.getcwd())
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import json
from tape.datasets import FluorescenceDataset, StabilityDataset, RemoteHomologyDataset, SecondaryStructureDataset
from tape.utils.setup_utils import setup_loader
from tape import ProteinBertModel, UniRepModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import random
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from tape.registry import Registry
import json
import pickle
import argparse
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import CPCProt.model.cpcprot as patched_cpc
from CPCProt.model import heads
from CPCProt.model.base_config import CPCProtConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()

parser.add_argument('--model_type', choices=['bert', 'cpc', 'unirep'])
parser.add_argument('--model_folder', type = str, help = 'only used for cpc')
parser.add_argument('--head_type', choices=['kNN', 'LR'])
parser.add_argument('--task', choices=['fluorescence', 'stability', 'remote_homology', 'secondary_structure'])
parser.add_argument('--output_folder', type = str)
parser.add_argument('--knn_k', type = int, default = 5)
parser.add_argument('--LR_C', type = float, default = 1.0)
parser.add_argument('--data_root', type = str)
parser.add_argument('--normalize', action = 'store_true')

args = parser.parse_args()

output_path = Path(args.output_folder)
output_path.mkdir(parents = True, exist_ok = True)

print(vars(args))

with open(output_path/'args.json', 'w') as f:
    json.dump(vars(args), f)
    
task_types = {
    'fluorescence': 'regression',
    'stability': 'regression',
    'remote_homology': 'classification',
    'secondary_structure': 'classification',
}

metrics = {
    'fluorescence': ['spearmanr', 'mse'],
    'stability': ['spearmanr', 'mse'],
    'remote_homology': ['accuracy'],
    'secondary_structure': ['accuracy'],
    
}

sets = {
    'fluorescence': ['valid','test'],
    'stability': ['valid', 'test'],
    'remote_homology': ['valid', 'test_fold_holdout', 'test_family_holdout', 'test_superfamily_holdout'],
    'secondary_structure': ['valid', 'casp12', 'ts115', 'cb513']
}

if args.head_type == 'kNN':
    if task_types[args.task] == 'regression':
        head = KNeighborsRegressor(n_neighbors = args.knn_k)
    else:
        head = KNeighborsClassifier(n_neighbors = args.knn_k)
elif args.head_type == 'LR':
    if task_types[args.task] == 'regression':
        head = Ridge(alpha = 1/(args.LR_C), random_state=42)
    else:
        head = LogisticRegression(C = args.LR_C, random_state=42, solver = 'saga' if args.task == 'secondary_structure' else 'lbfgs', n_jobs = -1)

head = [('clf', head)]
if args.normalize:
    head = [('scaler', StandardScaler())] + head

head = Pipeline(head)
        
funcs = {
    'fluorescence': 'get_z_mean',
    'stability': 'get_z_mean',
    'remote_homology': 'get_c_final',
    'secondary_structure': 'get_c_patched_seq'    
}

if args.model_type == 'cpc':
    base_model_path = Path(args.model_folder)
    cpc_args = CPCProtConfig()
    cpc_args_dict = json.load(open(base_model_path/'config.json', 'r'))
    default_cfg = json.load(open('../pretrain_config.json', 'r'))
    for key in cpc_args_dict:
        try:
            default_cfg[key] = cpc_args_dict[key]
        except:
            pass       

    cpc_args.__dict__ = default_cfg

    base_model = patched_cpc.PatchedCPCModel(cpc_args)
    state_dict = dict(torch.load(base_model_path/'best.ckpt'))
    for i in list(state_dict.keys()):
        if i.startswith('module.'):
            state_dict[i[7:]] = state_dict[i]
            del state_dict[i]
    base_model.load_state_dict(state_dict)
    base_model = heads.CPCProtEmbedding(base_model.to(device).eval(), emb_type = 'patched_cpc')
    emb_func = getattr(base_model, funcs[args.task])
elif args.model_type == 'bert':
    base_model = ProteinBertModel.from_pretrained('bert-base').eval().to(device)
elif args.model_type == 'unirep':
    base_model = UniRepModel.from_pretrained('babbler-1900').eval().to(device)
    
if args.model_type in ['unirep', 'bert']:   
    if args.task == 'secondary_structure':
        emb_func = lambda x: base_model(x['primary'])[0] # n_samples x n_tokens x emb_length
    else:
        emb_func = lambda x: base_model(x['primary'])[1] # n_samples x emb_length
        
if args.task == 'fluorescence':
    dataset_cls = FluorescenceDataset 
elif args.task == 'stability':
    dataset_cls = StabilityDataset 
elif args.task == 'remote_homology':
    dataset_cls = RemoteHomologyDataset
elif args.task == 'secondary_structure':
    dataset_cls = SecondaryStructureDataset
    
train_loader = setup_loader(dataset_cls(str(args.data_root), split = 'train', tokenizer = 'unirep' if args.model_type == 'unirep' else 'iupac')
                      , batch_size = 128 if args.task != 'secondary_structure' else 32,
                         local_rank = -1, n_gpu = 1, gradient_accumulation_steps = 1, num_workers = 1)

def get_embs(loader, emb_func):
    embs = []
    targets = []
    for counter, samples in enumerate(tqdm(loader)):
        data = {'primary': samples['input_ids'].to(device), 'input_mask': samples['input_mask'].to(device), 
            'protein_length': samples['input_mask'].sum(dim = 1).float().to(device)}
        with torch.no_grad():        
            temp = emb_func(data)

        if args.task == 'secondary_structure': 
            embs.append(temp.detach().cpu().flatten(start_dim = 0, end_dim = 1))
            targets.append(samples['targets'].flatten(start_dim = 0, end_dim = 1))
        else:
            embs.append(temp.detach().cpu())
            targets.append(samples['targets'])
    
    return embs, targets

embs, targets = get_embs(train_loader, emb_func)
embs = torch.cat(embs).numpy()
targets = torch.cat(targets).numpy().flatten()

print("Train matrix shape: " + str(embs.shape))

head = head.fit(embs, targets)
def evaluate_on_set(loader, emb_func, head):
    embs, targets = get_embs(loader, emb_func)
    embs = torch.cat(embs).numpy()
    targets = torch.cat(targets).numpy().flatten()
    preds = head.predict(embs)
    
    res = {}
    for m in metrics[args.task]:
        if m =='accuracy':
            res[m] = accuracy_score(targets, preds)
        else:
            res[m] = Registry.get_metric(m)(targets, preds)
    return res, targets, preds

res = {}
for s in sets[args.task]:
    loader = setup_loader(dataset_cls(str(args.data_root), split = s, tokenizer = 'unirep' if args.model_type == 'unirep' else 'iupac')
                      , batch_size = 64 if args.task != 'secondary_structure' else 16,
                         local_rank = -1, n_gpu = 1, gradient_accumulation_steps = 1, num_workers = 1)
    res[s], targets, preds = evaluate_on_set(loader, emb_func, head)
    print(s, res[s])   
    
with open(output_path/'results.json', 'w') as f:
    json.dump(str(res), f)
    
pickle.dump(head, open(output_path/'model', 'wb'))    
