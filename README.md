# CPCProt
Parameter-efficient embeddings for proteins, pretrained using a contrastive loss to maximize mutual information between local and sequentially global embeddings.

**For more details, see: [https://www.biorxiv.org/content/10.1101/2020.09.04.283929v1.full.pdf](https://www.biorxiv.org/content/10.1101/2020.09.04.283929v1.full.pdf).**

## Contents
* [Pretrained Model Weights](#pretrained-model-weights)
* [Embedding Proteins with CPCProt](#embedding-proteins-with-cpcprot)
    * [Installation](#installation)
    * [API for Embedding Protein Sequences](#api-for-embedding-protein-sequences)
    * [Embedding a FASTA File](#embedding-a-fasta-file)
* [Reproducibility](#reproducibility)
    * [Pretraining](#pretraining)
    * [Finetuning](#finetuning)
    
## Pretrained Model Weights

Pretrained model weights are hosted [here](http://hershey.csb.utoronto.ca/CPCProt/weights/).

To download weights for the default (i.e. most parameter-efficient) version of the CPCProt model:

```bash
cd <directory_to_store_weights>
wget http://hershey.csb.utoronto.ca/CPCProt/weights/CPCProt__pretrained/best.ckpt
```
To download the CPCProt_<sub>LSTM</sub> and CPCProt_<sub>GRU_Large</sub> model variants reported in our paper, replace `CPCProt__pretrained` with `CPCProt_LSTM__pretrained` and `CPCProt_GRU_large__pretrained`, respectively.

## Embedding Proteins with CPCProt 
### Installation

```bash
git clone https://github.com/amyxlu/CPCProt.git
cd CPCProt
pip install -e .
```

### API for Embedding Protein Sequences

To embed a single sequence using the default model configurations, this HuggingFace-like interface can be used:


```python
import torch
from CPCProt.tokenizer import Tokenizer
from CPCProt import CPCProtModel, CPCProtEmbedding

ckpt_path = "data/best.ckpt"  # Replace with actual path to CPCProt weights
model = CPCProtModel()
model.load_state_dict(torch.load(ckpt_path))
embedder = CPCProtEmbedding(model)
tokenizer = Tokenizer()

# Example primary sequence
seq = "LITRSVSRPLRYAVDIIEDIAQGNLRRDVSVTGKDEVSRLLAAMSSQRERLSA"

# Tokenize and convert to torch tensor
input = torch.tensor([tokenizer.encode(seq)])   # (1, L)

# We note three ways to obtain pooled embeddings from CPCProt.
# z_mean and c_mean are the averages of non-padded tokens in z and c, respectively.
# In our paper, we find that z_mean is best for tasks where local effects
# are important (e.g. deep mutational scanning tasks)
# c_final is the final position of the context vector.
# We find that this is best for tasks where global information
# is important (e.g. remote homology tasks).
z_mean = embedder.get_z_mean(input)   # (1, 512)
c_mean = embedder.get_c_mean(input)   # (1, 512)
c_final = embedder.get_c_final(input)  # (1, 512)

# $z$ is the output of the CPCProt encoder
z = embedder.get_z(input)  # (1, L // 11, 512)

# $c$ is the output of the CPCProt autoregressor
c = embedder.get_c(input)  # (1, L // 11, 512)
```

Instantiating `CPCProtModel()` will create a model using the default model configurations for embedding dimension, patch length, `K`, etc.

To embed using the CPCProt_<sub>LSTM</sub> and CPCProt_<sub>GRU_Large</sub> model variants, the `CPCProtConfig` class attributes must be modified.

```python
import json
import torch
from CPCProt import CPCProtConfig, CPCProtModel, CPCProtEmbedding 

config = CPCProtConfig()

# Update config class attributes with JSON config
with open("model_configs/gru_large.json") as f:  # Replace if path to config file is different
    config_dict = json.load(f)
config.__dict__ = config_dict 
model = CPCProtModel(config)

ckpt_path = "data/gru_large/best.ckpt"  # Replace with actual path to CPCProt_GRU_Large weights
state_dict = dict(torch.load(ckpt_path))

# GRU_Large and LSTM variants were trained on multi-GPUs using DataParallel
# Use this hack get state_dict keys to match
for i in list(state_dict.keys()):
    if i.startswith('module.'):
        state_dict[i[7:]] = state_dict[i]
        del state_dict[i]
model.load_state_dict(state_dict)

embedder = CPCProtEmbedding(model)

# Call methods for getting embeddings
# ...
```

### Embedding A FASTA File 
To obtain static embeddings for an entire FASTA file, see `embed_fasta.py`, which saves computed NumPy embeddings as a .pkl file. An example command line usage:

```bash

python embed_fasta.py \
  --fasta_file data/example.fasta \
  --output_file data/example_embeddings.pkl \
  --model_weights data/best.ckpt  # path to where the model weight was saved.

```

`python embed_fasta.py --help` will bring up more options, default settings, etc.
  
For now, please ensure the header is consistent with that in the `data/example.fasta` file. In the future, we will make the FASTA dataloader compatible with more header formats.

## Reproducibility
 
### Pretraining

This repository also includes code for pretraining for reproducibility purposes.

We use Sacred for hyperparameter tracking, logging, and easy configuration from JSON. See the Sacred [documentation](https://sacred.readthedocs.io/en/stable/index.html) for more details. Example configs with pretraining options are in the `model_config` folder.

Specifying Sacred parameters from the command line is documented [here](https://sacred.readthedocs.io/en/stable/command_line.html). Logging relies on Sacred logging observer classes; more information is [here](https://sacred.readthedocs.io/en/stable/observers.html). By default, the `FileStorageObserver` is used.

To make use of the training and evaluation framework for benchmarking against [Tasks Assessing Protein Embeddings (TAPE)](https://github.com/songlab-cal/tape), we build on top of the framework at [https://github.com/songlab-cal/tape](https://github.com/songlab-cal/tape). To install:

```
pip install tape-proteins==0.3
```

To run pretraining (ensure that `tape-proteins` and the packages specified in `environment.txt` are installed):

For consistency when comparing against benchmarks, we pretrain using the same data. For details on how to structure the data directory for pretraining, see the [data documentation in the TAPE repository](https://github.com/songlab-cal/tape#data).

There should be a symlink between the project home directory to the directory where data is stored. Logs will also be stored here, in a subdirectory `logs/`.
To specify hyperparameters from the command line:

```
python pretrain.py with "batch_size=128" 
```

An example for sweeping hyperparameters with SLURM is in `launch_scripts/sweep_batchsize.sh`. 

### Finetuning
`CPCProt/finetune_simple.py` trains a single model using a LR/kNN head on static CPC/BERT/UniREP embeddings. To train a grid of models on a cluster, use `launch_scripts/finetune_simple.sh`, updating `cpc_models_folder`, `output_folder` and `data_root`

As noted in our paper, for consistency with benchmarks, finetuning results using MLP/CNN heads uses the `tape-train` and `tape-eval` interfaces. We use a light wrapper around the interface (`CPCProt/finetune.py` and `CPCProt/evaluate.py`), which allows us register downstream heads. The TAPE interface does not allow us to specify downstream hyperparameters (ex: embedding method, CPC model path) as command line arguments - only as a JSON file.

An example call to the training and evaluation scripts is shown in `launch_scripts/finetune.sh`. The script `launch_scripts/run_finetuning.py` will generate json files for each hyperparameter combination for each model, and run them on a cluster. Update the path for `cpc_models_folder` in `run_finetuning.py` and `output_folder` and `data_root` in `finetune.sh`.

## Contact
amyxlu [at] cs [dot] toronto [dot] edu.
