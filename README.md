# CPCProt
Using a contrastive loss to pretrain protein representations by maximizing mutual information between local and sequentially global embeddings.

## Pretraining

Configurations, hyperparameter tracking, and logging is configured using [Sacred](https://sacred.readthedocs.io/en/stable/index.html) via `pretrain_config.json`. Specifying Sacred parameters from the command line is documented [here](https://sacred.readthedocs.io/en/stable/command_line.html). Logging relies on Sacred logging observer classes; more information is [here](https://sacred.readthedocs.io/en/stable/observers.html). By default, the `FileStorageObserver` is used.

To run pretraining with environment packages specified in `environment.txt`:

```
cd src
python pretrain.py
```

To specify hyperparameters from the command line:
```
cd src
python pretrain.py with "batch_size=128" 
```

An example for sweeping hyperparameters with SLURM is in `launch_scripts/sweep_batchsize.sh`. 

We build on top of the [TAPE framework](https://github.com/songlab-cal/tape). To install:

```
pip install tape-proteins==0.3
```

There should be a symlink between the project home directory to the directory where data is stored. Logs will also be stored here, in a subdirectory `logs/`. In the project home directory:

```
ln -s <path_to_directory> data
```

## Finetuning Using LR/kNN Heads
- `src/finetune_simple.py` trains a single model using a LR/kNN head on static CPC/BERT/UniREP embeddings.
- This script also demonstrates how to extract static embeddings from CPC models.
- To train a grid of models on a cluster, use `launch_scripts/finetune_simple.sh`, updating `cpc_models_folder`, `output_folder` and `data_root`

## Finetuning Using TAPE Heads
- We train and evaluate these models using the `tape-train` and `tape-eval` interface.
- We use a light wrapper around the interface (`src/finetune.py` and `src/evaluate.py`), which allows us register downstream heads.
- Unfortunately, the TAPE interface does not allow us to specify downstream hyperparameters (ex: embedding method, CPC model path) as command line arguments - only as a JSON file.
- An example call to the training and evaluation scripts is shown in `launch_scripts/finetune.sh`
- The script `launch_scripts/run_finetuning.py` will generate json files for each hyperparameter combination for each model, and run them on a cluster.  Update the path for `cpc_models_folder` in `run_finetuning.py` and `output_folder` and `data_root` in `finetune.sh`.
