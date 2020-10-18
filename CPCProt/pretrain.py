import sys
from pathlib import Path
import json
import os
import time
from pprint import pprint

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from CPCProt.model.base_config import CPCProtConfig
from CPCProt.model.cpcprot import CPCProtModel
from CPCProt.validation import Validation
from CPCProt.dataset import PfamDataset
from CPCProt.collate_fn import collate_fn
from CPCProt.schedulers import ConstantLRSchedule, WarmupConstantSchedule, WarmupLinearSchedule

from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
# from neptunecontrib.monitoring.sacred import NeptuneObserver
from pathlib import Path

PROJECT_NAME = "CPCProt"
EXPERIMENT_NAME = "pretrain"
LOG_DIR = Path(f'../data/logs/{PROJECT_NAME}/{EXPERIMENT_NAME}')
SETTINGS.CONFIG.READ_ONLY_CONFIG = False  # allow mutating the config dict inside main()

ex = Experiment(EXPERIMENT_NAME, interactive=True)
ex.observers.append(FileStorageObserver.create(LOG_DIR))
# ex.observers.append(NeptuneObserver())
ex.add_config("../pretrain_config.json")


@ex.automain
def main(_run, _config):
    # use sacred to control parameters for the CPCProtConfig object:
    print("ORIGINAL SACRED ID (for local logs): ", _run._id)
    cfg = CPCProtConfig()
    cfg.__dict__ = _config
    pprint(cfg.__dict__)

    ## Set-up for SLURM job ids ##
    try:
        JOB_ID = os.environ['SLURM_JOB_ID']
        print(f"SLURM job id {os.environ['SLURM_JOB_ID']}, host {os.environ['HOSTNAME']}.")
        _run.log_scalar('SLURM_ID', int(os.environ['SLURM_JOB_ID']))
        _run._id = JOB_ID
        preempt_ckpt_dir = Path(_config['preempt_ckpt_dir']) / os.environ['SLURM_JOB_ID']

    except KeyError:
        print(f"Using local machine")
        _run.log_scalar("SLURM_ID", -1)
        _config['use_multi_gpu'] = False  # workstation has imbalanced GPU usage
        preempt_ckpt_dir = Path("")
        JOB_ID = _run._id

    print("JOB ID: ", JOB_ID)
    model_ckpt_dir = Path(f"../data/model_ckpts/{EXPERIMENT_NAME}/{JOB_ID}")
    figure_savedir = model_ckpt_dir / "artifacts"

    try:
        model_ckpt_dir.mkdir(parents=True, exist_ok=False)
        figure_savedir.mkdir(parents=True, exist_ok=False)
    except:
        print("Model checkpoint figure directory exists from pre-emption. Will continue writing into same location.")
        pass


    ## Datasets ##
    min_len = (cfg.min_t + 1 + cfg.K) * cfg.patch_len

    if cfg.use_toy_data:
        train_dataset = PfamDataset(cfg.toy_train_data, min_len=min_len, max_len=cfg.max_len, scramble=cfg.scramble)
    else:
        train_dataset = PfamDataset(cfg.train_data, min_len=min_len, max_len=cfg.max_len, scramble=cfg.scramble)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_fn, drop_last=False)

    ## Model Set-Up ##
    model = CPCProtModel(cfg)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: ", num_params)
    _run.log_scalar("num_params", num_params)


    ## Multi GPU ##
    PARALLEL = cfg.use_multi_gpu and (torch.cuda.device_count() > 1) and cfg.use_cuda
    device = torch.device("cuda") if cfg.use_cuda else torch.device("cpu")

    if PARALLEL:
        model = nn.DataParallel(model)
    model.to(device)

    with open(model_ckpt_dir / 'config.json', 'w') as f:
        json.dump(cfg.__dict__, f)


    ## Optimizer and Scheduler ##
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    if cfg.scheduler == "constant":
        scheduler = ConstantLRSchedule(optimizer)
    elif cfg.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.scheduler_step_gamma)
    else:
        raise NotImplementedError


    ## Validation ##
    # NOTE: if not controlling for minimum length, then you might get an error when using a larger K.
    # But if the entire batch is being discarded because of insufficient length, it'll also throw an error.
    val_dataset = PfamDataset(cfg.val_data, min_len=cfg.patch_len, max_len=sys.maxsize, scramble=cfg.scramble)
    val_loader = DataLoader(val_dataset, batch_size=cfg.val_batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate_fn, drop_last=False)
    nce_validator = Validation(model, val_loader, parallel=PARALLEL)
    knn_validator = Validation(model, val_loader, embed_method="c_final", figure_savedir=figure_savedir,
                               clf_train_frac=cfg.clf_train_frac, emb_type="patched_cpc")


    ## Pretraining Set Up ##
    print(f"Number of training batches: {len(train_loader)}")
    _run.log_scalar("num_train_batches", len(train_loader))
    start = time.time()

    lowest_val_loss = sys.maxsize
    best_epoch = 0
    epoch = 0
    global_itr = 0
    cur_patience = 0  # test checkpointing if using VR2

    ## Load checkpoint if exists ##
    checkpoint_path = preempt_ckpt_dir / "checkpoint_last.ckpt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        global_itr = checkpoint['global_itr']
        cur_patience = checkpoint['cur_patience']
        cfg.__dict__ = checkpoint['cfg']
        lowest_val_loss = checkpoint['lowest_val_loss']
        best_epoch = checkpoint['best_epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    ## Main train loop ##
    while epoch < cfg.num_epochs:
        model.train()
        epoch_start = time.time()
        _run.log_scalar('lr', scheduler.get_lr()[0], epoch)

        for batch_idx, data in enumerate(train_loader):
            # data loader index doesn't deal well with checkpointing but we'll just let it slide for now
            if global_itr % cfg.log_train_every == 0:
                loss, acc = model(data, _run, global_itr, str_code="train")
                if PARALLEL:
                    loss = loss.mean()  # average across all GPUs
                    acc = acc.mean()

                _run.log_scalar("train_nce_loss", float(loss.item()), global_itr)
                _run.log_scalar("train_nce_acc", float(acc.item()), global_itr)
                print(f"Epoch {epoch} | Global_itr {global_itr} | Train loss {loss} | Train acc {acc}")
            else:
                loss, acc = model(data)
                if PARALLEL:
                    loss = loss.mean()  # average across all GPUs

            # optimization and gradient clipping
            loss.backward()
            # if cfg.grad_clip_max_norm:
            #     nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_max_norm)
            optimizer.step()
            optimizer.zero_grad()

            # preemption checkpointing
            if (global_itr % cfg.ckpt_every == 0) and os.path.exists(cfg.preempt_ckpt_dir):
                checkpoint = {
                    'epoch': epoch,
                    'global_itr': global_itr,
                    "cur_patience": cur_patience,
                    'cfg': cfg.__dict__,
                    'lowest_val_loss': lowest_val_loss,
                    'best_epoch': best_epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, checkpoint_path)

            # update itr!!
            global_itr += 1

        ## Save ckpt each epoch
        epoch_end = time.time()
        epoch_runtime = (epoch_end - epoch_start) / 60
        print(f'-- Epoch {epoch} trained in {epoch_runtime:.2f} minutes -- ')
        _run.log_scalar("completed_epoch", epoch)
        _run.log_scalar("epoch_train_time", float(epoch_runtime), epoch)

        ckpt_fpath = model_ckpt_dir / f"epoch{epoch}.ckpt"
        print(f"-- end of train epoch {epoch} | saved model to {ckpt_fpath} --")
        torch.save(model.state_dict(), ckpt_fpath)

        ## Validation at the end of the epoch
        del data
        _run.log_scalar("train_num_seqs_too_short", train_dataset.num_too_short)

        # Using lowest NCE validation loss to decide which checkpoint to save
        val_loss, val_acc = nce_validator.nce_validate(_run, epoch)
        knn_acc = knn_validator.embedding_validate(_run, epoch)
        print("validation nce acc:", val_acc)
        print("validation nce loss:", val_loss)
        print("validation knn acc:", knn_acc)

        if val_loss < lowest_val_loss:
            cur_patience = 0
            lowest_val_loss = val_loss
            best_epoch = epoch
            print(f" **** Epoch {epoch} | New lowest validation loss: {lowest_val_loss} **** ")
            _run.log_scalar("lowest_val_loss", float(lowest_val_loss), epoch)
            _run.log_scalar("best_epoch", float(best_epoch))

            # touch a file telling us the best epoch, instead of saving a new ckpt
            with open(model_ckpt_dir / f"BEST_EPOCH_{best_epoch}.txt", 'w') as f:
                f.write(f"lowest_val_loss,{lowest_val_loss}")

            ckpt_fpath = model_ckpt_dir / "best.ckpt"
            print(f"Epoch {epoch} Step {global_itr} | Saved model to {ckpt_fpath}")
            torch.save(model.state_dict(), ckpt_fpath)
        else:
            cur_patience += 1
            print(" --> no improvement on validation loss this epoch")
            print(f" --> number of epochs without improvement: [ {cur_patience}/{cfg.patience} ]")
            if cur_patience >= cfg.patience:
                print(" ***** patience reached; early stopping *****")
                end = time.time()
                runtime = (end - start) / 60
                torch.save(model.state_dict(), Path(str(model_ckpt_dir)) / "final.ckpt")
                print(" ***** saving final model *****")
                print(f" Training took {runtime:.2f} to run. Exiting.")
                sys.exit(1)

        ## Update scheduler and epoch counter
        scheduler.step()
        epoch += 1

    ## At training end
    end = time.time()
    runtime = (end - start) / 60
    torch.save(model.state_dict(), Path(str(model_ckpt_dir)) / "final.ckpt")
    print(f"Best epoch {best_epoch}")
    print(f"Saved final model after all epochs to {model_ckpt_dir}/final.ckpt")
    print(f'Completed in {runtime:.2f} minutes')
