import torch
import torch.nn as nn

def tanh_clip(x, clip_val=10.):
    '''
    soft clip values to the range [-clip_val, +clip_val]
    https://github.com/Philip-Bachman/amdim-public/blob/master/costs.py#L7
    '''
    if clip_val:
        x_clip = clip_val * torch.tanh((1. / clip_val) * x)
    else:
        x_clip = x
    return x_clip


def calc_nce_k(cfg, z_k, c_t, f, device):
    ''' At a given step $k \in 1,2,...,K$, get a scalar NCE loss.
    '''
    assert z_k.shape[0] == c_t.shape[0]
    N = z_k.shape[0]

    log_softmax = nn.LogSoftmax(dim=1).to(device)
    logits = f(z_k, c_t)
    logits = tanh_clip(logits, cfg.logit_soft_clip_val)
    probs = log_softmax(logits)   # aka p(z|c) / p(z).

    # Shape: (N,N) assert probs.shape == (N,N)
    # print("\t\t-- inside nce --")
    # print("\t\tN (z_k.shape)", N)
    # print("\t\tprobs.size", probs.size())
    # dev_idx = torch.cuda.current_device()
    # print("\t\tdevice idx", dev_idx)
    # print("\t\tdev idx type", type(dev_idx))
    # print("\t\tprobs tensor", probs)

    # Diagonals are the NCE loss for each sample,
    # Since we chose the noise distribution to come from other batch samples
    nce = torch.diag(probs)

    # Calc accuracy
    pred_positive_idx = torch.argmax(probs, dim=1)
    batch_idxs = torch.arange(0, N, device=device)
    correct = torch.eq(pred_positive_idx, batch_idxs).type(torch.float)
    acc = correct.sum() / N

    # Negative expectation over batch
    nce = -torch.mean(nce)
    return nce, acc


def calc_nce_t(cfg, z, c_t, t, critics, device):
    ''' At a given step $t \in 1, 2, ...max_L - K$, return an array of of K NCE losses.'''
    nces_all_k = torch.empty(cfg.K, device=device)
    accs_all_k = torch.empty(cfg.K, device=device)

    for k in range(1, cfg.K+1):  # k is 1-indexed
        z_k = z[:, t + k, :]
        f = critics[k-1]
        nce, acc = calc_nce_k(cfg, z_k, c_t, f, device)
        nces_all_k[k - 1] = nce
        accs_all_k[k - 1] = acc

    return nces_all_k, accs_all_k


def calc_nce(cfg, z, c, mask, critics, device, _run=None, global_itr=None, str_code=""):
    # initialize array of T x K results; to make masking indexing easier,
    # also include positions before min_t and outside of calculated positions
    max_L = z.shape[1]
    all_nces = torch.zeros(max_L, cfg.K, device=device)
    all_accs = torch.zeros(max_L, cfg.K, device=device)
    N = z.shape[0]

    # Loop through timesteps, from min_t up until max_L-K
    for t in range(cfg.min_t, max_L - cfg.K):
        c_t = c[:, t, :]
        nces_t, accs_t = calc_nce_t(cfg, z, c_t, t, critics, device)
        all_nces[t, :] = nces_t
        all_accs[t, :] = accs_t

    # Mask out padded regions from the loss calculation
    mask = mask[:, :, None].expand((N, max_L, cfg.K))   # broadcast to K elements in last dimension
    all_nces = all_nces[None, :, :].expand((N, max_L, cfg.K))
    all_accs = all_accs[None, :, :].expand((N, max_L, cfg.K))
    all_nces = mask * all_nces
    all_accs = mask * all_accs

    if torch.any(mask.sum(dim=0).sum(dim=0) == 0):
        print("Error: mask sum along length dimension is 0")
        print("mask", mask)

    avg_nce_k = all_nces.sum(dim=0).sum(dim=0) / mask.sum(dim=0).sum(dim=0)
    avg_acc_k = all_accs.sum(dim=0).sum(dim=0) / mask.sum(dim=0).sum(dim=0)
    if _run and global_itr:
       for k in range(cfg.K):
           assert not torch.isnan(avg_nce_k[k])
           assert not torch.isnan(avg_acc_k[k])
           _run.log_scalar(f"{str_code}_nce_k_{k}", avg_nce_k[k], global_itr)
           _run.log_scalar(f"{str_code}_acc_k_{k}", avg_acc_k[k], global_itr)
           print("global_itr", global_itr)
           print("avg_acc_k", avg_acc_k[k])

    nce = all_nces.sum() / mask.sum()  # expectation over non-padded
    acc = all_accs.sum() / mask.sum()
    assert not torch.any(torch.isnan(nce)), str(all_nces)
    assert not torch.any(torch.isnan(acc)), str(all_accs)
    return nce, acc


def get_nce_array_per_k(cfg, z, c, mask, critics, device, *args):
    max_L = z.shape[1]
    all_nces = torch.zeros(max_L, cfg.K, device=device)
    all_accs = torch.zeros(max_L, cfg.K, device=device)
    N = z.shape[0]

    # Loop through timesteps, from min_t up until max_L-K
    for t in range(cfg.min_t, max_L - cfg.K):
        c_t = c[:, t, :]
        nces_t, accs_t = calc_nce_t(cfg, z, c_t, t, critics, device)
        all_nces[t, :] = nces_t
        all_accs[t, :] = accs_t

    # Mask out padded regions from the loss calculation
    mask = mask[:, :, None].expand((N, max_L, cfg.K))   # broadcast to K elements in last dimension
    all_nces = all_nces[None, :, :].expand((N, max_L, cfg.K))
    all_accs = all_accs[None, :, :].expand((N, max_L, cfg.K))
    all_nces = mask * all_nces
    all_accs = mask * all_accs

    if torch.any(mask.sum(dim=0).sum(dim=0) == 0):
        print("Error: mask sum along length dimension is 0")
        print("mask", mask)

    avg_nce_k = all_nces.sum(dim=0).sum(dim=0) / mask.sum(dim=0).sum(dim=0)
    avg_acc_k = all_accs.sum(dim=0).sum(dim=0) / mask.sum(dim=0).sum(dim=0)
    return avg_nce_k, avg_acc_k
