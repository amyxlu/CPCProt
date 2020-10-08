import numpy as np
import torch
import warnings


def pad_sequences(sequences, max_seq_len=None, constant_value=0) -> np.ndarray:
    '''
    Modified from https://github.com/songlab-cal/tape/blob/master/tape/datasets.py
    Pad NumPy arrays up to constant length.
    '''
    if not isinstance(sequences[0], np.ndarray):
        # only one sequence, unsqueeze first dim to avoid bugs
        sequences = sequences[None, :]

    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()
    if max_seq_len:
        shape[1] = max_seq_len
    array = np.zeros(shape, sequences[0].dtype) + constant_value

    for arr, seq in zip(array, sequences):
            arrslice = tuple(slice(dim) for dim in seq.shape)
            arr[arrslice] = seq

    return array


def collate_fn(batch):
    '''List[Any] -> Dict[str, torch.Tensor]'''
    # unpack batch and cast to necessary types
    input_ids, clan, family, protein_length = tuple(zip(*batch))
    input_ids = np.array(input_ids)
    clan = torch.LongTensor(clan)
    family = torch.LongTensor(family)
    protein_length = torch.Tensor(protein_length)

    # mask out input seqs with "DROP"
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        mask = (input_ids == np.array(["DROP"] * len(input_ids)))

    input_ids, clan, family, protein_length = \
        input_ids[~mask], clan[~mask], family[~mask], protein_length[~mask]

    # pad sequences breaks with an index error if batch is too tiny
    torch_inputs = torch.from_numpy(pad_sequences(input_ids, 0))

    # technically, "protein length" should be "domain length", but keep the keys consistent with raw LMDB data
    # for toy datasets, the "family" key actually feeds in a pseudolabel. Keep keys consistent for easier training
    return {'primary': torch_inputs,
            'clan': clan,
            'family': family,
            'protein_length': protein_length
            }


def augment_collate_fn(batch):
    '''List[Any] -> Dict[str, torch.Tensor]'''
    # unpack batch and cast to necessary types
    seq1, seq2, len1, len2, family = tuple(zip(*batch))
    seq1 = np.array(seq1)
    seq2 = np.array(seq2)
    len1 = torch.Tensor(len1)
    len2 = torch.Tensor(len2)
    family = torch.LongTensor(family)

    # mask out input seqs with "DROP"
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        mask = (seq1 == np.array(["DROP"] * len(seq1)))
        # print('mask.sum()', mask.sum())

    seq1, seq2, len1, len2, family = seq1[~mask], seq2[~mask], len1[~mask], len2[~mask], family[~mask]

    seq1 = torch.from_numpy(pad_sequences(seq1, 0))
    seq2 = torch.from_numpy(pad_sequences(seq2, 0))

    return {"seq1": seq1,
            "seq2": seq2,
            "len1": len1,
            "len2": len2,
            "family": family}
