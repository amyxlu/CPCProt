'''
Custom datasets, in similar style as 
https://github.com/songlab-cal/tape/blob/master/tape/datasets.py
'''

import sys
from typing import Union, List, Tuple, Sequence, Dict, Any
from pathlib import Path
import lmdb
import pickle as pkl
import numpy as np
from torch.utils.data import Dataset
from tape.tokenizers import TAPETokenizer
from tape.datasets import LMDBDataset, FastaDataset, dataset_factory


class PfamDataset(Dataset):
    """
    Modified from https://github.com/songlab-cal/tape/blob/master/tape/datasets.py
    Creates the Pfam Dataset. Modified to trim lengths for the patched model.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 # split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False,
                 min_len: int = 0,
                 max_len: int = sys.maxsize,
                 scramble: bool = False):
        super().__init__()
        # if split not in ('train', 'valid', 'holdout'):
        #     raise ValueError(
        #         f"Unrecognized split: {split}. "
        #         f"Must be one of ['train', 'valid', 'holdout']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        self._min_len = min_len
        self._max_len = max_len
        self._scramble = scramble

        self.num_too_short = 0
        self.num_too_long = 0

        data_path = Path(data_path)
        # data_file = f'pfam/pfam_{split}.lmdb'
        # self.data = dataset_factory(data_path / data_file, in_memory)
        self.data = dataset_factory(data_path, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # must use `convert_tokens_to_ids` to avoid adding the [CLS] and [SEP] tokens:
        token_ids = np.array(self.tokenizer.convert_tokens_to_ids(item['primary']))

        ## added to trim lengths:
        if len(token_ids) < self._min_len:
            self.num_too_short += 1
            token_ids = "DROP"
        elif len(token_ids) > self._max_len:
            self.num_too_long += 1
            token_ids = token_ids[:self._max_len]

        if self._scramble:
            np.random.shuffle(token_ids)
        ######

        return token_ids, item['clan'], item['family'], item['protein_length']


class FASTADataset(Dataset):
    def __init__(self,
                 data_file: Path,
                 min_len: int = 0,
                 max_len: int = 100000000,
                 tokenizer: TAPETokenizer = 'iupac',
                 scramble=False):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        self.min_len = min_len
        self.max_len = max_len
        self.num_too_short = 0
        self.num_too_long = 0
        self.data = dict()

        idx = 0  # may or may not always correspond to the ID in the Fasta...

        with open(data_file, 'r') as f:
            for line in f:
                if line[0] == ">":
                    line = line.rstrip().split("_")
                    clan = int(line[3])
                    pseudolabel = int(line[7])

                else:
                    seq = line.rstrip()
                    if len(seq) < self.min_len:
                        self.num_too_short += 1
                        continue

                    elif len(seq) > self.max_len:
                        self.num_too_long += 1
                        seq = seq[:self.max_len]

                    self.data[idx] = {
                        'primary': seq,
                        'pseudolabel': pseudolabel,
                        'clan': clan,
                        'protein_length': len(seq)
                    }
                    idx += 1

        self._num_examples = len(self.data)
        self._scramble = scramble

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = np.array(self.tokenizer.convert_tokens_to_ids(item['primary']))
        if self._scramble:
            np.random.shuffle(token_ids)

        return np.array(token_ids), item['pseudolabel'], item['clan'], item['protein_length']

    def __len__(self):
        return len(self.data)


class AugmentDataset(Dataset):
    def __init__(self,
                 data_file: Union[str, Path],
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 num_augmentations: int = None,
                 add_special_tokens: bool = True,
                 min_len: int = 0,
                 max_len: int = sys.maxsize,
                 scramble: bool = False):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))

        self._env = env
        self._num_examples = num_examples
        self._num_augmentations = num_augmentations  # only for the HMM augmentation data, None otherwise
        self._add_special_tokens = add_special_tokens

        self._min_len = min_len
        self._max_len = max_len
        self._scramble = scramble

        self.num_too_short = 0
        self.num_too_long = 0

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        # grab corresponding augmentations of a given family
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        else:
            with self._env.begin(write=False) as txn:
                subdict = pkl.loads(txn.get(str(index).encode()))

        # sample two augmentations from (0, self._num_augmentations)
        if self._num_augmentations:
            # aka using HMM-generated data
            aug_idx1, aug_idx2 = np.random.randint(0, self._num_augmentations, (2,))
        else:
            # aka using real homologs data
            num_seqs_in_family = len(subdict)
            aug_idx1, aug_idx2 = np.random.randint(0, num_seqs_in_family, (2,))
        seq1, seq2 = subdict[aug_idx1], subdict[aug_idx2]

        # encode -- [CLS] and [SEP] as necessary
        if self._add_special_tokens:
            seq1 = self.tokenizer.encode(seq1)
            seq2 = self.tokenizer.encode(seq2)
        else:
            seq1 = self.tokenizer.convert_tokens_to_ids(seq1)
            seq2 = self.tokenizer.convert_tokens_to_ids(seq2)

        # add placeholder token for sequences to drop
        if (len(seq1) < self._min_len) or (len(seq2) < self._min_len):
            self.num_too_short += 1
            seq1, seq2 = "DROP", "DROP"

        elif (len(seq1) > self._max_len) or (len(seq2) > self._max_len):
            self.num_too_long += 1
            seq1 = seq1[:self._max_len]
            seq2 = seq2[:self._max_len]

        # option for a sanity check
        if self._scramble:
            np.random.shuffle(seq1)
            np.random.shuffle(seq2)

        # also return the original index, which denotes the family
        # return seq1, seq2, aug_idx1, aug_idx2, index
        return seq1, seq2, len(seq1), len(seq2), index
