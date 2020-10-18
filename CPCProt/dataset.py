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
from CPCProt.tokenizer import Tokenizer
from tape.datasets import LMDBDataset, FastaDataset, dataset_factory


class PfamDataset(Dataset):
    """
    Modified from https://github.com/songlab-cal/tape/blob/master/tape/datasets.py
    Creates the Pfam Dataset. Modified to trim lengths for the patched model.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 in_memory: bool = False,
                 min_len: int = 0,
                 max_len: int = sys.maxsize,
                 scramble: bool = False):
        super().__init__()
        self.tokenizer = Tokenizer()
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
                 tokenizer: Tokenizer = 'iupac',
                 scramble=False):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        if isinstance(tokenizer, str):
            tokenizer = Tokenizer(vocab=tokenizer)
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
                    pseudolabel = int(line[5])

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
