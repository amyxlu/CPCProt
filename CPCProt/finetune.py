'''
Entry point for compatibility with built-in TAPE train functions.
'''

from tape.main import create_base_parser, create_train_parser
import CPCProt.model.heads
from tape.main import run_train
import os
import sys
sys.path.append(os.getcwd())

run_train()
