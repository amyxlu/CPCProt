import argparse
import pickle as pkl
from pathlib import Path
import torch
import numpy as np
from CPCProt.dataset import FASTADataset
from CPCProt.collate_fn import collate_fn
from CPCProt import CPCProtModel, CPCProtEmbedding

parser = argparse.ArgumentParser()
parser.add_argument("--fasta_file", type=Path, required=True, help="Path to input FASTA file to be embedded.")
parser.add_argument("--output_file", type=Path, required=False, help="Path to output file name (use the .pkl file extension). Uses current directory by default.", default="./embeddings.pkl")
parser.add_argument("--model_weights", type=Path, required=True, help="Path to directory where CPCProt model weights are stored.")
parser.add_argument("--pooling_method", type=str, required=False, choices=["z_mean", "c_mean", "c_final"], help="Method for pooling output to obtain embeddings. Defaults to z_mean.", default="z_mean")
parser.add_argument("--batch_size", type=int, required=False, default=128, help="Batch size to use when embedding.")
args = parser.parse_args()

model = CPCProtModel()
embedder = CPCProtEmbedding(model)
dataset = FASTADataset(args.fasta_file)
loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=args.batch_size)

all_embs = []
for batch in loader:
    if args.pooling_method == "z_mean":
        emb = embedder.get_z_mean(batch)
    elif args.pooling_method == "c_mean":
        emb = embedder.get_c_mean(batch)
    elif args.pooling_method == "c_final":
        emb = embedder.get_c_final(batch)
    else:
        print("--pooling_method must take one of 'z_mean', 'c_mean', or 'c_final'.")
        raise NotImplementedError

    emb = emb.detach().cpu().numpy()
    all_embs.append(emb)

all_embs = np.concatenate(all_embs, axis=0)
with open(args.output_file, "wb") as f:
    pkl.dump(all_embs, f)

print(f"Embedding file saved to {args.output_file}.")
