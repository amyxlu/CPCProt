import pickle as pkl
import math
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import lmdb
import pickle
from tqdm.auto import tqdm
from CPCProt.tokenizer import Tokenizer
from CPCProt.collate_fn import pad_sequences

def one_hot_encode(x, num_classes):
    '''assumes x is zero-indexed.

    '''
    if not type(x) == np.ndarray:
        x = np.array(x)
    N = len(x)
    # num_classes = int(torch.max(x) + 1)  # add 1 because zero indexing

    out = np.zeros((N, num_classes))
    out[np.arange(N), x] = 1
    return out


def train_val_split(X, y, train_frac):
    idxs = np.arange(X.shape[0])
    np.random.seed(42)
    np.random.shuffle(idxs)
    split_at = math.floor(train_frac * X.shape[0])
    train_idxs, val_idxs = idxs[:split_at], idxs[split_at:]
    X_train, y_train = X[train_idxs], y[train_idxs]
    X_val, y_val = X[val_idxs], y[val_idxs]
    return X_train, y_train, X_val, y_val


def train_val_split_within_family(df, train_num_samples, val_num_samples):
    train_out = pd.DataFrame(columns=df.columns)
    val_out = pd.DataFrame(columns=df.columns)
    families = np.unique(df.family)

    for fam in families:
        tmp = df[df.family == fam]
        assert tmp.shape[0] >= train_num_samples + val_num_samples
        tmptrain = tmp[:train_num_samples]
        tmpval = tmp[train_num_samples : train_num_samples+val_num_samples]

        train_out = pd.concat([train_out, tmptrain])
        val_out = pd.concat([val_out, tmpval])
    return train_out, val_out


def df_to_lmdb(df, lmdb_outfile):
    '''Writes a pd.DataFrame out to a LMDB file, for loading into the 
    LMDBDataset class (this is a great memory mapped way to load data)
    '''
    map_size = df.values.nbytes * 100
    num_examples = df.shape[0]
    
    # create a dictionary with df index as the key. The values
    # will be loaded by the PyTorch dataloader
    d = df.to_dict(orient='index')
    
    # Create a new LMDB environment
    env = lmdb.open(lmdb_outfile, map_size=map_size)

    # Start a new write transaction
    with env.begin(write=True) as txn:
        
        # dataset object needs to know the range of acceptable 
        # indices when grabbing from this memory mapped db.
        txn.put("num_examples".encode(), pkl.dumps(num_examples))
        
        # pickle dictionary and dump to the key:
        for i in range(num_examples):
            item = d[i]
            txn.put(str(i).encode(), pkl.dumps(item))

    env.close()


def lmdb_to_df(lmdb_file):
    env = lmdb.open(lmdb_file)
    lmdb_txn = env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    with env.begin(write=False) as txn:
        num_examples = pickle.loads(txn.get(b'num_examples'))
        
    data = []
    with env.begin(write=False) as txn:
        for key, value in tqdm(lmdb_cursor, total = num_examples):
            temp = pickle.loads(value)
            # there's a record with 'num_examples': int; we don't want that in the df
            if isinstance(temp, dict): 
                temp['id'] = int(key.decode())
                data.append(temp)
                
    df = pd.DataFrame(data).set_index('id')
    return df


def get_TSNE_emb(x, n_components=2, as_df=False):
    tsne = TSNE(n_components=n_components)
    tsne_emb = tsne.fit_transform(x)
    if as_df:
        return pd.DataFrame(tsne_emb)
    else:
        return tsne_emb


def collect_labeled_embeddings(loader, model, device, num_samples=64, target='family'):
    '''
    :param loader: torch.utils.data.DataLoader object
    :param model: TAPE model which returns a tuple (sequence_embedding, pooled_embedding) 
    :param num_samples: how many sample to load? Note: will only load complete batches
    :param target: string, must match a key in the dictionary that the dataloader returns.
    :returns: numpy arrays of the embeddings and a corresponding target array.
    '''

    # print(f'Using {pooler.config.embedding_pooling_method} to obtain embeddings.')
    
    embs = []
    targets = []
    if target != "family":
        # implement if we want to validate by clan or something
        raise NotImplementedError
        
    for batch_idx, data in enumerate(loader): #, total=(num_samples//loader.batch_size)):
        if (batch_idx+1) > (num_samples // loader.batch_size):
            print(f"Loaded {batch_idx * loader.batch_size} testing samples. Exiting")
            break
            
        x = data['primary'].to(device)
        _, emb = model(x)
        embs.append(emb.detach().cpu().numpy())
        targets.append(data[target].cpu().numpy())

    
    embs = np.concatenate(embs, axis=0)
    targets = np.concatenate(targets, axis=0)

    return embs, targets


def get_top_n_families(df, n=3):
    ''' :param df: DataFrame with the same columns as the Pfam data hosted by TAPE folks
    '''
    fams, counts = np.unique(df.family, return_counts=True)
    sorted_tuples = sorted(list(zip(fams, counts)), key=lambda tup:tup[1])[::-1]
    top_n_fams = [t[0] for t in sorted_tuples[:n]]
    top_n_fams_counts = [t[1] for t in sorted_tuples[:n]]
    return top_n_fams, top_n_fams_counts


def grab_families_from_df(df, families_list):
    out_df = pd.DataFrame(columns=df.columns)

    for fam in families_list:
        tmp = df[df.family == fam]
        out_df = pd.concat([out_df, tmp])
        
    return out_df


def calc_conv_output_size(W, F, S, P):
    return ((W - F + 2*P) // S) + 1


def calc_same_width_padding(W, F, S):
    '''Rearrange equation to calculate padding size
    '''
    return np.ceil(((W - 1) * S - W + F) / 2)


def calc_filter_from_output(W, out, S=1, P=0):
    assert S == 1
    assert P == 0
    return W - out + 1


def get_random_rows_from_df(df, num_rows_to_choose):
    df = df.reset_index().drop(['index'], axis=1)
    
    idxs = np.arange(df.shape[0])
    idxs = np.random.choice(idxs, num_rows_to_choose)
    
    df = df.iloc[idxs, :]
    return df


def get_n_samples_per_family(df, families, n):
    out = pd.DataFrame(columns=df.columns)
    for fam in families:
        tmp = df[df.family == fam]
        tmp = get_random_rows_from_df(tmp, n)
        out = pd.concat([out, tmp])
    return out


def write_df_to_fasta(df, outpath):
    with open(outpath, 'w') as f:
        for i in range(df.shape[0]):
            row = df.iloc[i, :]
            f.write(f">FAMILY_{row.family}_CLAN_{row.clan}_PSEUDOID_{i}\n")
            f.write(row.primary + "\n")


def npz_to_embedding(npz_fpath, embed_method):
    ''' '''
    arrays = np.load(npz_fpath, allow_pickle=True)
    keys = list(arrays.keys())

    # grab the first array out of NPZ just to get the embedding dimension
    D = len(arrays[keys[0]].tolist()[embed_method])
    embeds = np.empty((len(keys), D))  # N x D

    for i in range(len(keys)):
        key = keys[i]
        arr = arrays[key].tolist()[embed_method]
        embeds[i, :] = arr
    
    return embeds, keys


def sample_n_rows(arr, n):
    idx = np.arange(len(arr))
    np.random.shuffle(idx)
    return arr[idx[:n]]


def fasta_to_padded_data(fasta_fpath: str):
    tokenizer = Tokenizer()
    families = []
    clans = []
    seqs = []

    with open(fasta_fpath) as f:
        for line in f:
            if line[0] == ">":
                line = line.rstrip().split("_")
                families.append(int(line[1]))
                clans.append(int(line[3]))
            else:
                seq = tokenizer.convert_tokens_to_ids(line.rstrip())
                seqs.append(np.array(seq))

    families = np.array(families)
    clans = np.array(clans)
    seqs = pad_sequences(np.array(seqs))


def get_mean_protein_length_by_family(df):
    mean_protein_lengths = []
    num_seqs = []
    unique_fams = np.unique(df.family)
    
    for fam in unique_fams:
        tmp = df[df.family == fam]
        mean_protein_lengths.append(tmp.protein_length.mean())
        num_seqs.append(tmp.shape[0])
        #print(f"{fam},{tmp.protein_length.mean():.3f}")
    
    out = pd.DataFrame({'family': unique_fams,
                        'mean_protein_lengths': mean_protein_lengths,
                        'num_seqs': num_seqs})
    return out


# CLASSIFICATION ####### 

def logistic_regression(X_train, y_train, X_val, y_val, result_prefix=None, seed=42):
    ''' Multiclass cross-entropy LR. 
    '''
    model = LogisticRegression(multi_class='multinomial',
                                solver='sag',
                                max_iter=10000,
                                random_state=seed).fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    acc = (y_pred == y_val).astype(int).sum() / len(y_pred)
    clf_report = classification_report(y_val, y_pred)
    print("Accuracy: ", acc)
    print(clf_report)

    if result_prefix:
        out_path = f"{result_prefix}_lr.txt"
        with open(out_path, 'w') as f:
            f.write(f"lr accuracy,{acc}\n")
            f.write(str(clf_report))
        print(f"Results written to {out_path}.")

    return y_pred


def k_nearest_neighbor(X_train, y_train, X_val, y_val, result_prefix=None, k=5):
    model = KNeighborsClassifier(n_neighbors=k, p=2).fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = (y_pred == y_val).astype(int).sum() / len(y_pred)
    clf_report = classification_report(y_val, y_pred)
    print("Accuracy: ", acc)
    print(clf_report)

    if result_prefix:
        out_path = f"{result_prefix}_{k}nn.txt"
        with open(out_path, 'w') as f:
            f.write(f"{k}-nn accuracy,{acc}\n")
            f.write(str(clf_report))
        print(f"Results written to {out_path}")

    return y_pred
