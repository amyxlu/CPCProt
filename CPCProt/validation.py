from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sklearn.metrics as skm
from CPCProt.model.heads import CPCProtEmbedding
from CPCProt.utils import train_val_split, one_hot_encode

SEED = 42
np.random.seed(SEED)

def _get_TSNE_emb(x, n_components=2):
    tsne = TSNE(n_components=n_components)
    tsne_emb = tsne.fit_transform(x)
    return tsne_emb


def _plot_TSNE(X, y, figure_savedir="./", embed_method="", _run=None, epoch=""):
    tsne_emb = _get_TSNE_emb(X)
    test_classes, counts = np.unique(y, return_counts=True)
    num_labels = len(test_classes)
    # _run.log_scalar("val_num_samples", len(y), epoch)
    # print("Test classes: ", test_classes)
    print("TSNE Samples per test class: ", counts)
    print("TSNE Number of test classes: ", num_labels)
    d = dict(zip(np.unique(y), sns.color_palette("husl", num_labels)))
    row_colors = [d[fam] for fam in y]

    tsne_emb_df = pd.DataFrame({
        "x": tsne_emb[:, 0],
        "y": tsne_emb[:, 1],
        "family": row_colors,  # workaround for sns bug https://github.com/mwaskom/seaborn/issues/1515
    })

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(x="x", y="y", hue="family", data=tsne_emb_df, ax=ax)
    ax.legend_.remove()
    figure_fpath = Path(figure_savedir) / f"tsne_{embed_method}_epoch{epoch}.png"
    plt.show()
    plt.savefig(figure_fpath)
    plt.close()
    print("Saved TSNE figure to ", figure_fpath)
    if _run:
        _run.add_artifact(figure_fpath)


def _k_nearest_neighbor(X_train, y_train, X_val, y_val, k=1, embed_method="", _run=None, epoch=None):
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    test_classes, test_counts = np.unique(y_val, return_counts=True)
    
    print("KNN train classes", train_classes)
    print("KNN samples per train class", train_counts)
    print("KNN test classes", test_classes)
    print("KNN samples per test class", test_counts)

    clf = KNeighborsClassifier(n_neighbors=k, p=2).fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    acc = (y_pred == y_val).astype(int).sum() / len(y_pred)
    if _run:
        _run.log_scalar(f"val_{embed_method}_{k}nn_acc", float(acc), epoch)
    else:
        print("*** KNN acc: ", acc, "***")
    return acc

    # y_pred, y_val = one_hot_encode(y_pred, num_val_classes), one_hot_encode(y_val, num_val_classes)
    # auroc = skm.roc_auc_score(y_val, y_pred, average='macro', multi_class='ovr')
    # auprc = skm.average_precision_score(y_val, y_pred, average='macro')
    # precision = skm.precision_score(y_val, y_pred, average='micro')
    # recall = skm.recall_score(y_val, y_pred, average='micro')

    # _run.log_scalar(f"val_{k}nn_auroc", auroc, epoch)
    # _run.log_scalar(f"val_{k}nn_auprc", auprc, epoch)
    # _run.log_scalar(f"val_{k}nn_micro_precision", precision, epoch)
    # _run.log_scalar(f"val_{k}nn_micro_recall", recall, epoch)


#  def _logistic_regression( X_train, y_train, X_val, y_val, _run=None, epoch=None):
#      ''' Multiclass cross-entropy LR.
#      '''
#      try:
#          clf = LogisticRegression(multi_class='multinomial',
#                                   solver='sag',
#                                   max_iter=10000,
#                                   random_state=42).fit(X_train, y_train)
#          y_pred = clf.predict(X_val)
#          acc = (y_pred == y_val).astype(int).sum() / len(y_pred)
#          _run.log_scalar("val_lr_acc", acc, epoch)
#      except:
#          print("Error in logistic regression validation, but ignoring it to continue training...")
#          pass


class MetricsAccumulator:
    def __init__(self):
        self.loss_ticker = 0
        self.acc_ticker = 0
        self.total_loss = 0
        self.total_acc = 0

    def accumulate_loss(self, loss):
        self.total_loss += loss
        self.loss_ticker += 1

    def accumulate_acc(self, acc):
        self.total_acc += acc
        self.loss_ticker += 1

    def get_loss(self):
        '''Get average loss between now and the last time that this
        method is called -- this resets the loss accumuators.'''
        try:
            avg_loss = self.total_loss / self.loss_ticker
        except ZeroDivisionError:
            avg_loss = np.nan
        self.total_loss = 0
        self.loss_ticker = 0
        return avg_loss

    def get_acc(self):
        '''Get average accuracy between now and the last time that this
        method is called -- this resets the accuracy accumuators.'''
        try:
            avg_acc = self.total_acc / self.acc_ticker
        except ZeroDivisionError:
            avg_acc = np.nan
        self.total_acc = 0
        self.acc_ticker = 0
        return avg_acc


class Validation:
    ''' Validation class for the PatchedCPC and StridedCPC models.
    '''
    def __init__(self, model, loader, parallel=True, max_val_batches=None,
                 embed_method='c_final', figure_savedir="./", 
                 clf_train_frac=0.7, emb_type="patched_cpc"):
        self.loss_ticker = 0
        self.acc_ticker = 0
        self.total_loss = 0
        self.total_acc = 0

        self.model = model.eval()
        self.loader = loader
        self.max_val_batches = max_val_batches
        self.parallel = parallel

        self.figure_savedir = figure_savedir
        self.clf_train_frac = clf_train_frac

        # embedder head is right now only used for "patched_cpc" and "strided_cpc"
        # for the augment model, just pull the embedding directly from the model, since no length dim.
        self.embedder = CPCProtEmbedding(self.model, emb_type)
        self.embed_method = embed_method

        if not emb_type in ['patched_cpc', 'strided_cpc']:
            print(f"Validation error: {emb_type} is not a valid `emb_type`")
        self.emb_type = emb_type

    def nce_validate(self, _run=None, epoch=None):
        ''' Wrapper for calculating NCE loss/acc (load pairs)
        '''
        with torch.no_grad():
            for i, data in enumerate(self.loader):
                if self.max_val_batches:
                    if i >= self.max_val_batches:
                        break
                self._nce_val_batch(data, _run, epoch)

        # reset loss/acc counters and log
        loss = self.total_loss / self.loss_ticker
        acc = self.total_acc / self.acc_ticker
        if _run:
            _run.log_scalar(f"val_nce_loss", float(loss), epoch)
            _run.log_scalar(f"val_nce_acc", float(acc), epoch)
        else:
            print("loss", loss)
            print("acc", acc)
        self.total_acc, self.total_loss, self.loss_ticker, self.acc_ticker = 0, 0, 0, 0
        return loss, acc   # for deciding whether or not to save model.

    def embedding_validate(self, _run=None, epoch=None):
        '''
        Wrapper method for all of the embedding-related validations:
        * retrieve embeddings (load single sequence)
        * plot embeddings using TSNE
        * get a 1NN classification accuracy.
        '''
        all_embs = []
        all_fams = []

        with torch.no_grad():
            for i, data in enumerate(self.loader):
                if self.max_val_batches:
                    if i >= self.max_val_batches:
                        break

                print(f"epoch {epoch}: getting validation {self.emb_type} {self.embed_method} embeddings for batch {i}...")
                emb = self._get_embeddings_batch(data)
                all_embs.append(emb)
                all_fams.append(data['family'].detach().cpu().numpy())

        # plot and save TSNE
        all_embs = np.concatenate(all_embs, axis=0)
        all_fams = np.concatenate(all_fams, axis=0)
        _plot_TSNE(all_embs, all_fams,
                   figure_savedir=self.figure_savedir, embed_method=self.embed_method,
                   _run=_run, epoch=epoch)

        # split data, train kNN, and log performance
        X_train, y_train, X_val, y_val = train_val_split(all_embs, all_fams, self.clf_train_frac)
        print("printing `y_val` to check if split is consistent:")
        print(y_val[:10])
        knn_acc = _k_nearest_neighbor(X_train, y_train, X_val, y_val, k=1, embed_method=self.embed_method,
                                      _run=_run, epoch=epoch)
        return knn_acc

    def _nce_val_batch(self, data, _run=None, epoch=None):
        loss, acc = self.model(data, _run, epoch, str_code="val")
        if self.parallel:
            loss = loss.mean()  # average across all GPUs
            acc = acc.mean()

        self.total_loss += loss.item()
        self.loss_ticker += 1
        self.total_acc += acc.item()
        self.acc_ticker += 1

    def _get_embeddings_batch(self, data):
        if self.embed_method == "z_mean":
            emb = self.embedder.get_z_mean(data).detach().cpu().numpy()
        elif self.embed_method == "c_mean":
            emb = self.embedder.get_c_mean(data).detach().cpu().numpy()
        elif self.embed_method == "c_final":
            emb = self.embedder.get_c_final(data).detach().cpu().numpy()
        else:
            raise ValueError(f"Validation error: can't obtain embeddings for {self.embed_method} and {self.emb_type}.")
        return emb
