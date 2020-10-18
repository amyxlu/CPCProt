import json
import torch
import torch.nn as nn
from tape.models.modeling_resnet import ProteinResNetConfig, ProteinResNetModel
from tape import ProteinBertModel
from CPCProt.model.base_config import CPCAbstractModel, CPCProtConfig
from CPCProt.model.encoder import PatchedConvEncoder, PatchedConvEncoder2
from CPCProt.model.autoregressor import GRUAutoregressor, LSTMAutoregressor
from CPCProt.model.critics import batch_dot_product
from CPCProt.model.losses import calc_nce


# The code builds upon the TAPE package to make use of downstream training
# mechanisms. However, TAPE abstract models needs a TAPE config class to be specified.
# A default config is specified here if user is only using this module to
# obtain embeddings using the configurations (most parameter-efficient variant)
# in our paper.
DEFAULT_CONFIG = CPCProtConfig(
    use_cuda = torch.cuda.is_available(),
    enc_hidden_dim = 512,
    encoder_type = "patched_conv",
    autoregressor_type = "gru",
    encoder_relu = True,
    critic_type = "dot_product",
    K = 4,
    patch_len = 11,
    norm_type = "channel_norm",
    vocab_size = 30
)

class CPCProtModel(CPCAbstractModel):
    def __init__(self, cfg=None):
        if not cfg:
            cfg = DEFAULT_CONFIG

        super().__init__(cfg)
        self.cfg = cfg
        self.device = torch.device('cuda') if cfg.use_cuda else torch.device('cpu')

        # get encoder:
        if self.cfg.encoder_type == "patched_conv":
            self.enc = PatchedConvEncoder(self.cfg)
            self._enc_hidden_dim = cfg.enc_hidden_dim
        elif self.cfg.encoder_type == "patched_conv_large":
            self.enc = PatchedConvEncoder2(self.cfg)
            self._enc_hidden_dim = cfg.enc_hidden_dim
        elif self.cfg.encoder_type == "bert":
            # use pretrained weights
            self.enc = ProteinBertModel.from_pretrained("bert-base")
            self._enc_hidden_dim = 768
        elif self.cfg.encoder_type == "resnet":
            # kaiming initialized weights
            resnet_cfg = ProteinResNetConfig()          # use defaults
            self.enc = ProteinResNetModel(resnet_cfg)   # default: 512
            self._enc_hidden_dim = resnet_cfg.hidden_size
        else:
            self._enc_hidden_dim = None
            raise NotImplementedError
        self.enc.to(self.device)

        # Get autoregressor
        # for dot product critic, z and c has same hidden dimensions
        if self.cfg.autoregressor_type == "gru":
            self.autoregressor = GRUAutoregressor(cfg, self._enc_hidden_dim, self._enc_hidden_dim).to(self.device)
        elif self.cfg.autoregressor_type == "lstm":
            self.autoregressor = LSTMAutoregressor(cfg, self._enc_hidden_dim, self._enc_hidden_dim).to(self.device)
        else:
            raise NotImplementedError

        if cfg.critic_type == "bilinear":
            raise NotImplementedError
        if cfg.critic_type == "dot_product":
            # parameterless, but make a list for each k just for compatibility with using per-position critics
            # a la original CPC paper
            self.critics = [batch_dot_product] * cfg.K
        else:
            raise NotImplementedError
    
    def get_z(self, x):
        # torch.arange does not include end point
        patch_ends = torch.arange(self.cfg.patch_len, x.shape[1]+1, self.cfg.patch_len, device=self.device)
        max_L = len(patch_ends)  # the true number of patches
        z = torch.zeros((x.shape[0], max_L, self._enc_hidden_dim), device=self.device)

        # loop through the patches and obtain sequence of z
        for t in range(max_L):
            if t >= max_L:
                # discard the last patch that's < 11 AAs
                break
            else:
                end = patch_ends[t]
            patch = x[:, end-self.cfg.patch_len:end]
            embedded_patch, _ = self.enc(patch)   # (N, max_L, H)

            # instead of using the embedded [CLS] token, use a 'global average pool'
            z_t = embedded_patch.mean(dim=1)   # (N, H)
            z[:, t, :] = z_t
        return z

    def get_c(self, z):
        max_L = z.shape[1]
        c = self.autoregressor(z[:,:max_L,:]).to(self.device)
        return c

    def forward(self, data, _run=None, global_itr=None, str_code="", return_early = False):
        if isinstance(data, dict):
            x = data['primary'].to(self.device) # (N, max_L, H_enc)
            prot_lens = data['protein_length'].to(self.device)
        elif isinstance(data, torch.Tensor):
            x = data.to(self.device)
            prot_len = torch.tensor(x.shape[1]).to(self.device)
        else:
            print("Input to CPCProt model must be a Torch tensor or the dictionary returned by training dataloaders.")
            raise
        z = self.get_z(x)

        # return tensors without calculating NCE
        # this is a workaround to avoid accessing attributes when DataParallel-ing this model
        if return_early == 'z':
            return z
        
        c = self.get_c(z)
        if return_early == 'c':
            return c

        # Mask out padded regions from the loss calculation
        N = z.shape[0]
        max_L = z.shape[1]
        idxs = torch.arange(max_L)[None, :].expand(N, max_L).to(self.device)
        prot_lens = (prot_lens / self.cfg.patch_len).floor().long()
        prot_lens = prot_lens[:, None].expand((N, max_L))
        mask = ((idxs < prot_lens - self.cfg.K) & (idxs >= self.cfg.min_t)).float()  # N x max_L

        nce, acc = calc_nce(self.cfg, z, c, mask, self.critics, self.device, _run, global_itr, str_code)
        return nce, acc
