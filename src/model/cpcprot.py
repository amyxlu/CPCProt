import torch
import torch.nn as nn
from tape.models.modeling_resnet import ProteinResNetConfig, ProteinResNetModel
from tape import ProteinBertModel
from model.base_config import CPCAbstractModel, CPCConfig
from model.encoder import PatchedConvEncoder, PatchedConvEncoderLarge
from model.autoregressor import GRUAutoregressor, LSTMAutoregressor
from model.critics import batch_dot_product
from model.losses import calc_nce

class CPCProtModel(CPCAbstractModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.device = torch.device('cuda') if cfg.use_cuda else torch.device('cpu')

        # get encoder:
        if self.cfg.encoder_type == "patched_conv":
            self.enc = PatchedConvEncoder(self.cfg)
            self._enc_hidden_dim = cfg.enc_hidden_dim
        elif self.cfg.encoder_type == "patched_conv_large":
            self.enc = PatchedConvEncoderLarge(self.cfg)
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
        x = data['primary'].to(self.device) # (N, max_L, H_enc)
        z = self.get_z(x)        

        # return tensors without calculating NCE
        if return_early == 'z':
            return z
        
        c = self.get_c(z)
        if return_early == 'c':
            return c

        # Mask out padded regions from the loss calculation
        N = z.shape[0]
        max_L = z.shape[1]
        idxs = torch.arange(max_L)[None, :].expand(N, max_L).to(self.device)
        prot_lens = data['protein_length'].to(self.device)
        prot_lens = (prot_lens / self.cfg.patch_len).floor().long()
        prot_lens = prot_lens[:, None].expand((N, max_L))
        mask = ((idxs < prot_lens - self.cfg.K) & (idxs >= self.cfg.min_t)).float()  # N x max_L

        nce, acc = calc_nce(self.cfg, z, c, mask, self.critics, self.device, _run, global_itr, str_code)
        return nce, acc
