import torch
import torch.nn as nn
import torch.nn.functional as F
from CPCProt.model.base_config import CPCProtConfig, CPCAbstractModel
from CPCProt.model.autoregressor import PositionalEncoding

class IDModule(nn.Module):
    '''https://github.com/facebookresearch/CPC_audio/blob/master/cpc/model.py
    '''
    def __init__(self, *args, **kwargs):
        super(IDModule, self).__init__()

    def forward(self, x):
        return x


class ChannelNorm(nn.Module):
    '''https://github.com/facebookresearch/CPC_audio/blob/master/cpc/model.py
    '''
    def __init__(self,
                 numFeatures,
                 epsilon=1e-05,
                 affine=True):

        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1,
                                                              numFeatures, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        cumMean = x.mean(dim=1, keepdim=True)
        cumVar = x.var(dim=1, keepdim=True)
        x = (x - cumMean)*torch.rsqrt(cumVar + self.epsilon)

        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


class PatchedConvEncoder(CPCAbstractModel):
    """ Simple convolutional neural network encoder for the patched model
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.encoder_relu = True if cfg.encoder_relu else False
        if cfg.norm_type == "instance_norm":
            def normLayer(x): return nn.InstanceNorm1d(x, affine=True)
        elif cfg.norm_type == "identity":
            normLayer = IDModule
        elif cfg.norm_type == "channel_norm":
            normLayer = ChannelNorm
        else:
            normLayer = IDModule

        self.embedding = nn.Embedding(cfg.vocab_size, 32)
        self.conv1 = nn.Conv1d(32, 64, 4)
        self.norm1 = normLayer(64)
        self.conv2 = nn.Conv1d(64, 64, 6)
        self.norm2 = normLayer(64)
        self.conv3 = nn.Conv1d(64, cfg.enc_hidden_dim, 3)
        self.norm3 = normLayer(cfg.enc_hidden_dim)

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # from LongTensor of indices to embedding tensor
        x = x.permute(0, 2, 1)  # Conv layers are NCL
        if self.encoder_relu:
            x = F.relu(self.norm1(self.conv1(x)))
            x = F.relu(self.norm2(self.conv2(x)))
            x = F.relu(self.norm3(self.conv3(x)))
        else:
            x = self.norm1(self.conv1(x))
            x = self.norm2(self.conv2(x))
            x = self.norm3(self.conv3(x))

        x = x.permute(0, 2, 1).contiguous()  # back to NLC
        avg = x.mean(dim=1)  # return a "pool" embedding for consistency with TAPE models
        return x, avg


class PatchedConvEncoder2(CPCAbstractModel):
    """ A larger convolutional neural network encoder.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.encoder_relu = True if cfg.encoder_relu else False
        if cfg.norm_type == "instance_norm":
            def normLayer(x):
                return nn.InstanceNorm1d(x, affine=True)
        elif cfg.norm_type == "identity":
            normLayer = IDModule
        elif cfg.norm_type == "channel_norm":
            normLayer = ChannelNorm
        else:
            normLayer = IDModule

        self.embedding = nn.Embedding(cfg.vocab_size, 64)
        self.conv1 = nn.Conv1d(64, 128, 4)
        self.norm1 = normLayer(128)
        self.conv2 = nn.Conv1d(128, 256, 4)
        self.norm2 = normLayer(256)
        self.conv3 = nn.Conv1d(256, 512, 3)
        self.norm3 = normLayer(512)
        self.conv4 = nn.Conv1d(512, cfg.enc_hidden_dim, 3)
        self.norm4 = normLayer(cfg.enc_hidden_dim)

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # from LongTensor of indices to embedding tensor
        x = x.permute(0, 2, 1)  # Conv layers are NCL
        if self.encoder_relu:
            x = F.relu(self.norm1(self.conv1(x)))
            x = F.relu(self.norm2(self.conv2(x)))
            x = F.relu(self.norm3(self.conv3(x)))
            x = F.relu(self.norm4(self.conv4(x)))
        else:
            x = self.norm1(self.conv1(x))
            x = self.norm2(self.conv2(x))
            x = self.norm3(self.conv3(x))
            x = self.norm4(self.conv4(x))

        x = x.permute(0, 2, 1).contiguous()  # back to NLC
        avg = x.mean(dim=1)  # return an average pool embedding
        return x, avg


class StridedConvEncoder(CPCAbstractModel):

    def __init__(self, cfg: CPCProtConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.enc_hidden_dim)

        layers = []
        for _ in range(cfg.enc_num_layers):
            layers.append(
                nn.Conv1d(cfg.enc_hidden_dim,
                          cfg.enc_hidden_dim,
                          cfg.enc_filter_len,
                          padding=cfg.enc_filter_len // 2,
                          )
            )
            layers.append(
                ChannelNorm(cfg.enc_hidden_dim)
            )
            layers.append(
                nn.ReLU()
            )
        self.layers = nn.Sequential(*layers)


    def forward(self, input_ids):
        # Embed the input_ids
        embed = self.embedding(input_ids)
        embed = embed.permute(0, 2, 1)  # Conv layers are NCW
        embed = self.layers(embed)

        # Re-permute the sequence embedding to be NWC
        embed = embed.permute(0, 2, 1).contiguous()
        avg = embed.mean(dim=1)  # return a "pool" embedding for consistency with TAPE models
        return embed, avg


class TransformerEncoder(CPCAbstractModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.device = torch.device('cuda') if cfg.use_cuda else torch.device('cpu')
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.enc_hidden_dim)

        self.pos_encoder = PositionalEncoding(cfg.enc_hidden_dim, device=self.device)
        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.enc_hidden_dim,
                                                   nhead=cfg.transformer_nhead,
                                                   dim_feedforward=cfg.transformer_dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=cfg.transformer_num_layers)
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).to(self.device)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input_ids):
        src = self.embedding(input_ids)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        avg = output.mean(dim=1)  # return a "pool" embedding for consistency with TAPE models
        return output, avg
