import torch
import torch.nn as nn
from CPCProt.model.base_config import CPCProtConfig, CPCAbstractModel
import math

class GRUAutoregressor(CPCAbstractModel):
    def __init__(self, config, in_hidden_dim, gru_hidden_dim):
        super().__init__(config)
        # config is not used, just for compatibility
        self.config = config
        self._gru_hidden_dim = gru_hidden_dim

        # input dim to GRU is hidden dim of encoder, which depends on which is used.
        # for dot product critics, in_hidden_dim should equal gru_hidden_dim.
        self.gru = nn.GRU(input_size=in_hidden_dim,
                          hidden_size=gru_hidden_dim,
                          batch_first=True)

    def forward(self, input_seq):
        device = input_seq.device
        regress_hidden_state = torch.zeros(1,
                                           input_seq.size(0),
                                           self._gru_hidden_dim,
                                           device=device)

        self.gru.flatten_parameters()
        output, regress_hidden_state = self.gru(input_seq, regress_hidden_state)
        return output


class LSTMAutoregressor(CPCAbstractModel):
    def __init__(self, cfg, in_hidden_dim, lstm_hidden_dim):
        super().__init__(cfg)
        # config is not used, just done for compatibility
        self.cfg = cfg
        self._lstm_hidden_dim = lstm_hidden_dim

        # input dim to LSTM is hidden dim of encoder, which depends on which is used.
        # for dot product critics, in_hidden_dim should equal gru_hidden_dim.
        self.lstm = nn.LSTM(input_size=in_hidden_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=cfg.lstm_num_layers,
                            batch_first=True,
                            bidirectional=False)

    def forward(self, input_seq):
        device = input_seq.device
        h0 = torch.randn(self.cfg.lstm_num_layers, input_seq.size(0), self._lstm_hidden_dim, device=device)
        c0 = torch.randn(self.cfg.lstm_num_layers, input_seq.size(0), self._lstm_hidden_dim, device=device)
        self.lstm.flatten_parameters()
        output, (hn, cn) = self.lstm(input_seq, (h0, c0))
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(CPCAbstractModel):
    def __init__(self, cfg, hidden_dim):
        super().__init__(cfg)
        self.cfg = cfg
        self.hidden_dim = hidden_dim
        self.device = torch.device('cuda') if cfg.use_cuda else torch.device('cpu')

        self.pos_encoder = PositionalEncoding(hidden_dim, device=self.device)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                   nhead=cfg.transformer_nhead,
                                                   dim_feedforward=cfg.transformer_dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=cfg.transformer_num_layers)
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).to(self.device)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        return output