import torch
import torch.nn as nn
from model.base_config import CPCConfig, CPCAbstractModel

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
