import torch
import torch.nn as nn
import torch.nn.functional as F
from tape.models.modeling_utils import SequenceToSequenceClassificationHead, ValuePredictionHead, SequenceClassificationHead, PairwiseContactPredictionHead, accuracy
from tape.datasets import pad_sequences
from CPCProt.model.base_config import CPCProtConfig
from CPCProt.model.cpcprot import CPCProtModel
from tape.registry import registry
import numpy as np
import json
from tape import ProteinConfig, ProteinModel

class DownstreamConfig(ProteinConfig):
    def __init__(self,
                vocab_size_or_config_json_file,
                CPC_model_path = '',
                CPC_args_path = '',
                hidden_size = -1,
                bidirectional = True,
                num_layers = -1,
                emb_method = '',
                freeze_CPC = True,
                emb_type = 'patched_cpc'):
        super().__init__()
        self.vocab_size_or_config_json_file = vocab_size_or_config_json_file
        self.CPC_model_path = CPC_model_path
        self.CPC_args_path = CPC_args_path
        self.hidden_size = hidden_size
        self.emb_method = emb_method
        self.freeze_CPC = freeze_CPC
        # LSTM specific
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.emb_type = emb_type

class CPCProtEmbedding(nn.Module):
    def __init__(self, model, emb_type: str = "patched_cpc"):
        super().__init__()
        self.parallel = isinstance(model, torch.nn.DataParallel)
        self.cpc = model
        self.emb_type = emb_type
        if self.parallel:
            self.device = torch.device(model.output_device)
        else:
            self.device = model.device
            
    def get_staggered_sequence_embs(self, data, out_type = 'z'):
        primary = data['primary'].to(self.device)
        with torch.no_grad():
            temp_z = self.cpc(data, return_early=out_type)
            z_dim = temp_z.shape[-1]
            num_patches = temp_z.shape[-2]
        max_len = int(data['protein_length'].max().item())
        # old_out = torch.zeros(primary.shape[0], max_len, z_dim, device = self.device)
        outs = []
        patch_len = self.cpc.module.cfg.patch_len if self.parallel else self.cpc.cfg.patch_len 
        
        for i in range(patch_len):
            temp_data = torch.zeros(primary.shape[0], max_len, device = self.device, dtype = torch.long)
            temp_data[:, :max_len - i] = primary[:, i:] # append zeros at the end
            outs.append(self.cpc({'primary': temp_data}, return_early = out_type))
            # last AAs for longest sequences will have zero embedding, since last patch is dropped
        out = torch.stack(outs, dim = 2).view(primary.shape[0], num_patches * patch_len, z_dim)
        out = torch.cat((out, torch.zeros(primary.shape[0], max_len - num_patches * patch_len, z_dim).to(out.device)), dim = 1)
        # assert(torch.all(torch.eq(out, old_out)))
        return out
            
    def get_z_patched_seq(self, data):
        # z for sequence-based classification tasks
        return self.get_staggered_sequence_embs(data, out_type = 'z')
    
    def get_c_patched_seq(self, data):        
        return self.get_staggered_sequence_embs(data, out_type = 'c')

    def get_z(self, data, return_mask=False):
        patch_len = self.cpc.module.cfg.patch_len if self.parallel else self.cpc.cfg.patch_len

        if isinstance(data, dict):
            x = data['primary'].to(self.device) # (N, max_L, H_enc)
            prot_len = data['protein_length'].to(self.device)
            num_patches = (prot_len / patch_len).floor()
            mask = torch.tensor(pad_sequences([np.ones(int(i)) for i in num_patches]))
        elif isinstance(data, torch.Tensor):
            x = data.to(self.device)
            # not masking anything out in this case
            mask = torch.ones((x.shape[0], x.shape[1] // patch_len))
        else:
            print("Input to CPCProt model must be a Torch tensor or the dictionary returned by training dataloaders.")
            raise

        z = self.cpc(x, return_early='z')
        mask = mask.to(dtype=torch.int, device=self.device)

        if return_mask:
            return (z, mask)
        else:
            return z

    def get_c(self, data, return_mask = False):
        z, mask = self.get_z(data, return_mask=True)
        if self.parallel:
            # workaround for accessing model attributes when DataParallel
            c = self.cpc(data, return_early='c')
        else:
            c = self.cpc.get_c(z)

        if return_mask:
            return (c, mask)
        else:
            return c

    def get_z_mean(self, data):
        z, mask = self.get_z(data, return_mask=True)
        mask = mask[:, :, None].expand_as(z)
        return (z*mask).sum(dim=1) / mask.sum(dim = 1)

    def get_c_final(self, data):
        c, mask = self.get_c(data, return_mask = True)
        L = mask.sum(dim=1)
        index = (L - 1)[:, None, None].expand(L.shape[0], 1, c.shape[2]).long().to(c.device)  # (N,1,C)
        return torch.gather(c, dim=1, index=index).squeeze()

    def get_c_mean(self, data):
        c, mask = self.get_c(data, return_mask = True)
        if self.emb_type == 'patched_cpc':
            mask = mask[:, :, None].expand_as(c)
        else:
            raise NotImplementedError
        return (c*mask).sum(dim=1) / mask.sum(dim=1)


###### For calculating downstream benchmarks only #####

def get_metrics(prediction, targets):
    loss = nn.CrossEntropyLoss(ignore_index=-1)(prediction.view(-1, prediction.size(2)), targets.view(-1))
    is_correct = prediction.float().argmax(-1) == targets
    is_valid_position = targets != -1

    # cast to float b/c otherwise torch does integer division
    num_correct = torch.sum(is_correct * is_valid_position).float()
    accuracy = num_correct / torch.sum(is_valid_position).float()
    metrics = {'acc': accuracy}
    return (loss, metrics)

# parent class to abstract away same constructor logic
class ProteinFinetuneModel(ProteinModel):
    config_class = DownstreamConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.cpc_args = CPCProtConfig()
        cpc_args_dict = json.load(open(config.CPC_args_path, 'r'))
        self.cpc_args.__dict__ = cpc_args_dict

        if self.config.emb_type == 'patched_cpc':
            self.cpc = CPCProtModel(self.cpc_args)
        else:
            raise

        self.input_size = self.cpc_args.enc_hidden_dim
        state_dict = dict(torch.load(config.CPC_model_path))
        for i in list(state_dict.keys()):
            if i.startswith('module.'):
                state_dict[i[7:]] = state_dict[i]
                del state_dict[i]
        self.cpc.load_state_dict(state_dict)
        self.embedder = CPCProtEmbedding(self.cpc, config.emb_type)
        self.emb_method_to_call = getattr(self.embedder, config.emb_method)
        if config.freeze_CPC:
            self.cpc.eval()
        else:
            self.cpc.train()

        # manually get embedding dim
        with torch.no_grad():
            fake_data = {'primary': torch.ones(4, 50).long(), 'protein_length': torch.tensor([20, 30, 40, 50]).float()}
            fake_embs = self.emb_method_to_call(fake_data)
            self.emb_dim = fake_embs.shape[-1]

    def forward(self, input_ids, input_mask = None, targets = None):
        data = {'primary': input_ids, 'input_mask':input_mask, 'protein_length': input_mask.sum(dim = 1).float()}
        if self.config.freeze_CPC:
            self.cpc.eval()
            with torch.no_grad():
                x = self.emb_method_to_call(data)
        else:
            self.cpc.train()
            x = self.emb_method_to_call(data)
        return self.classify(x, targets)

# TAPE standard
@registry.register_task_model('secondary_structure', 'CPCSeqToSeq')
class ConvSeqToSeq(ProteinFinetuneModel):
    config_class = DownstreamConfig
    def __init__(self, config):
        super().__init__(config)
        self.classify = SequenceToSequenceClassificationHead(
            self.emb_dim, config.num_labels, ignore_index=-1)

# TAPE standard
@registry.register_task_model('fluorescence', 'CPCValueClf')
@registry.register_task_model('stability', 'CPCValueClf')
class MLPValuePredictor(ProteinFinetuneModel):
    def __init__(self, config):
        super().__init__(config)
        self.classify = ValuePredictionHead(self.emb_dim)

# TAPE standard
@registry.register_task_model('remote_homology', 'CPCSeqClf')
class MLPSeqClf(ProteinFinetuneModel):
    def __init__(self, config):
        super().__init__(config)
        self.classify = SequenceClassificationHead(
            self.emb_dim, config.num_labels)