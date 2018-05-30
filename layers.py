import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from config import cfg

def cuda_(var):
    return var.cuda() if cfg.cuda else var

def orth_gru(gru):
    gru.reset_parameters()
    for _, hh, _, _ in gru.all_weights:
        for i in range(0, hh.size(0), gru.hidden_size):
            torch.nn.init.orthogonal(hh[i:i+gru.hidden_size],gain=1)
    return gru

class SPPLayer(nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size() # [B,C,T,H]
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = (h // (2 ** i),1)
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x

class DynamicEncoder(nn.Module):
    """
    Encoder: No need for input lens to be sorted
    """
    def __init__(self, input_size, hidden_size, n_layers, dropout, bidir=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=bidir)
        self.gru = orth_gru(self.gru)

    def forward(self, input_seqs, input_lens, hidden=None):
        """
        forward procedure. No need for inputs to be sorted
        :param input_seqs: Variable of [T,B]
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        input_seqs = input_seqs.transpose(0, 1)  # [B,T,D]
        sort_idx = np.argsort(-input_lens)
        unsort_idx = cuda_(torch.LongTensor(np.argsort(sort_idx)))
        input_lens = input_lens[sort_idx]
        sort_idx = cuda_(torch.LongTensor(sort_idx))
        input_seqs = input_seqs[sort_idx].transpose(0, 1)  # [T,B,E]
        #input_lens = cuda_(torch.from_numpy(input_lens).long())
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_seqs, input_lens)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        return outputs, hidden

class SelfAttn(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs):
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]

        # Create variable to store attention energies
        attn_energies = self.score(encoder_outputs).unsqueeze(1) #[B,1,T]
        return torch.bmm(attn_energies, encoder_outputs).squeeze(1) #[B,H]

    def score(self, encoder_outputs):
        energy = self.attn(encoder_outputs) # [B,T,H]
        energy = self.v(F.tanh(energy))  # [B,T,1]
        return F.softmax(energy.squeeze(2),1)  # [B*T]

class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out