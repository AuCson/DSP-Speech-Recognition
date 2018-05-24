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
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_seqs, input_lens)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        return outputs, hidden


class RNN(nn.Module):
    """
    Vanilla RNN classifier
    Use max-pooling and avg-pooling as features
    """
    def __init__(self):
        super().__init__()
        self.enc = DynamicEncoder(39,100,1,0.0)
        self.out = nn.Linear(200,20)

    def forward(self, mfcc0, mfcc1, mfcc2, len0):
        """

        :param mfcc: [T,B,H]
        :return:
        """
        #len0_v = Variable(torch.FloatTensor(len0))
        enc_out, hidden = self.enc(torch.cat([mfcc0, mfcc1, mfcc2], dim=2), len0) # [T,B,H]
        #sum_enc_out = enc_out.sum(0)
        #avg_pool = sum_enc_out / len0_v.unsqueeze(1)
        #max_pool,_ = torch.max(enc_out,0)
        out = self.out(torch.cat([hidden[0],hidden[1]], dim=1))
        return out

class HRNN(nn.Module):
    """
    Hierarchical RNN classifier
    Level 1: 10ms stride
    Lelel 2: 100ms stride, that is 10 step
    """
    def __init__(self):
        super().__init__()
        self.hir = 10
        self.enc1 = DynamicEncoder(39, 50, n_layers=1, dropout=0.5, bidir=True)
        self.enc2 = DynamicEncoder(50, 50, n_layers=1, dropout=0.5, bidir=True)
        self.out = nn.Linear(150, 20)

    def forward(self, mfcc0, mfcc1, mfcc2, len0):
        len1 = [(_+self.hir-1) // self.hir for _ in len0]
        len0_v = Variable(torch.FloatTensor(len0))
        len1_v = Variable(torch.FloatTensor(len1))
        enc_out, hidden = self.enc1(torch.cat([mfcc0, mfcc1, mfcc2], dim=2), len0)

        feat = []
        for t in range(0, enc_out.size(0), self.hir):
            feat.append(enc_out[t])

        feat = torch.stack(feat) # [T2,B,H]
        enc_out2, hidden2 = self.enc2(feat, len1)

        out = self.out(torch.cat([hidden2[0], hidden2[1]], dim=1))
        return out

