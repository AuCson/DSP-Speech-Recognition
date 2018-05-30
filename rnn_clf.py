import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from config import cfg
from transformer import TransformerEncoder

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
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)
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


class RNN(nn.Module):
    """
    Vanilla RNN classifier
    Use max-pooling and avg-pooling as features
    """
    def __init__(self):
        super().__init__()
        self.enc = DynamicEncoder(39,50,1,0.0)
        self.out = nn.Linear(100,20)

    def forward(self, mfcc0, mfcc1, mfcc2, len0):
        """

        :param mfcc: [T,B,H]
        :return:
        """
        #len0_v = Variable(torch.FloatTensor(len0))
        enc_out, hidden = self.enc(torch.cat([mfcc0, mfcc1, mfcc2], dim=2), len0) # [T,B,H]
        sum_enc_out = enc_out.sum(0)
        #avg_pool = sum_enc_out / len0_v.unsqueeze(1)
        #max_pool,_ = torch.max(enc_out,0)
        out = self.out(torch.cat([hidden[0],hidden[1]], dim=1))
        return out

class CNNRNN(nn.Module):
    """
    CNN-RNN classifier
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,(9,9))
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(16 * 31, 50)
        self.enc = nn.GRU(50,50,bidirectional=True)
        self.out = nn.Linear(100,20)

    def forward(self, mfcc0, mfcc1, mfcc2, len0):
        """

        :param mfcc: [T,B,H]
        :return:
        """
        inp = torch.cat([mfcc0, mfcc1, mfcc2], dim=2) # [T,B,H]
        conv_inp = inp.transpose(0,1).unsqueeze(1) # [B,1,T,H]
        conv = self.conv1(conv_inp).transpose(0,2).transpose(1,2) # [B,L,T,H]
        conv = conv.contiguous()
        conv = conv.view(conv.size(0),conv.size(1),-1)
        inp_rnn = self.fc(conv)
        enc_out, hidden = self.enc(inp_rnn) # [T,B,H]
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
        self.hir = 5
        self.hidden_size = 200
        self.enc1 = DynamicEncoder(39, 200, n_layers=2, dropout=0.2, bidir=True)
        self.enc2 = DynamicEncoder(200, 200, n_layers=1, dropout=0.0, bidir=True)
        self.out = nn.Linear(400, 20)

    def forward(self, mfcc0, mfcc1, mfcc2, len0):
        len1 = [(_+self.hir-1) // self.hir for _ in len0]
        len1 = np.array(len1)
        len0_v = cuda_(Variable(torch.from_numpy(len0).float()))
        len1_v = cuda_(Variable(torch.from_numpy(len1).float()))
        enc_out, hidden = self.enc1(torch.cat([mfcc0, mfcc1, mfcc2], dim=2), len0)

        enc_out = enc_out[:,:,-self.hidden_size:]
        #hidden =  hidden[-2:,:,:]

        feat = []
        for t in range(0, enc_out.size(0), self.hir):
            feat.append(enc_out[t])
        feat = torch.stack(feat) # [T2,B,H]
        enc_out2, hidden2 = self.enc2(feat, len1)

        sum_enc_out = enc_out2.sum(0)
        avg_pool = sum_enc_out / len1_v.unsqueeze(1)
        max_pool, _ = torch.max(enc_out2, 0)
        out = self.out(torch.cat([avg_pool, max_pool], dim=1))
        out = F.dropout(out, 0.2)
        return out

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_enc = TransformerEncoder(39, 100, n_head=1, d_k=100, d_v=100)
        self.rnn_enc_1 = DynamicEncoder(78, 200, n_layers=1, dropout=0.0, bidir=True)
        self.rnn_enc_2 = DynamicEncoder(200, 200, n_layers=1, dropout=0.0, bidir=True)

        self.hir = 5
        self.out = nn.Linear(400,20)
        self.hidden_size = 200

    def forward(self, mfcc0, mfcc1, mfcc2, len0, return_feature=False):
        len1 = [(_ + self.hir - 1) // self.hir for _ in len0]
        len1 = np.array(len1)
        len0_v = cuda_(Variable(torch.from_numpy(len0).float()))
        len1_v = cuda_(Variable(torch.from_numpy(len1).float()))
        inp = torch.cat([mfcc0, mfcc1, mfcc2],dim=2)
        attn_out = F.dropout(self.attn_enc(inp),0.5)
        rnn_inp = torch.cat([inp, attn_out], dim=2)
        enc_out, hidden = self.rnn_enc_1(rnn_inp, len0)
        enc_out = enc_out[:, :, -self.hidden_size:]

        feat = []
        for t in range(0, enc_out.size(0), self.hir):
            feat.append(enc_out[t])
        feat = torch.stack(feat)

        enc_out2, hidden2 = self.rnn_enc_2(feat, len1)

        sum_enc_out = enc_out2.sum(0)
        avg_pool = sum_enc_out / len1_v.unsqueeze(1)
        max_pool,_ = torch.max(enc_out2,0)
        feat = torch.cat([avg_pool, max_pool], dim=1)
        out = self.out(feat)
        out = F.dropout(out, 0.2)
        if not return_feature:
            return out
        else:
            return out, feat

class CNNClassifier(nn.Module):
    pass