import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from config import cfg
from transformer import TransformerEncoder
from collections import OrderedDict
from layers import *
from hmrnn import HM_LSTM

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

class HRNN(nn.Module):
    """
    Hierarchical RNN classifier
    Level 1: 10ms stride
    Lelel 2: 50ms stride, that is 5 step
    """
    def __init__(self, feat_size=39):
        super().__init__()
        self.hir = 5
        self.hidden_size = 200
        self.enc1 = DynamicEncoder(feat_size, 200, n_layers=2, dropout=0.2, bidir=True)
        self.enc2 = DynamicEncoder(200, 200, n_layers=1, dropout=0.0, bidir=True)
        self.out = nn.Linear(400, 20)

    def forward(self, inp, len0, return_feature=False):
        len1 = [(_+self.hir-1) // self.hir for _ in len0]
        len1 = np.array(len1)
        len0_v = cuda_(Variable(torch.from_numpy(len0).float()))
        len1_v = cuda_(Variable(torch.from_numpy(len1).float()))
        enc_out, hidden = self.enc1(inp, len0)

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
        feat = torch.cat([avg_pool, max_pool], dim=1)
        out = self.out(feat)
        out = F.dropout(out, 0.2)
        if not return_feature:
            return out
        else:
            return out, feat

class HRNN_Att(HRNN):
    """
    Hierarchical RNN classifier with attention
    Level 1: 10ms stride
    Lelel 2: 50ms stride, that is 5 step
    """
    def __init__(self,feat_size=39):
        super().__init__()
        self.hir = 5
        self.hidden_size = 200
        self.enc1 = DynamicEncoder(feat_size, 200, n_layers=2, dropout=0.2, bidir=True)
        self.enc2 = DynamicEncoder(200, 200, n_layers=1, dropout=0.0, bidir=True)
        self.attn = SelfAttn(200)
        self.out = nn.Linear(400, 20)

    def forward(self, inp, len0, return_feature=False):
        len1 = [(_+self.hir-1) // self.hir for _ in len0]
        len1 = np.array(len1)
        len0_v = cuda_(Variable(torch.from_numpy(len0).float()))
        len1_v = cuda_(Variable(torch.from_numpy(len1).float()))
        enc_out, hidden = self.enc1(inp, len0)

        enc_out = enc_out[:,:,-self.hidden_size:]
        #hidden =  hidden[-2:,:,:]

        feat = []
        for t in range(0, enc_out.size(0), self.hir):
            feat.append(enc_out[t])
        feat = torch.stack(feat) # [T2,B,H]
        enc_out2, hidden2 = self.enc2(feat, len1)

        sum_enc_out = enc_out2.sum(0)

        attn = self.attn(enc_out2)

        avg_pool = sum_enc_out / len1_v.unsqueeze(1)
        max_pool, _ = torch.max(enc_out2, 0)
        feat = torch.cat([attn, max_pool], dim=1)
        out = self.out(feat)
        out = F.dropout(out, 0.2)
        if not return_feature:
            return out
        else:
            return out, feat

class HMRNN(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.hir = 5
        self.hidden_size = 200
        self.enc1 = DynamicEncoder(feat_size, 200, n_layers=1, dropout=0.0, bidir=True)
        self.enc2 = HM_LSTM(1.0, 200, [200,200])
        self.out = nn.Linear(600, 20)

    def forward(self, inp, len0, return_feature=False):
        len1 = [(_+self.hir-1) // self.hir for _ in len0]
        len1 = np.array(len1)
        len0_v = cuda_(Variable(torch.from_numpy(len0).float()))
        len1_v = cuda_(Variable(torch.from_numpy(len1).float()))

        enc_out, hidden = self.enc1(inp, len0)
        enc_out = F.dropout(enc_out, 0.2)
        h_1, h_2, z_1, z_2, hidden = self.enc2(enc_out, None) # [B,T,H]
        hiddens = []
        for i in range(h_2.size(0)):
            hiddens.append(h_2[i,len0[i]-1])
        hiddens = torch.stack(hiddens) # [B,H]
        sum_enc_out = h_2.sum(0)
        avg_pool = sum_enc_out / len0_v.unsqueeze(1)
        max_pool, _ = torch.max(h_2, 0)
        feat = torch.cat([avg_pool, max_pool, hiddens], dim=1)
        out = self.out(feat)
        out = F.dropout(out, 0.2)
        if not return_feature:
            return out
        else:
            return out, feat

    def mask(self, len0, max_len):
        mask = np.zeros((len(len0), max_len)) # [B,T]
        for b in range(len(len0)):
            for t in range(len0[b]):
                mask[b][t] = 1.0
        mask = cuda_(Variable(torch.from_numpy(mask).float()))
        return mask

    def adjust_param(self):
        self.enc2.a += 0.5

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_enc = TransformerEncoder(39, 200, n_head=1, d_k=100, d_v=100)
        self.rnn_enc_1 = DynamicEncoder(78, 200, n_layers=1, dropout=0.2, bidir=True)
        self.rnn_enc_2 = DynamicEncoder(200, 200, n_layers=1, dropout=0.0, bidir=True)

        self.hir = 5
        self.out = nn.Linear(400,20)
        self.hidden_size = 200

    def forward(self, inp, len0, return_feature=False):
        len1 = [(_ + self.hir - 1) // self.hir for _ in len0]
        len1 = np.array(len1)
        len0_v = cuda_(Variable(torch.from_numpy(len0).float()))
        len1_v = cuda_(Variable(torch.from_numpy(len1).float()))
        attn_out,_ = self.attn_enc(inp)
        attn_out = F.dropout(attn_out, 0.5)
        enc_out, hidden = self.rnn_enc_1(torch.cat([inp, attn_out], 2), len0)
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
