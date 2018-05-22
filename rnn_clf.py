import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class RNN(nn.Module):
    """
    Vanilla RNN classifier
    Use max-pooling and avg-pooling as features
    """
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(39,100)
        self.out = nn.Linear(200,20)

    def forward(self, mfcc0, mfcc1, mfcc2, len0):
        """

        :param mfcc: [T,B,H]
        :return:
        """
        len0_v = Variable(torch.FloatTensor(len0))
        enc_out, hidden = self.gru(torch.cat([mfcc0, mfcc1, mfcc2], dim=2)) # [T,B,H]
        sum_enc_out = enc_out.sum(0)
        avg_pool = sum_enc_out / len0_v.unsqueeze(1)
        max_pool,_ = torch.max(enc_out,0)
        out = self.out(torch.cat([avg_pool,max_pool],1))
        return out

class HRNN(nn.Module):
    """
    Hierarchical RNN classifier
    Level 1: 10ms stride
    Lelel 2: 100ms stride
    """
    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(39, 50)
        self.gru2 = nn.GRU(50, 50)
        self.out = nn.Linear(150, 20)



