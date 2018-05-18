import torch
from torch import nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(13,100)
        self.out = nn.Linear(200,20)

    def forward(self, mfcc):
        """

        :param mfcc: [T,B,H]
        :return:
        """
        enc_out, hidden = self.gru(mfcc) # [T,B,H]
        avg_pool = enc_out.sum(0) / enc_out.size(0)
        max_pool,_ = torch.max(enc_out,0)
        out = self.out(torch.cat([avg_pool,max_pool],1))
        return out