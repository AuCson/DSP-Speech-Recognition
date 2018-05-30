from rnn_clf import HRNN
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np
from rnn_clf import cuda_

class GradientReverse(torch.autograd.Function):
    """
    Identical mapping from input to output
    but reverse the gradient during backwards
    """
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad):
        return -grad * self.gamma


class AdvHRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = HRNN()
        self.rev = GradientReverse(0.01)
        self.d1 = nn.Linear(400, 200)
        self.d2 = nn.Linear(200, 35)

    def forward(self, mfcc0, mfcc1, mfcc2, len0):
        out_lb, feat = self.g(mfcc0, mfcc1, mfcc2, len0, return_feature=True)
        feat = self.rev(feat)
        out1 = F.tanh(self.d1(feat))
        out2 = self.d2(out1)
        return out_lb, out2