from rnn_clf import Transformer
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
        return x

    def backward(self, grad):
        return -grad * self.gamma


class AdvTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = Transformer()
        self.rev = GradientReverse()
        self.d = nn.Linear(400, 35)

    def forward(self, mfcc0, mfcc1, mfcc2, len0):
        out1, feat = self.g(mfcc0, mfcc1, mfcc2, len0, return_feature=True)
        feat = self.rev(feat)
        out2 = self.d(feat)
        return out1, out2