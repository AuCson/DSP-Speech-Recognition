"""

https://github.com/HanqingLu/MultiscaleRNN.git

@article{DBLP:journals/corr/ChungAB16,
  author    = {Junyoung Chung and
               Sungjin Ahn and
               Yoshua Bengio},
  title     = {Hierarchical Multiscale Recurrent Neural Networks},
  journal   = {CoRR},
  volume    = {abs/1609.01704},
  year      = {2016}
}

"""
import torch
from torch.autograd import Function, Variable
import torch.nn.functional as Func
from torch import nn
from torch.nn import Module, Parameter
import math
import time
from layers import cuda_

def hard_sigm(a, x):
    temp = torch.div(torch.add(torch.mul(x, a), 1), 2.0)
    output = torch.clamp(temp, min=0, max=1)
    return output

class bound(Function):
    def forward(self, x):
        # forward : x -> output
        self.save_for_backward(x)
        output = x > 0.5
        return output.float()

    def backward(self, output_grad):
        # backward: output_grad -> x_grad
        x = self.saved_tensors
        x_grad = None

        if self.needs_input_grad[0]:
            x_grad = output_grad.clone()

        return x_grad

class HM_LSTMCell(Module):
    def __init__(self, bottom_size, hidden_size, top_size, a, last_layer):
        super(HM_LSTMCell, self).__init__()
        self.bottom_size = bottom_size
        self.hidden_size = hidden_size
        self.top_size = top_size
        self.a = a
        self.last_layer = last_layer
        '''
        U_11 means the state transition parameters from layer l (current layer) to layer l
        U_21 means the state transition parameters from layer l+1 (top layer) to layer l
        W_01 means the state transition parameters from layer l-1 (bottom layer) to layer l
        '''
        self.U_11 = Parameter(torch.FloatTensor(4 * self.hidden_size + 1, self.hidden_size))
        if not self.last_layer:
            self.U_21 = Parameter(torch.FloatTensor(4 * self.hidden_size + 1, self.top_size))
        self.W_01 = Parameter(torch.FloatTensor(4 * self.hidden_size + 1, self.bottom_size))
        self.bias = Parameter(torch.FloatTensor(4 * self.hidden_size + 1))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for par in self.parameters():
            par.data.uniform_(-stdv, stdv)

    def forward(self, c, h_bottom, h, h_top, z, z_bottom):
        # h_bottom.size = bottom_size * batch_size
        s_recur = torch.mm(self.W_01, h_bottom)
        if not self.last_layer:
            s_topdown_ = torch.mm(self.U_21, h_top)
            s_topdown = z.expand_as(s_topdown_) * s_topdown_
        else:
            s_topdown = cuda_(Variable(torch.zeros(s_recur.size()), requires_grad=False))
        s_bottomup_ = torch.mm(self.U_11, h)
        s_bottomup = z_bottom.expand_as(s_bottomup_) * s_bottomup_

        f_s = s_recur + s_topdown + s_bottomup + self.bias.unsqueeze(1).expand_as(s_recur)
        # f_s.size = (4 * hidden_size + 1) * batch_size
        f = Func.sigmoid(f_s[0:self.hidden_size, :])  # hidden_size * batch_size
        i = Func.sigmoid(f_s[self.hidden_size:self.hidden_size*2, :])
        o = Func.sigmoid(f_s[self.hidden_size*2:self.hidden_size*3, :])
        g = Func.tanh(f_s[self.hidden_size*3:self.hidden_size*4, :])
        z_hat = hard_sigm(self.a, f_s[self.hidden_size*4:self.hidden_size*4+1, :])

        one = cuda_(Variable(torch.ones(f.size()), requires_grad=False))
        z = z.expand_as(f)
        z_bottom = z_bottom.expand_as(f)

        c_new = z * (i * g) + (one - z) * (one - z_bottom) * c + (one - z) * z_bottom * (f * c + i * g)
        h_new = z * o * Func.tanh(c_new) + (one - z) * (one - z_bottom) * h + (one - z) * z_bottom * o * Func.tanh(c_new)

        # if z == 1: (FLUSH)
        #     c_new = i * g
        #     h_new = o * Func.tanh(c_new)
        # elif z_bottom == 0: (COPY)
        #     c_new = c
        #     h_new = h
        # else: (UPDATE)
        #     c_new = f * c + i * g
        #     h_new = o * Func.tanh(c_new)

        z_new = bound()(z_hat)

        return h_new, c_new, z_new


class HM_LSTM(Module):
    def __init__(self, a, input_size, size_list):
        super(HM_LSTM, self).__init__()
        self.a = a
        self.input_size = input_size
        self.size_list = size_list

        self.cell_1 = HM_LSTMCell(self.input_size, self.size_list[0], self.size_list[1], self.a, False)
        self.cell_2 = HM_LSTMCell(self.size_list[0], self.size_list[1], None, self.a, True)

    def forward(self, inputs, hidden):
        # inputs.size = (batch_size, time steps, embed_size/input_size)
        inputs = inputs.transpose(0,1)
        time_steps = inputs.size(1)
        batch_size = inputs.size(0)

        if hidden is None:
            h_t1 = cuda_(Variable(torch.zeros(self.size_list[0], batch_size).float(), requires_grad=False))
            c_t1 = cuda_(Variable(torch.zeros(self.size_list[0], batch_size).float(), requires_grad=False))
            z_t1 = cuda_(Variable(torch.zeros(1, batch_size).float(), requires_grad=False))
            h_t2 = cuda_(Variable(torch.zeros(self.size_list[1], batch_size).float(), requires_grad=False))
            c_t2 = cuda_(Variable(torch.zeros(self.size_list[1], batch_size).float(), requires_grad=False))
            z_t2 = cuda_(Variable(torch.zeros(1, batch_size).float(), requires_grad=False))
        else:
            (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2) = hidden
        z_one = cuda_(Variable(torch.ones(1, batch_size).float(), requires_grad=False))

        h_1 = []
        h_2 = []
        z_1 = []
        z_2 = []
        for t in range(time_steps):
            h_t1, c_t1, z_t1 = self.cell_1(c=c_t1, h_bottom=inputs[:, t, :].t(), h=h_t1, h_top=h_t2, z=z_t1, z_bottom=z_one)
            h_t2, c_t2, z_t2 = self.cell_2(c=c_t2, h_bottom=h_t1, h=h_t2, h_top=None, z=z_t2, z_bottom=z_t1)  # 0.01s used
            h_1 += [h_t1.t()]
            h_2 += [h_t2.t()]
            z_1 += [z_t1.t()]
            z_2 += [z_t2.t()]

        hidden = (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2)
        return torch.stack(h_1, dim=1), torch.stack(h_2, dim=1), torch.stack(z_1, dim=1), torch.stack(z_2, dim=1), hidden
