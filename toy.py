import numpy as np
from python_speech_features import mfcc, delta
from config import *
from reader import Reader
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import robust_scale
from sklearn.metrics import accuracy_score
import pickle
from rnn_clf import RNN
import torch
from torch.autograd import Variable

class _ModelBase:
    def __init__(self):
        self.reader = Reader()




class RNNModel(_ModelBase):
    def __init__(self):
        super().__init__()
        self.clf = RNN()

    def pad(self, feat):
        if feat.shape[0] < 400:
            feat = np.pad(feat, ((0, 400 - feat.shape[0]), (0, 0)), 'constant')
        else:
            feat = feat[:400]
        return feat

    def train(self):
        train_data = self.reader.mini_batch_iterator(self.reader.train)
        optim = torch.optim.Adam(self.clf.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        for itr, total_iter, (feat, label) in train_data:
            optim.zero_grad()
            mfcc_feat = [self.pad(mfcc(robust_scale(sig), rate, nfft=512, winlen=0.01)) for sig, rate in feat]
            d_mfcc_feat = np.stack([delta(_, 3) for _ in mfcc_feat]) # [B,T,H]
            d_mfcc_feat = d_mfcc_feat.transpose((1,0,2))
            input = Variable(torch.from_numpy(d_mfcc_feat).float())
            label = Variable(torch.LongTensor(label))
            out = self.clf(input)
            loss = criterion(out, label)
            loss.backward()
            optim.step()
            print('%d/%d loss:%f' % (itr,total_iter,loss.data[0]))
        f = open('toy.pkl', 'wb')
        torch.save(self.clf.state_dict(), f)
        f.close()

    def eval(self):
        dev_data = self.reader.mini_batch_iterator(self.reader.val_person)
        y,pred = [],[]
        self.clf.eval()
        for itr, total_iter, (feat, label) in dev_data:
            mfcc_feat = [self.pad(mfcc(robust_scale(sig), rate, nfft=512, winlen=0.01)) for sig, rate in feat]
            d_mfcc_feat = np.stack([delta(_, 3) for _ in mfcc_feat])  # [B,T,H]
            d_mfcc_feat = d_mfcc_feat.transpose((1, 0,2))
            input = Variable(torch.from_numpy(d_mfcc_feat).float())
            out = self.clf(input)
            _,pred_ = torch.max(out,1)
            pred_ = pred_.data.numpy()
            y.extend(label)
            pred.extend(pred_.tolist())
            print('%d/%d loss' % (itr, total_iter))
        acc = accuracy_score(y,pred)
        print(acc)


if __name__ == '__main__':
    m = RNNModel()

    state_dict = torch.load('toy.pkl')
    m.clf.load_state_dict(state_dict)
    for i in range(10):
        m.train()
