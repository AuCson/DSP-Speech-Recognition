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

class ToyModel:
    def __init__(self):
        self.reader = Reader()
        self.m = RNN()

    def train(self):

        def pad(feat):
            if feat.shape[0] < 400:
                feat = np.pad(feat, ((0,400-feat.shape[0]),(0,0)),'constant')
            else:
                feat = feat[:400]
            return feat
        '''
        train_data = self.reader.mini_batch_iterator(self.reader.train)
        optim = torch.optim.Adam(self.m.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        for itr, total_iter, (feat, label) in train_data:
            optim.zero_grad()
            mfcc_feat = [pad(mfcc(robust_scale(sig), rate, nfft=512, winlen=0.01)) for sig, rate in feat]
            d_mfcc_feat = np.stack([delta(_, 3) for _ in mfcc_feat]) # [B,T,H]
            d_mfcc_feat = d_mfcc_feat.transpose((1,0,2))
            input = Variable(torch.from_numpy(d_mfcc_feat).float())
            label = Variable(torch.LongTensor(label))
            out = self.m(input)
            loss = criterion(out, label)
            loss.backward()
            optim.step()
            print('%d/%d loss:%f' % (itr,total_iter,loss.data[0]))
        f = open('toy.pkl', 'wb')
        torch.save(self.m.state_dict(), f)
        f.close()
        '''
        dev_data = self.reader.mini_batch_iterator(self.reader.val_person)
        y,pred = [],[]
        self.m.eval()
        for itr, total_iter, (feat, label) in dev_data:

            mfcc_feat = [pad(mfcc(robust_scale(sig), rate, nfft=512, winlen=0.01)) for sig, rate in feat]
            d_mfcc_feat = np.stack([delta(_, 3) for _ in mfcc_feat])  # [B,T,H]
            d_mfcc_feat = d_mfcc_feat.transpose((1, 0,2))
            input = Variable(torch.from_numpy(d_mfcc_feat).float())
            out = self.m(input)
            _,pred_ = torch.max(out,1)
            pred_ = pred_.data.numpy()
            y.extend(label)
            pred.extend(pred_.tolist())
            print('%d/%d loss' % (itr, total_iter))
        acc = accuracy_score(y,pred)
        print(acc)
        '''
        clf = SGDClassifier(loss='log')
        for itr, total_iter, (feat, label) in train_data:
            print('%d/%d' % (itr,total_iter), end='\r')
            mfcc_feat = [pad(mfcc(robust_scale(sig), rate, nfft=512, winlen=0.01)) for sig,rate in feat]
            d_mfcc_feat = np.array([delta(_, 3).reshape(-1) for _ in mfcc_feat])
            clf.partial_fit(d_mfcc_feat, label, np.arange(20))


        clf = pickle.load(f)
        f.close()
        dev_data = self.reader.mini_batch_iterator(self.reader.val_inst)
        y,pred = [],[]
        for itr, total_iter, (feat, label) in dev_data:
            print('%d/%d' % (itr, total_iter), end='\r')
            mfcc_feat = [pad(mfcc(robust_scale(sig), rate, nfft=512, winlen=0.01)) for sig, rate in feat]
            d_mfcc_feat = np.array([delta(_, 3).reshape(-1) for _ in mfcc_feat])
            y_ = clf.predict(d_mfcc_feat)
            y.extend(label)
            pred.extend(y_.tolist())
        acc = accuracy_score(y,pred)
        print(acc)
        '''


if __name__ == '__main__':
    m = ToyModel()

    state_dict = torch.load('toy.pkl')
    m.m.load_state_dict(state_dict)
    for i in range(10):
        m.train()
