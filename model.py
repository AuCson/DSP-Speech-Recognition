import numpy as np
from features import mfcc, delta, to_frames
from config import *
from reader import Reader
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import robust_scale
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from rnn_clf import RNN, HRNN, cuda_, Transformer
import torch
from torch.autograd import Variable
from features.endpoint import basic_endpoint_detection
import plotter
import re
import argparse

class _ModelBase:
    def __init__(self):
        self.reader = Reader(debug=False)

    def deviation(self, arr, smooth=1):
        l = []
        for i in range(len(arr)-smooth):
            l.append(arr[i+smooth]-arr[i])
        return np.array(l)

    def pad(self, feat):
        if feat.shape[0] < 400:
            feat = np.pad(feat, ((0, 400 - feat.shape[0]), (0, 0)), 'constant')
        else:
            feat = feat[:400]
        return feat

    def feature_extract(self, sig, rate, filename):
        """
        extract every features for training
        :return: 
        """
        reg = re.compile('(\d+)-(\d+)-(\d+).wav')
        audio = reg.match(filename).group(2)
        left, right = basic_endpoint_detection(sig, rate)
        sound = sig[left:right].reshape(-1,1)
        #plotter.plot_frame(sound, show=True)
        mfcc0 = mfcc(sound, rate, winlen=cfg.frame, winstep=cfg.step, nfft=1024, winfunc=np.hamming)
        mfcc0 = delta(mfcc0,3)
        # mean normalize
        mfcc0 = mfcc0 - np.mean(mfcc0)
        mfcc1 = self.deviation(mfcc0,5)
        mfcc2 = self.deviation(mfcc1,5)
        '''
        if audio in ['01','00']:
            print(filename)
            plotter.plot_mfcc(mfcc0,'311')
            plotter.plot_mfcc(mfcc1,'312')
            plotter.plot_mfcc(mfcc2,'313')
            plotter.show()
        '''
        return self.pad(mfcc0), self.pad(mfcc1), self.pad(mfcc2), len(mfcc0), len(mfcc1), len(mfcc2)


class RNNModel(_ModelBase):
    def __init__(self):
        super().__init__()
        self.clf = Transformer()
        self.clf = cuda_(self.clf)

    def train(self):
        #train_data = self.reader.mini_batch_iterator(self.reader.train)
        train_data = self.reader.mini_batch_iterator(self.reader.train)
        optim = torch.optim.Adam(self.clf.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        for itr, total_iter, feat, label, files in train_data:
            optim.zero_grad()
            features = [self.feature_extract(a,b,filename) for (a,b),filename in zip(feat,files)]
            features = [np.array(_) for _ in zip(*features)]
            mfcc0, mfcc1, mfcc2 = cuda_(Variable(torch.from_numpy(features[0]).float())), \
                                  cuda_(Variable(torch.from_numpy(features[1]).float())),\
                                 cuda_(Variable(torch.from_numpy(features[2]).float()))
            mfcc0, mfcc1, mfcc2 = mfcc0.transpose(0,1), mfcc1.transpose(0,1), mfcc2.transpose(0,1)
            label = cuda_(Variable(torch.LongTensor(label)))
            out = self.clf(mfcc0, mfcc1, mfcc2, features[3])
            loss = criterion(out, label)
            loss.backward()
            optim.step()
            print('%d/%d loss:%f' % (itr,total_iter,loss.data[0]))
            #break
        #self.eval()


    def eval(self):
        dev_data = self.reader.mini_batch_iterator(self.reader.val_person)
        y,pred = [],[]
        self.clf.eval()
        for itr, total_iter, feat, label, files in dev_data:
            features = [self.feature_extract(a, b, filename) for (a, b), filename in zip(feat, files)]
            features = [np.array(_) for _ in zip(*features)]
            mfcc0, mfcc1, mfcc2 = cuda_(Variable(torch.from_numpy(features[0]).float())), \
                                  cuda_(Variable(torch.from_numpy(features[1]).float())),\
                                 cuda_(Variable(torch.from_numpy(features[2]).float()))
            mfcc0, mfcc1, mfcc2 = mfcc0.transpose(0, 1), mfcc1.transpose(0, 1), mfcc2.transpose(0, 1)
            out = self.clf(mfcc0, mfcc1, mfcc2, features[3])
            _,pred_ = torch.max(out,1)
            pred_ = pred_.data.cpu().numpy().tolist()
            y.extend(label)
            pred.extend(pred_)
            print('%d/%d loss' % (itr, total_iter))
            #for i,_ in enumerate(pred_):
            #    if pred_[i] != label[i]:
            #       logger.info(files[i])
        acc = accuracy_score(y,pred)
        print(acc)
        #cm = confusion_matrix(y, pred)
        #print(cm)
        self.clf.train()
        return acc


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('-cfg',nargs='*')
    args = parser.parse_args()
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    m = RNNModel()
    if args.mode == 'test':
        state_dict = torch.load(cfg.model_path)
        m.clf.load_state_dict(state_dict)
        m.eval()
    elif args.mode == 'train':
        prev_acc = 0
        for i in range(10):
            m.train()
            acc = m.eval()
            if acc > prev_acc:
                f = open(cfg.model_path, 'wb')
                torch.save(m.clf.state_dict(), f)
                f.close()
                prev_acc = acc
            else:
                break

    elif args.mode == 'adjust':
        state_dict = torch.load(cfg.model_path)
        m.clf.load_state_dict(state_dict)
        for i in range(10):
            m.train()
        #break
    #m.eval()