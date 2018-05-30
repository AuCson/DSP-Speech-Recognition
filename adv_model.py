import numpy as np
from features import mfcc, delta, to_frames
from config import *
from reader import Reader
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import robust_scale, scale
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from features.preprocess import downsampling
from rnn_clf import RNN, HRNN, cuda_, Transformer
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from features.endpoint import basic_endpoint_detection
import plotter
import re
import argparse
import random
from adv_clf import AdvTransformer
from model import _ModelBase

class AdvModel(_ModelBase):
    def __init__(self):
        super().__init__()
        self.clf = AdvTransformer()
        self.clf = cuda_(self.clf)

    def train(self, lr=cfg.lr):
        train_data = self.reader.mini_batch_iterator(self.reader.train, requires_speaker=True)
        optim = torch.optim.Adam(self.clf.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        for itr, total_iter, feat, label, files, speaker in train_data:
            optim.zero_grad()
            features = [self.feature_extract_mfcc(a,b,filename,augment=True) for (a,b),filename in zip(feat,files)]
            features = [_ for _ in zip(*features)]
            features[0], features[1], features[2] = self.pad_batch(features[0], features[1], features[2])
            features[3], features[4], features[5] = np.array(features[3]), np.array(features[4]), np.array(features[5])
            mfcc0, mfcc1, mfcc2 = cuda_(Variable(torch.from_numpy(features[0]).float())), \
                                  cuda_(Variable(torch.from_numpy(features[1]).float())),\
                                 cuda_(Variable(torch.from_numpy(features[2]).float()))
            mfcc0, mfcc1, mfcc2 = mfcc0.transpose(0,1), mfcc1.transpose(0,1), mfcc2.transpose(0,1)
            label = cuda_(Variable(torch.LongTensor(label)))
            speaker = cuda_(Variable(torch.LongTensor(speaker)))
            out1, out2 = self.clf(mfcc0, mfcc1, mfcc2, features[3])
            loss1 = criterion(out1, label)
            loss2 = criterion(out2, speaker)
            loss = loss1 + loss2
            loss.backward()
            optim.step()
            printer.info('%d/%d loss:%f %f %f' % (itr,total_iter,loss1.data[0], loss2.data[0], loss.data[0]))
            #break
        #self.test()

    def test_iter(self, itr, total_iter, feat, label, files):
        features = [self.feature_extract_mfcc(a, b, filename) for (a, b), filename in zip(feat, files)]
        features = [_ for _ in zip(*features)]
        features[0], features[1], features[2] = self.pad_batch(features[0], features[1], features[2])
        features[3], features[4], features[5] = np.array(features[3]), np.array(features[4]), np.array(features[5])
        mfcc0, mfcc1, mfcc2 = cuda_(Variable(torch.from_numpy(features[0]).float())), \
                              cuda_(Variable(torch.from_numpy(features[1]).float())), \
                              cuda_(Variable(torch.from_numpy(features[2]).float()))
        mfcc0, mfcc1, mfcc2 = mfcc0.transpose(0, 1), mfcc1.transpose(0, 1), mfcc2.transpose(0, 1)
        out,_ = self.clf(mfcc0, mfcc1, mfcc2, features[3])
        out = F.softmax(out, dim=1)
        prob, pred_ = torch.max(out, 1)
        pred_ = pred_.data.cpu().numpy().tolist()
        prob = prob.data.cpu().numpy().tolist()
        return pred_, prob

    def test(self):
        dev_data = self.reader.mini_batch_iterator(self.reader.val_person)
        y,pred = [],[]
        pred_prob = []
        err = []
        self.clf.eval()
        for itr, total_iter, feat, label, files in dev_data:
            pred_, prob_ = self.test_iter(itr, total_iter, feat, label, files)
            y.extend(label)
            pred.extend(pred_)
            pred_prob.extend(prob_)
            printer.info('%d/%d loss' % (itr, total_iter))
            for i,_ in enumerate(pred_):
                if pred_[i] != label[i]:
                    err_info = '{} {} {}'.format(files[i],pred_[i], label[i])
                    err.append(err_info)
            #if itr > 1000: break
        acc = accuracy_score(y,pred)
        printer.info(acc)
        for e in err:
            printer.info(e)
        #cm = confusion_matrix(y, pred)
        #print(cm)
        self.clf.train()
        return acc


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

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
    if cfg.cuda:
        torch.cuda.set_device(cfg.cuda_device)

    m = AdvModel()
    if args.mode == 'test':
        state_dict = torch.load(cfg.model_path)
        m.clf.load_state_dict(state_dict)
        m.test()
    elif args.mode == 'train':
        prev_acc = 0
        lr = cfg.lr
        early_stop = 3
        for i in range(10):
            m.train(lr)
            acc = m.test()
            if acc > prev_acc:
                f = open(cfg.model_path, 'wb')
                torch.save(m.clf.state_dict(), f)
                f.close()
                prev_acc = acc
                lr *= 0.8
            else:
                early_stop -= 1
                lr *= 0.5
                if not early_stop:
                    break

    elif args.mode == 'adjust':
        state_dict = torch.load(cfg.model_path)
        m.clf.load_state_dict(state_dict)
        for i in range(10):
            m.train()
        #break
    #m.test()
