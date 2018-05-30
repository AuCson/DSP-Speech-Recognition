import numpy as np
from features import mfcc, delta, to_frames
from config import *
from reader import Reader
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import robust_scale, scale
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from features.preprocess import downsampling
from rnn_clf import RNN, HRNN, cuda_, Transformer, CNN_SP, HRNN_Att, HMRNN
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from features.endpoint import basic_endpoint_detection
import plotter
import re
import argparse
import random

class _ModelBase:
    def __init__(self):
        self.reader = Reader(debug=False)
        self.clf = None

    def deviation(self, arr, smooth=1):
        l = []
        for i in range(len(arr) - smooth):
            l.append(arr[i + smooth] - arr[i])
        return np.array(l)

    def pad(self, b, shape):
        if len(b) < 200:
            return np.pad(b,((0, 200 - len(b)),(0,0)),'constant',constant_values=0)
        else:
            return np.array(b[:200])

    def pad2(self, b, shape):
        return np.pad(b,shape,'constant',constant_values=0)

    def pad_batch(self, mfcc0, mfcc1, mfcc2):
        #max_len = max([len(_) for _ in mfcc0])
        max_len = 200
        l0,l1,l2 = [],[],[]
        for b0, b1, b2 in zip(mfcc0, mfcc1, mfcc2):
            l0.append(self.pad(b0, ((0, max_len - len(b0)),(0,0))))
            l1.append(self.pad(b1, ((0, max_len - len(b1)), (0, 0))))
            l2.append(self.pad(b2, ((0, max_len - len(b2)), (0, 0))))
        return np.array(l0), np.array(l1), np.array(l2)


    def feature_extract_mfcc(self, sig, rate, filename, augment=False):
        """
        extract every features for training
        :return: 
        """
        reg = re.compile('(\d+)-(\d+)-(\d+).wav')
        audio = reg.match(filename).group(2)
        left, right, amp, zcr = basic_endpoint_detection(sig, rate, return_feature=True)

        if augment:
            shift_min, shift_max = 0, int(0.1 * rate)
            s_l, s_r = random.randint(shift_min, shift_max), random.randint(shift_min, shift_max)
            left -= s_l
            right += s_r
            if left < 0: left = 0
            if right < 0 : right = 0

        sound = sig[left:right].reshape(-1, 1)
        sound = scale(sound, with_mean=False)
        #plotter.plot_frame(sound, show=True)
        #mfcc0 = mfcc(sound, rate, winlen=cfg.frame, winstep=cfg.step, nfft=512, winfunc=np.hamming)
        mfcc0 = mfcc(sound, rate, winlen=cfg.frame, winstep=cfg.step, nfft=1536, winfunc=np.hamming)
        mfcc0 = mfcc0 - np.mean(mfcc0)
        mfcc1 = self.deviation(mfcc0,2)
        mfcc2 = self.deviation(mfcc1,2)
        mfcc0 = delta(mfcc0,3)
        mfcc1 = delta(mfcc1,3)
        mfcc2 = delta(mfcc2,3)
        # mean normalize

        '''
        if audio in ['01','00']:
            print(filename)
            plotter.plot_mfcc(mfcc0,'311')
            plotter.plot_mfcc(mfcc1,'312')
            plotter.plot_mfcc(mfcc2,'313')
            plotter.show()
        '''
        return mfcc0, mfcc1, mfcc2, min(len(mfcc0),200), min(len(mfcc1),200), min(len(mfcc2),200)

    def load(self):
        state_dict = torch.load(cfg.model_path)
        self.clf.load_state_dict(state_dict)


class RNNModel(_ModelBase):
    def __init__(self):
        super().__init__()
        self.clf = HMRNN()
        #self.clf = CNN_SP()
        self.clf = cuda_(self.clf)

    def train(self, lr=cfg.lr):
        #train_data = self.reader.mini_batch_iterator(self.reader.train)
        train_data = self.reader.mini_batch_iterator(self.reader.train)
        optim = torch.optim.Adam(self.clf.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        for itr, total_iter, feat, label, files in train_data:
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
            out = self.clf(mfcc0, mfcc1, mfcc2, features[3])
            loss = criterion(out, label)
            loss.backward()
            optim.step()
            printer.info('%d/%d loss:%f' % (itr,total_iter,loss.data[0]))
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
        out = self.clf(mfcc0, mfcc1, mfcc2, features[3])
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

    m = RNNModel()
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
