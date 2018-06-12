import numpy as np
from features import mfcc, delta, to_frames
from config import *
from reader import Reader
import features
from sklearn.preprocessing import robust_scale, scale
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from rnn_clf import RNN, HRNN, cuda_, Transformer, HRNN_Att, HMRNN
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from features.endpoint import basic_endpoint_detection, robust_endpoint_detection
import plotter
import re
import argparse
import random


def cvar(arr):
    v = cuda_(Variable(torch.from_numpy(arr).float()))
    return v

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

    def pad_batch(self, feat):
        #max_len = max([len(_) for _ in mfcc0])
        max_len = 200
        r = []
        for b in feat:
            r.append(self.pad(b, ((0, max_len - len(b)),(0,0))))
        return np.array(r)

    def endpoint_detect(self, sig, rate, augment=False):
        left, right = basic_endpoint_detection(sig, rate)
        if augment:
            shift_min, shift_max = 0, int(0.1 * rate)
            s_l, s_r = random.randint(shift_min, shift_max), random.randint(shift_min, shift_max)
            left -= s_l
            right += s_r
            if left < 0: left = 0
            if right < 0 : right = 0

        sound = sig[left:right].reshape(-1, 1)
        sound = scale(sound, with_mean=False)
        return sound

    def feature_extract_mfcc(self, sound, rate):
        """
        extract every features for training
        - frequency space: MFCC. pitch
        :return: 
        """
        reg = re.compile('(\d+)-(\d+)-(\d+).wav')
        #plotter.plot_frame(sound, show=True)
        mfcc0 = mfcc(sound.reshape(1,-1), rate, winlen=cfg.frame, winstep=cfg.step, nfft=1536, winfunc=np.hamming)
        mfcc0 = mfcc0 - np.mean(mfcc0)
        #mfcc1 = self.deviation(mfcc0,2)
        #mfcc2 = self.deviation(mfcc1,2)
        #mfcc0 = delta(mfcc0,3)
        mfcc1 = delta(mfcc0,2)
        mfcc2 = delta(mfcc1,2)

        '''
        if audio in ['01','00']:
            print(filename)
            plotter.plot_mfcc(mfcc0,'311')
            plotter.plot_mfcc(mfcc1,'312')
            plotter.plot_mfcc(mfcc2,'313')
            plotter.show()
        '''
        return (mfcc0, mfcc1, mfcc2), min(len(mfcc0),200)

    def feature_extract_pitch(self, sound, rate):
        PITCH_SCALE = 150
        pitch0, _ = features.pitch_detect_sr(sound.reshape(-1), rate, winlen=cfg.frame, step=cfg.step)
        pitch0 = np.array(pitch0).reshape(-1, 1) / PITCH_SCALE
        pitch1 = self.deviation(pitch0)
        return [pitch0.reshape(-1,1), pitch1.reshape(-1,1)]

    def feature_extract_timespace(self, sound, rate):
        amp0 = scale(features.amplitude_feature(sound.reshape(-1), rate, winlen=cfg.frame, step=cfg.step)).reshape(-1,1)
        amp1 = self.deviation(amp0)
        return [amp0.reshape(-1,1), amp1.reshape(-1,1)]

    def load(self):
        state_dict = torch.load(cfg.model_path)
        self.clf.load_state_dict(state_dict)


class RNNModel(_ModelBase):
    def __init__(self):
        super().__init__()
        self.clf = HRNN()
        #self.clf = CNN_SP()
        self.clf = cuda_(self.clf)

    def get_batch_full(self, feat, files, augment=False):
        sound = [self.endpoint_detect(sig, rate, augment=augment) for sig, rate in feat]
        rate = [_[1] for _ in feat]
        features = []
        len0 = []
        for s, r in zip(sound, rate):
            f = []
            mfcc_feat, l = self.feature_extract_mfcc(s,r) # [T,H]
            len0.append(l)
            for item in mfcc_feat:
                f.append(item)
            if cfg.use_pitch:
                f.extend(self.feature_extract_pitch(s,r))
            if cfg.use_timefeat:
                f.extend(self.feature_extract_timespace(s,r))
            features.append(f)

        features = [_ for _ in zip(*features)]

        inp = [cvar(self.pad_batch(_)).transpose(0, 1) for _ in features]
        inp = torch.cat(inp, 2)
        return inp, np.array(len0)

    def train(self, lr=cfg.lr):
        #train_data = self.reader.mini_batch_iterator(self.reader.train)
        train_data = self.reader.mini_batch_iterator(self.reader.train)
        optim = torch.optim.Adam(self.clf.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        for itr, total_iter, feat, label, files in train_data:
            optim.zero_grad()
            inp, len0 = self.get_batch_full(feat, files, augment=True)
            label = cuda_(Variable(torch.LongTensor(label)))
            out = self.clf(inp, len0)
            loss = criterion(out, label)
            loss.backward()
            optim.step()
            printer.info('%d/%d loss:%f' % (itr,total_iter,loss.data[0]))


    def test_iter(self, itr, total_iter, feat, label, files):
        inp, len0 = self.get_batch_full(feat, files)
        out = self.clf(inp, len0)
        prob_ = F.softmax(out, dim=1)
        _, pred_ = torch.max(out, 1)
        pred_ = pred_.data.cpu().numpy().tolist()
        prob_ = prob_.data.cpu().numpy().tolist()
        return pred_, prob_

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
            try:
                m.clf.m.adjust_param()
            except AttributeError:
                pass

    elif args.mode == 'adjust':
        state_dict = torch.load(cfg.model_path)
        m.clf.load_state_dict(state_dict)
        for i in range(10):
            m.train()
        #break
    #m.test()
