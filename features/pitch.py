"""

Author: Xisen Jin
Date: 2018.05.24

Pitch detection using cesptrum
"""
import numpy as np
from preprocess import preemphasis
from sigproc import to_frames, window
from reader import Reader
from plotter import plot_frame, show, scatter
from endpoint import basic_endpoint_detection
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from itertools import chain
from sklearn.metrics import accuracy_score
import pickle


def pitch_feature(sig, rate):
    """
    Main call function for pitch features
    :param sig: signal after endpoint detection
    :param rate: 
    :return: 
    """
    pitch, frames = pitch_detect(sig, rate)
    # xi, yi, meanshift, slope_a, slope_b = find_smooth_subsequence(pitch)
    p = sub_endpoint_detect(frames)

    strs_1, str_idx_1 = find_smooth_subsequence(pitch[:p],base_tor=4)
    strs_2, str_idx_2 = find_smooth_subsequence(pitch[p:],base_tor=4, bias=p)
    print(str_idx_1,str_idx_2)
    feat = slope(strs_1), slope(strs_2), meanshift(strs_1, strs_2)
    plot_frame(pitch, where='212',show=True)
    return feat

def slope(seq):
    x = np.arange(0, len(seq))
    z = np.polyfit(x, seq, 1)
    return z[0]

def meanshift(seq1, seq2):
    return np.mean(seq2) - np.mean(seq1)

def downsampling(sig, src_rate, dst_rate):
    cnt = -1
    s = []
    for i in range(len(sig)):
        if i * dst_rate / src_rate > cnt + 1e-8:
            cnt += 1
            s.append(sig[i])
    return np.array(s)


def sub_endpoint_detect(frames):
    amp = [np.abs(_).sum() for _ in frames]
    p = 0
    max_score = -1000
    for i in range(10,len(amp)-10):
        if np.any(amp[i-2:i+3] < amp[i]):
            continue
        scores = []
        for j in range(i-10,i+11):
            scores.append(amp[j] - amp[i])
        s = sum(scores)
        if s > max_score:
            max_score = s
            p = i
    plot_frame(amp, where='211',sep=[p])
    return p

def pitch_detect(sig, rate, winlen=0.04, step=0.005):
    sig = downsampling(sig, rate, 10000)
    frames = to_frames(sig, 10000, winlen, step)
    g = []
    g_idx = []
    for frame in frames:
        scores, scores_idx = pitch_detect_frame(frame, 10000)
        g.append(scores)
        g_idx.append(scores_idx)
    # for l in g:
    #    print(l)
    pitch = max_pitch(g, g_idx)
    return pitch, frames


def max_pitch(g, g_idx):
    pitch = []
    for l, d in zip(g, g_idx):
        if l:
            idx = d[np.argmax(l)]
            p = 1 / (0.04 / 512 * idx)
            pitch.append(p)
        else:
            pitch.append(0)
    return pitch


def pitch_detect_frame(frame, rate):
    # add hamming window
    frame = window(frame, rate, 500, 512, 'hamming')
    Xw = np.fft.fft(frame, len(frame))
    log_Xw = np.log(np.abs(Xw))
    log_sig = np.fft.ifft(log_Xw, len(frame))
    scores, scores_idx = peak_score(np.abs(log_sig[:len(log_sig) // 2]))
    return scores, scores_idx


def peak_score(sig):
    l = []
    idx = []
    for i in range(20, 100):
        flg = True
        for item in list(sig[i - 5:i + 6]):
            if sig[i] < item:
                flg = False
        if not flg:
            continue
        p = i
        while p > 0 and sig[p] <= sig[i]:
            p -= 1
        q = i
        while q < len(sig) and sig[q] <= sig[i]:
            q += 1
        l.append(min(i - p, q - i))
        idx.append(i)
    return l, idx


def find_smooth_subsequence(pitch, base_tor=2, base_thres=50, bias=0):
    def main_func(tor=2, thres=50):
        i = 0
        strs = []
        str_idx = []
        while i < len(pitch):
            j = i + 1
            prev = pitch[i]
            k = tor
            seg = [pitch[i]]
            while j < len(pitch):
                if abs(pitch[j] - prev) > thres:
                    k -= 1
                else:
                    seg.append(pitch[j])
                    prev = pitch[j]
                if not k:
                    strs.append(seg)
                    str_idx.append((i + bias, j + bias))
                    break
                j += 1
            if j == len(pitch):
                strs.append(seg)
                str_idx.append((i + bias, j + bias))
                break
            i = j - tor + 1
        return strs, str_idx

    strs, str_idx = [], []
    tor, thres = base_tor, base_thres
    while len(strs) < 2:
        strs, str_idx = main_func(tor, thres)
        thres -= 10
        tor += 1
    obj = sorted(zip(strs, str_idx), key=lambda x: -len(x[0]))
    strs, str_idx = tuple(zip(*obj))

    return strs[0], str_idx[0]


if __name__ == '__main__':

    r = Reader(debug=False)
    train_itr = r.iterator(r.train)
    val_itr = r.iterator(r.val_person)
    cnt = 0
    p = []
    labels = []
    X = []
    X_test = []
    labels_test = []
    scaler = RobustScaler(with_centering=False)

    for idx, l, (sig, rate), label, filename in train_itr:
        reg = re.compile('(\d+)-(\d+)-(\d+).wav')
        audio = reg.match(filename).group(2)
        person = reg.match(filename).group(1)
        if audio not in ['00', '01']:
            continue
        print(person, label, filename)
        cnt += 1
        l, r = basic_endpoint_detection(sig, rate)
        #sig = preemphasis(sig)
        frames = to_frames(sig[l:r], rate)
        feature = pitch_feature(sig[l:r], rate)
        p.append((feature[0],feature[2]))
        X.append(feature)
        labels.append(label)
        if cnt == 100: break
    scatter(p,labels,'111',True)

    '''
    cnt = 0
    for idx, l, (sig, rate), label, filename in val_itr:
        reg = re.compile('(\d+)-(\d+)-(\d+).wav')
        audio = reg.match(filename).group(2)
        person = reg.match(filename).group(1)
        if audio not in ['00','01']:
            continue
        print(person,label,filename)
        cnt += 1
        l ,r = basic_endpoint_detection(sig, rate)
        frames = to_frames(sig[l:r], rate)
        feature = pitch_feature(sig[l:r], rate)
        X_test.append(feature)
        labels_test.append(label)
        if cnt == 100:
            break

    fs = open('models/scale.pkl','wb')
    fc = open('models/rfc.pkl','wb')


    scaler.fit(X)
    X = scaler.transform(X)
    X_test = scaler.transform(X_test)

    #clf = RandomForestClassifier()
    clf = SVC()
    clf.fit(X, labels)

    pickle.dump(clf, fc)
    pickle.dump(scaler, fs)

    pred = clf.predict(X_test)
    acc = accuracy_score(labels_test, pred)
    print(acc)
    '''
