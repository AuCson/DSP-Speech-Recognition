"""

Author: Xisen Jin
Date: 2018.05.24

Pitch detection using cesptrum
"""
import numpy as np
from features.preprocess import preemphasis
from features.sigproc import to_frames, window
from reader import Reader
from plotter import plot_frame, show, scatter
from features.endpoint import basic_endpoint_detection
import re
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from itertools import chain
from sklearn.metrics import accuracy_score
import pickle


def pitch_feature(sig, rate, gender='male'):
    """
    Main call function for pitch features
    :param sig: signal after endpoint detection
    :param rate: 
    :return: 
    """

    pitch, frames = pitch_detect(sig, rate, gender=gender)
    # xi, yi, meanshift, slope_a, slope_b = find_smooth_subsequence(pitch)
    p = sub_endpoint_detect(frames)
    p_bias = 5 if p > 15 else 0
    strs_1, str_idx_1 = find_smooth_subsequence(pitch[p_bias:p],base_tor=4,bias=p_bias)
    p_bias = 5 if len(pitch) - p > 15 else 0
    strs_2, str_idx_2 = find_smooth_subsequence(pitch[p:],base_tor=4, bias=p)
    l1,l2 = str_idx_1[1] - str_idx_1[0], str_idx_2[1] - str_idx_2[0]
    print(str_idx_1,str_idx_2)
    feat = slope(strs_1), slope(strs_2), meanshift(strs_1, strs_2), l2 / l1, (str_idx_2[0] - str_idx_1[1])/len(frames)
    print(feat)
    #plot_frame(pitch, where='212',show=True)
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
    #plot_frame(amp, where='211',sep=[p])
    if p == 0:
        return len(amp) // 2
    return p

def pitch_detect(sig, rate, winlen=0.0512, step=0.01, gender='male'):
    sig = downsampling(sig, rate, 10000)
    frames = to_frames(sig, 10000, winlen, step)
    g = []
    g_idx = []
    for frame in frames:
        scores, scores_idx = pitch_detect_frame(frame, 10000, gender)
        g.append(scores)
        g_idx.append(scores_idx)
    # for l in g:
    #    print(l)
    pitch = robust_max_pitch(g, g_idx)
    return pitch, frames


def max_pitch(g, g_idx):
    pitch = []
    for l, d in zip(g, g_idx):
        if l:
            idx = d[np.argmax(l)]
            p = 1 / (0.0001 * idx)
            pitch.append(p)
        else:
            pitch.append(0)
    return pitch


def robust_max_pitch(g, g_idx):
    """
    more robust to half-frequency error
    :param g: 
    :param g_idx: 
    :return: 
    """
    C = 50
    pitch = max_pitch(g, g_idx)
    for i in range(1,len(pitch)):
        if abs(2 * pitch[i] - pitch[i-1]) < C and pitch[i] < 170:
            pitch[i] = 2 * pitch[i]
    for i in range(len(pitch)-2, 0, -1):
        if abs(2 * pitch[i] - pitch[i+1]) < C and pitch[i] < 170:
            pitch[i] = 2 * pitch[i]
    return pitch

def pitch_detect_frame(frame, rate, gender):
    # add hamming window
    frame = window(frame, rate, 1000, 'hamming')
    Xw = np.fft.fft(frame, len(frame))
    log_Xw = np.log(np.abs(Xw))
    #log_Xw = np.concatenate([log_Xw,np.zeros(1000)])
    log_sig = np.fft.ifft(log_Xw, len(log_Xw))
    scores, scores_idx = peak_score(np.abs(log_sig[:len(log_sig) // 2]), gender)
    #plot_frame(np.abs(frame), where='211')
    #plot_frame(np.abs(log_sig), where='212',show=True)
    return scores, scores_idx


def peak_score(sig, gender='male'):
    l = []
    idx = []

    #min_f, max_f = (20, 66) if gender == 'female' else (30,100)  # 150-500Hz, 50-330Hz
    min_f, max_f = 20, 100
    for i in range(min_f, max_f):
        flg = True
        for item in list(sig[i - 3:i + 4]):
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


def find_smooth_subsequence(pitch, base_tor=2, base_thres=30, bias=0):
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
    strs, str_idx = main_func(tor, thres)
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
        #if not (person[3:7] == '0713' and int(person[7:]) < 300):
        #    continue
        if audio not in ['00', '01']:
            continue
        print(person, label, filename)
        cnt += 1
        l, r = basic_endpoint_detection(sig, rate)
        sig = preemphasis(sig, coeff=1.0)
        frames = to_frames(sig[l:r], rate)
        feature = pitch_feature(sig[l:r], rate)
        p.append((feature[0],feature[2]))
        X.append(feature)
        labels.append(label)
        if cnt == 1000: break
    scatter(p,labels,'111',True)


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
    #clf = SVC(kernel='rbf')
    clf = GradientBoostingClassifier()
    clf.fit(X, labels)

    pickle.dump(clf, fc)
    pickle.dump(scaler, fs)

    pred = clf.predict(X_test)
    acc = accuracy_score(labels_test, pred)
    print(acc)

