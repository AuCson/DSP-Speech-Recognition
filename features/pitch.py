"""

Author: Xisen Jin
Date: 2018.05.24

Pitch detection using cesptrum
"""
import sys
sys.path.insert(0, '../')

import features
import numpy as np
from features.preprocess import preemphasis
from features.sigproc import to_frames, window, acr
from reader import Reader
from plotter import plot_frame, show, scatter
from features.endpoint import basic_endpoint_detection, get_amplitude
import re
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import scale, RobustScaler
from sklearn.metrics import accuracy_score
import pickle
from features.preprocess import downsampling

def pitch_feature(sig, rate, gender='male'):
    """
    Main call function for pitch features
    :param sig: signal after endpoint detection
    :param rate:
    :return:
    """

    pitch, frames = pitch_detect_sr(sig, rate)
    p = sub_endpoint_detect(frames)
    p_bias = 5 if p > 15 else 0
    amp = scale(get_amplitude(frames))
    amp1, amp2 = amp[p_bias:p], amp[p:]
    strs_1, str_idx_1 = find_smooth_subsequence(pitch[p_bias:p],bias=p_bias)
    strs_2, str_idx_2 = find_smooth_subsequence(pitch[p:],bias=p)
    #l1,l2 = str_idx_1[1] - str_idx_1[0], str_idx_2[1] - str_idx_2[0]
    print(str_idx_1,str_idx_2)
    feat = slope(strs_1), slope(strs_2)
    print(feat)
    plot_frame(pitch, where='212',show=True)
    return feat

def slope(seq):
    x = np.arange(0, len(seq))
    z = np.polyfit(x, seq, 1)
    return z[0]

def quad_params(seq):
    x = np.arange(0, len(seq))
    z = np.polyfit(x, seq, 2)
    return z[1] * z[1] - 4 * z[0] * z[2]

def peakshift(seq1, seq2):
    return np.max(seq2) - np.max(seq1)


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
    log_sigs = []
    for frame in frames:
        frame = center_clip(frame, False)
        log_sig = pitch_detect_frame(frame, 10000, gender)
        log_sigs.append(log_sig)
    log_sigs = smooth(log_sigs)
    scores = [peak_score(log_sig) for log_sig in log_sigs]
    pitch = robust_max_pitch(scores)
    return pitch, frames

def pitch_detect_sr(sig, rate, winlen=0.0512, step=0.01):
    """
    detect the pitch with self relevance score.
    """
    sig = downsampling(sig, rate, 10000)
    frames = to_frames(sig, 10000, winlen, step)
    scores = []
    for frame in frames:
        frame = center_clip(frame, False)
        score = pitch_detect_frame_sr(frame, 10000)
        scores.append(score)
    scores = smooth(scores, 2)
    #scores = [peak_score(score) for score in scores]
    pitch = max_pitch(scores, bias=20)
    return pitch, frames
    
def pitch_detect_frame_sr(frame, rate):
    """
    resolution is $rate
    0.0512 * 10000 = 512 points
    move 1 point gives 100000 Hz (T=1e-4)
    move n points: T = n * 1e-4
    range should be within 50 - 500 Hz
    That is 20 points to 200 points. 
    each points gives 1e-4 resolutioFn in time scope
    """

    frame = window(frame, rate, 50, 900, 'hamming')
    frame = np.abs(frame)
    min_shift = 20
    max_shift = 200
    scores = []
    for n in range(min_shift, max_shift):
        scores.append(acr(frame, n))
    #plot_frame(frame, where='212')
    #plot_frame(scores,where='211',show=True)
    return scores
            

def pitch_detect_frame(frame, rate, gender):
    # add hamming window
    frame = window(frame, rate, 50, 1000, 'hamming')
    Xw = np.fft.fft(frame, len(frame))
    log_Xw = np.log(np.abs(Xw))
    log_sig = np.fft.ifft(log_Xw, len(log_Xw))
    #plot_frame(np.abs(frame), where='211')
    #plot_frame(np.abs(log_sig), where='212',show=True)
    return np.abs(log_sig)

def center_clip(frame, binary=True):
    med = np.median(frame[frame>=0])
    new_frame = []
    for i in range(len(frame)):
        if frame[i] > med:
            new_frame.append(1 if binary else frame[i] - med)
        elif frame[i] < -med:
            new_frame.append(-1 if binary else frame[i] + med)
        else:
            new_frame.append(0)
    return np.array(new_frame)

def smooth(g, degree=2):
    g = np.array(g)
    for i in range(len(g)):
        left = i - degree if i - degree>= 0 else 0
        right = i + degree if i + degree < len(g) else len(g) - 1
        g[i] = np.mean(g[left:right], axis=0)
    g = g.tolist()
    return g

def max_pitch(g, bias=20):
    pitch = []
    for l in g:
        idx = bias + np.argmax(l)
        p = 1 / (0.0001 * idx)
        pitch.append(p)
    return pitch

def greedy_max_pitch(g, bias=20):
    """
    find the first peak all the time
    :param g:
    :param bias:
    :return:
    """
    pitch = []
    for l in g:
        p = 0
        for i in range(len(l)-1):
            if l[i] > l[i+1]:
                p = 1 / (0.0001 * (i+bias))
                break
        pitch.append(p)
    return pitch

def robust_max_pitch(g, bias=20):
    """
    more robust to half-frequency error
    :param g:
    :param g_idx:
    :return:
    """
    C = 50
    pitch = max_pitch(g, bias)
    for i in range(1,len(pitch)):
        if abs(2 * pitch[i] - pitch[i-1]) < C and pitch[i] < 170:
            pitch[i] = 2 * pitch[i]
    for i in range(len(pitch)-2, 0, -1):
        if abs(2 * pitch[i] - pitch[i+1]) < C and pitch[i] < 170:
            pitch[i] = 2 * pitch[i]
    return pitch

def dp_max_pitch(g):
    g = np.array(g)
    dp = np.zeros(g.shape)
    prev = np.zeros(g.shape, dtype=int)
    for i in range(1,dp.shape[0]):
        for j in range(0,dp.shape[1]):
            reward = [dp[i-1][k] - 5 * abs(k-j) + g[i][j] for k in range(dp.shape[1])]
            max_rw = max(reward)
            step = reward.index(max_rw)
            dp[i][j] = max_rw
            prev[i][j] = step
    i = dp.shape[0] - 1
    path = np.zeros(g.shape[0])
    while i >= 0:
        path[i] = 10000 / step
        step = prev[i][step]
        i -= 1
    return path.tolist()

def peak_score(sig, gender='male'):
    l = []
    idx = []

    #min_f, max_f = (20, 66) if gender == 'female' else (30,100)  # 150-500Hz, 50-330Hz
    min_f, max_f = 20, 100
    for i in range(min_f, max_f):
        p = i
        while p > 0 and sig[p] <= sig[i]:
            p -= 1
        q = i
        while q < len(sig) and sig[q] <= sig[i]:
            q += 1
        l.append(min(i - p, q - i))
        idx.append(i)
    return l


def find_smooth_subsequence(pitch, base_tor=3, base_thres=30, bias=0):
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
        #if person.endswith('83'): continue
        #if not (person[3:7] == '0713' and int(person[7:]) < 300):
        #    continue
        #if audio not in ['00', '01']:
        #    continue
        print(person, label, filename)
        cnt += 1
        l, r = basic_endpoint_detection(sig, rate)
        #l,r= 0,len(sig)
        #sig = preemphasis(sig, coeff=0.97)
        frames = to_frames(sig[l:r], rate)
        feature = pitch_feature(sig[l:r], rate)
        p.append((feature[0],feature[-1]))
        X.append(feature)
        labels.append(label)
        if cnt == 100: break
    scatter(p,labels,'111',True)


    cnt = 0
    p = []
    for idx, l, (sig, rate), label, filename in val_itr:
        reg = re.compile('(\d+)-(\d+)-(\d+).wav')
        audio = reg.match(filename).group(2)
        person = reg.match(filename).group(1)
        if audio not in ['00','01']:
            continue
        print(person,label,filename)
        cnt += 1
        l ,r = basic_endpoint_detection(sig, rate)
        #sig = preemphasis(sig, coeff=0.97)
        frames = to_frames(sig[l:r], rate)
        feature = pitch_feature(sig[l:r], rate)
        X_test.append(feature)
        labels_test.append(label)
        p.append((feature[0], feature[-1]))
        if cnt == 100:
            break
    fs = open('models/scale.pkl','wb')
    fc = open('models/rfc.pkl','wb')


    scaler.fit(X)
    X = scaler.transform(X)
    X_test = scaler.transform(X_test)

    #clf = RandomForestClassifier()
    clf = SVC(kernel='rbf')
    #clf = GradientBoostingClassifier()
    clf.fit(X, labels)

    pickle.dump(clf, fc)
    pickle.dump(scaler, fs)

    pred = clf.predict(X_test)
    scatter(p, labels_test, '111', True)
    acc = accuracy_score(labels_test, pred)
    print(acc)

