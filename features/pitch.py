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

def downsampling(sig, src_rate, dst_rate):
    cnt = -1
    s = []
    for i in range(len(sig)):
        if i * dst_rate / src_rate > cnt + 1e-8:
            cnt += 1
            s.append(sig[i])
    return np.array(s)

def pitch_detect(sig, rate, winlen=0.04, step=0.005):
    sig = downsampling(sig, rate, 10000)
    frames = to_frames(sig, 10000, winlen, step)
    g = []
    g_idx = []
    for frame in frames:
        scores,scores_idx = pitch_detect_frame(frame, 10000)
        g.append(scores)
        g_idx.append(scores_idx)
    #for l in g:
    #    print(l)
    pitch = simple(g, g_idx)
    return pitch

def simple(g,g_idx):
    pitch = []
    for l,d in zip(g,g_idx):
        idx = d[np.argmax(l)]
        v = max(l)
        if v > 150:
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
    for i in range(30,100):
        p = i
        while p > 0 and sig[p] <= sig[i]:
            p-=1
        q = i
        while q < len(sig) and sig[q] <= sig[i]:
            q+=1
        l.append(q-p)
        idx.append(i)
    return l, idx

def find_smooth_subsequence(pitch, tor=3):
    def slope(seq):
        x = np.arange(0, len(seq))
        z = np.polyfit(x, seq, 1)
        return z[0]

    def meanshift(seq1, seq2):
        return np.mean(seq2) - np.mean(seq1)

    i = 0
    strs = []
    str_idx = []
    while i < len(pitch):
        j = i+1
        prev = pitch[i]
        k = tor
        seg = [pitch[i]]
        while j < len(pitch):
            if abs(pitch[j] - prev) > 50:
                k -= 1
            else:
                seg.append(pitch[j])
                prev = pitch[j]
            if not k:
                strs.append(seg)
                str_idx.append((i,j))
                break
            j += 1
        if j == len(pitch):
            strs.append(seg)
            str_idx.append((i, j))
            break
        i = j - tor + 1

    obj = sorted(zip(strs,str_idx), key=lambda x:-len(x[0]))
    strs, str_idx = tuple(zip(*obj))

    xi, yi = str_idx[0], str_idx[1]
    x,y = strs[0], strs[1]
    if xi[0] > yi[0]:
        x,y = y,x
        xi,yi = yi,xi
    print(xi)
    print(yi)
    print(meanshift(x,y))
    print(slope(x),slope(y))
    return meanshift(x,y), slope(x), slope(y)


if __name__ == '__main__':
    r = Reader(debug=False)
    itr = r.iterator(r.train)
    cnt = 0
    p = []
    labels = []
    for idx, l, (sig, rate), label, filename in itr:
        reg = re.compile('(\d+)-(\d+)-(\d+).wav')
        audio = reg.match(filename).group(2)
        person = reg.match(filename).group(1)
        if audio not in ['00','01']:
            continue
        print(person,label,filename)
        cnt += 1
        l ,r = basic_endpoint_detection(sig, rate)
        frames = to_frames(sig[l:r], rate)
        pitch = pitch_detect(sig[l:r], rate)
        meanshift, slope_a, slope_b = find_smooth_subsequence(pitch)
        p.append((slope_a, meanshift))
        labels.append(label)
        #plot_frame(pitch,where='111',show=True)
        if cnt == 100:
            break
    scatter(p,labels,'111',True)