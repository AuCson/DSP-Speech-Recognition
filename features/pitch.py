"""

Author: Xisen Jin
Date: 2018.05.24

Pitch detection using cesptrum
"""
import numpy as np
from preprocess import preemphasis
from sigproc import to_frames, window
from reader import Reader
from plotter import plot_frame, show
from endpoint import basic_endpoint_detection
import re

def pitch_detect(sig, rate, winlen=0.04, step=0.010):
    sig = downsampling(sig, rate, 10000)
    frames = to_frames(sig, 10000, winlen, step)
    l = []
    prev = 0
    flg = False
    for frame in frames:
        pitch = pitch_detect_frame(frame, 10000)
        if pitch - prev < 50 or not flg:
            l.append(pitch)
            flg = True
            prev = pitch
        else:
            l.append(prev)
            flg = False
    print(l)
    return l

def downsampling(sig, src_rate, dst_rate):
    cnt = -1
    s = []
    for i in range(len(sig)):
        if i * dst_rate / src_rate > cnt + 1e-8:
            cnt += 1
            s.append(sig[i])
    return np.array(s)

def pitch_detect_frame(frame, rate):
    # add hamming window
    #frame = frame / np.max(frame)
    #plot_frame(np.abs(frame), where='411')
    frame = window(frame, rate, 1000, 512, 'square')
    Xw = np.fft.fft(frame, len(frame))
    log_Xw = np.log(np.abs(Xw))
    log_sig = np.fft.ifft(log_Xw, len(frame))
    #plot_frame(np.abs(frame), where='412')
    #plot_frame(np.abs(Xw), where='413')
    #plot_frame(np.abs(log_Xw), where='413')
    v, idx = find_peak(np.abs(log_sig[:len(log_sig) // 2]))
    #plot_frame(np.abs(log_sig), where='414')
    #show()
    #print(idx)
    p = 1 / (0.04 / 512 * idx)
    return p

def find_peak(sig):
    l = []
    idx = []
    for i in range(20,100):
        t = list(sig[i-3:i+3])
        flg = True
        for j in t:
            if sig[i] < j:
                flg = False
        if flg:
            l.append(sig[i])
            idx.append(i)
    #mu = np.mean(l)
    #sigma = np.std(l)
    #print(mu,sigma,max(l),idx[np.argmax(l)])
    if l:
        return max(l), idx[np.argmax(l)]
    else:
        return -1,-1

if __name__ == '__main__':
    r = Reader(debug=False)
    itr = r.iterator(r.val_person)
    cnt = 0
    for idx, l, (sig, rate), label, filename in itr:
        reg = re.compile('(\d+)-(\d+)-(\d+).wav')
        audio = reg.match(filename).group(2)
        person = reg.match(filename).group(1)
        if audio not in ['00','01'] or person != '15307130334':
            continue
        print(person,label)
        cnt += 1
        #if cnt < 9:
        #    continue
        #sig = preemphasis(sig)
        l ,r = basic_endpoint_detection(sig, rate)
        frames = to_frames(sig[l:r], rate)
        pitch = pitch_detect(sig[l:r], rate)

        #plot_frame(frames,where='211')
        plot_frame(pitch,where='212',show=True)
