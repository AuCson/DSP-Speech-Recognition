"""

Author: Xisen Jin
Date: 2018.05.20
Purpose: Endpoint detection algorithms

"""
from sigproc import *
from plotter import plot_frame, show
import numpy as np
from reader import Reader
from preprocess import preemphasis
from config import cfg, meta

def basic_endpoint_detection(sig, rate):
    """
    Basic endpoint detection based on energy and zpr. 
    :param frames: list of 1-d numpy array
    :return: 
    """
    frames = to_frames(sig, rate, t=cfg.frame, step=cfg.step)
    amp = get_amplitude(frames, 'dirc')
    sep_point = amplitude_rule(amp)
    left, right = sep_point[0][0], sep_point[-1][1]
    if right - left < 50: # too short, 0.5 sec
        sep_point = amplitude_rule(amp, 0.125)
    if right - left > 100: # too long
        sep_point = amplitude_rule(amp, 0.25, 0.100, left * cfg.frame, (len(frames) - right) * cfg.frame)

    left, right = sep_point[0][0], sep_point[-1][1]
    p = []
    for item in sep_point:
        p.append(item[0])
        p.append(item[1])
    #plot_frame(amp,where='312',sep=p)
    zcr = get_zcr(frames)
    left2, right2 = zcr_rule(zcr, left, right)
    #plot_frame(zcr, where='313')
    #plot_frame(amp, where='312', sep=[left,right,left2,right2], show=True)

    return int(left2 / cfg.step), int(right2 / cfg.step)

def get_amplitude(frames, window='dirc', use_sq=False):
    """
    calculate convolution between |x| or x(n)^2 and the window
    :param use_sq: use x^2 instead of x
    :param frames: 
    :return: 
    """
    energy = []
    l = frames[0].shape[-1]
    if window == 'dirc':
        window = np.ones(1)
    elif window == 'hamming':
        window = np.hamming(l)
    for frame in frames:
        frame = np.abs(frame) if not use_sq else np.square(frame)
        energy.append(np.convolve(frame, window, 'same'))
    energy = [np.mean(_) for _ in energy]
    return energy


def amplitude_rule(amp, mh=0.25, th=0.100, l_sil=0.100, r_sil=0.100, sigma=3):
    """
    M_H: high threshold, MAX * 0.25
    M_L: low threshold, MU(SIL) + 3 * SIGMA(SIL)
    T_H: threshold for v > M_H. 100ms
    :param amp: 
    :return: 
    """
    p = []
    # assume first and ast 100ms is silience
    sil = amp[:int(l_sil / cfg.frame)] + amp[-int(r_sil/cfg.frame):]
    sil = sorted(sil)[:-2] # get rid of extreme points
    s_mean, s_sigma = np.mean(sil), np.std(sil)
    T_H = th / cfg.frame
    M_L = s_mean + sigma * s_sigma
    M_H = max(np.max(amp) * mh, M_L)
    print(M_H,M_L)
    i = 0
    while i < len(amp):
        if amp[i] >= M_H:
            j = k = i
            while amp[k] > M_H and k < len(amp):
                k += 1
            if k - j < T_H:
                i = k
            else:
                while amp[j] > M_L and j >= 0:
                    j -= 1
                while amp[k] > M_L and k < len(amp):
                    k += 1
                p.append((j,k))
                i = k
        i += 1
    if not p:
        return [(0,len(amp))]
    return p


def get_zcr(frames):
    """
    calculate zero pass rate
    :param frames: 
    :return: 
    """
    zpr = []
    for frame in frames:
        c = 0
        for i in range(len(frame)-1):
            if frame[i] * frame[i+1] < 0:
                c += 1
        zpr.append(c)
    return zpr


def zcr_rule(zcr, left, right, max_shift=0.400, l_sil=0, r_sil=0.100):
    """
    renew the left and right endpoint according to the zero cross rate.
    use 3 sigma principle
    :param zcr: 
    :param left: 
    :param right: 
    :return: 
    """
    max_shift /= cfg.frame
    sil = zcr[:int(l_sil / cfg.frame)] + zcr[-int(r_sil / cfg.frame):]
    mu, sigma = np.mean(sil), np.std(sil)
    thres = mu + 3 * sigma
    j = left
    while j >= 0 and left - j <= max_shift and zcr[j] > thres:
        j -= 1
    k = right
    while k < len(zcr) and k - right <= max_shift and zcr[k] > thres:
        k += 1
    return j,k

if __name__ == '__main__':
    r = Reader(debug=False)
    itr = r.iterator(r.train)
    for idx, l, (sig, rate), label, filename in itr:
        plot_frame(sig,where='311',show=False)
        print(label)
        #sig = preemphasis(sig)

        basic_endpoint_detection(sig, rate)
        show(True,f=filename[:-4])