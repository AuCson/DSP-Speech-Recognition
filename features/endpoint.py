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

def basic_endpoint_detection(frames, rate):
    """
    Basic endpoint detection based on energy and zpr. 
    :param frames: list of 1-d numpy array
    :return: 
    """
    amp = get_amplitude(frames, 'dirc')
    sep_point = amplitude_rule(amp)
    left, right = sep_point[0][0], sep_point[-1][1]
    if right - left < 50: # too short, 0.5 sec
        sep_point = amplitude_rule(amp, 0.125, 0.100, 0.100)

    p = []
    for item in sep_point:
        p.append(item[0])
        p.append(item[1])
    plot_frame(amp,where='312',sep=p)

    zpr = get_zpr(frames)

    plot_frame(zpr, where='313')


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


def amplitude_rule(amp, mh=0.25, th=0.100, sil=0.100):
    """
    M_H: high threshold, MAX * 0.25
    M_L: low threshold, MU(SIL) + 3 * SIGMA(SIL)
    T_H: threshold for v > M_H. 100ms
    :param amp: 
    :return: 
    """
    p = []
    # assume first and ast 100ms is silience
    sil = amp[:int(sil / cfg.frame)] + amp[-int(sil/cfg.frame):]
    sil = sorted(sil)[:-2] # get rid of extreme points
    s_mean, s_sigma = np.mean(sil), np.std(sil)
    T_H = th / cfg.frame
    M_L = s_mean + 3 * s_sigma
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
    return p


def get_zpr(frames):
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


def zpr_rule(zpr, left, right):
    pass

if __name__ == '__main__':
    r = Reader(debug=True)
    itr = r.iterator(r.train)
    for (sig, rate),label, filename in itr:
        plot_frame(sig,where='311',show=False, filename=filename)
        print(label)
        sig = preemphasis(sig)
        frames = to_frames(sig, rate)
        basic_endpoint_detection(frames, rate)
        show(True,f=filename[:-4])