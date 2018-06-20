"""

Author: Xisen Jin
Date: 2018.05.20
Purpose: Endpoint detection algorithms

"""
from features.sigproc import *
from plotter import plot_frame, show
import numpy as np
from reader import Reader
from features.preprocess import preemphasis
from config import cfg, meta

def max_pitch(l, rate, bias=20):
    idx = bias + np.argmax(l)
    p = 1 / (1.0 / rate * idx)
    return p

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

noise_rec = []

def basic_endpoint_detection(sig, rate, return_feature=False):
    """
    Basic endpoint detection based on energy and zpr. 
    :param frames: list of 1-d numpy array
    :return: 
    """
    frames = to_frames(sig, rate, t=cfg.frame, step=cfg.step)
    amp = get_amplitude(frames)
    sep_point = amplitude_rule(amp)
    left, right = sep_point[0][0], sep_point[-1][1]
    if right - left < 50: # too short, 0.5 sec
        sep_point = amplitude_rule(amp, 0.125)
    #if right - left > 100: # too long
    #    sep_point = amplitude_rule(amp, 0.25, 0.100, left * cfg.frame, (len(frames) - right) * cfg.frame)

    left, right = sep_point[0][0], sep_point[-1][1]
    p = []
    for item in sep_point:
        p.append(item[0])
        p.append(item[1])
    #plot_frame(amp,where='312',sep=p)
    zcr = get_zcr(frames)
    left2, right2 = zcr_rule(zcr, left, right)
    #plot_frame(zcr, where='313')
    #plot_frame(amp, where='111', sep=[left,right,left2,right2],show=True)

    if right2 - left2 < 50:
        left2 = 0
        right2 = len(frames)
    if not return_feature:
        return int(left2 * cfg.step * rate),  int(right2 * cfg.step * rate)
    else:
        return int(left2 * cfg.step * rate),  int(right2 * cfg.step * rate), amp, zcr

def robust_endpoint_detection(sig, rate):
    # filter: 50 - 900 Hz

    frames = to_frames(sig, rate, cfg.frame, step=cfg.step)
    amp = get_amplitude(frames)
    sep_point = amplitude_rule(amp, 0.5, frames=frames, use_acr=True, rate=rate)
    left, right = sep_point[0][0], sep_point[-1][1]
    #if right - left > 100: # too long
    #    sep_point = amplitude_rule(amp, 0.125, frames=frames, use_acr=True, rate=rate)

    left, right = sep_point[0][0], sep_point[-1][1]
    p = []
    for item in sep_point:
        p.append(item[0])
        p.append(item[1])
    #plot_frame(amp,where='312',sep=p)
    zcr = get_zcr(frames)
    left2, right2 = zcr_rule(zcr, left, right)
    #plot_frame(zcr, where='313')
    #plot_frame(amp, where='111', sep=[left, right, left2, right2], show=True)

    if right2 - left2 < 50:
        left2 = 0
        right2 = len(frames)
    return int(left2 * cfg.step * rate),  int(right2 * cfg.step * rate)

def get_noise(amp, sep_point):
    MAX_NOISE = 1e30
    if sep_point[0] == (0, len(amp)):
        return MAX_NOISE
    left = 0
    noise = 0
    l = 0
    for item in sep_point:
        noise += np.sum(amp[left:item[0]])
        l += item[0] - left
        left = item[1]
    noise += np.sum(amp[left:])
    l += len(amp) - left
    return noise / l

def get_amplitude(frames, window='square', use_sq=False):
    """
    calculate convolution between |x| or x(n)^2 and the window
    :param use_sq: use x^2 instead of x
    :param frames: 
    :return: 
    """
    energy = []
    l = frames[0].shape[-1]
    if window == 'square':
        window = np.ones(1)
    elif window == 'hamming':
        window = np.hamming(l)
    for frame in frames:
        frame = np.abs(frame) if not use_sq else np.square(frame)
        energy.append(np.convolve(frame, window, 'same'))
    energy = [np.mean(_) for _ in energy]
    return energy

def amplitude_feature(sig, rate, winlen, step):
    frames = to_frames(sig, rate, winlen, step)
    amp = get_amplitude(frames)
    return amp

def amplitude_rule(amp, mh=0.25, th=0.100, l_sil=0.100, r_sil=0.100, sigma=3, use_acr=False, frames=None, rate=None):
    """
    M_H: high threshold, MAX * 0.25
    M_L: low threshold, MU(SIL) + 3 * SIGMA(SIL)
    T_H: threshold for v > M_H. 100ms
    :param amp: 
    :return: 
    """

    def acr_rule(frame):
        acrs = [acr(frame,n) for n in range(rate//500, rate//50)]
        return max(acrs) / acr(frame,0) > 0.55

    #pitch = [acr_rule(frame) for frame in frames]
    #plot_frame(pitch, where='211', show=False)

    p = []
    # assume first and ast 100ms is silience
    sil = amp[:int(l_sil / cfg.step)] + amp[-int(r_sil/cfg.step):]
    sil = sorted(sil)[:-2] # get rid of extreme points
    s_mean, s_sigma = np.mean(sil), np.std(sil)
    T_H = th / cfg.frame
    M_L = s_mean + sigma * s_sigma
    M_H = max(np.max(amp) * mh, M_L)
    #print(M_H,M_L)
    i = 0

    while i < len(amp):
        if amp[i] >= M_H:
            j = k = i
            while k < len(amp) and amp[k] > M_H:
                k += 1
            if k - j < T_H:
                i = k
            else:
                while j > 0 and amp[j] > M_L and (not use_acr or acr_rule(frames[j])):
                    j -= 1
                while k < len(amp) and amp[k] > M_L  and (not use_acr or acr_rule(frames[k])):
                    k += 1
                #if use_acr and j > 5:
                #    j -= 5
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
    #frames = frames.tolist()
    for frame in frames:
        c = 0
        arr1 = frame[:-1] * frame[1:] < 0
        #for i in range(len(frame)-1):
        #    if frame[i] > 0 and frame[i+1] < 0 or frame[i] < 0 and frame[i+1] > 0:
        #        c += 1
        c = (arr1 == True).sum()
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
    sil = zcr[:int(l_sil / cfg.step)] + zcr[-int(r_sil / cfg.step):]
    mu, sigma = np.mean(sil), np.std(sil)
    thres = mu + 3 * sigma
    j = left
    while j > 0 and left - j <= max_shift and zcr[j] > thres:
        j -= 1
    k = right
    while k < len(zcr) and k - right <= max_shift and zcr[k] > thres:
        k += 1
    return j,k

if __name__ == '__main__':
    r = Reader(debug=False)
    itr = r.iterator(r.train)
    for idx, l, (sig, rate), label, filename in itr:
        #plot_frame(sig,where='311',show=False)
        print(label, filename)
        #sig = preemphasis(sig)
        robust_endpoint_detection(sig, rate)
        #show(True,f=filename[:-4])
        if idx == 100: break

    #print(noise_rec)
    #plot_frame(noise_rec, where='111',show=True)