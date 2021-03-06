"""

Author: Xisen Jin
Date: 2018.05.20
Purpose: Signal Plotter. 

"""
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm

PLT_CNT = 0

def plot_frame(frames,frame_len=1,name='frames',where='111',show=False, sep=None, bias=0):
    x = np.arange(bias, len(frames)+bias, frame_len)
    plt.subplot(where)
    plt.title(name)
    plt.plot(x, frames, c='orange')
    if sep:
        for c in sep:
            plt.axvline(c)
    if show:
        plt.show()

def plot_mfcc(mfcc, where, show=False):
    plt.subplot(where)
    mfcc_data = np.swapaxes(mfcc, 0, 1)
    plt.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    if show:
        plt.show()

def scatter(points, label, where, show=False):
    plt.subplot(where)
    if len(points) == 2:
        plt.scatter(points[0],points[1],c=label)
    else:
        x = [_[0] for _ in points]
        y = [_[1] for _ in points]
        plt.scatter(x,y,c=label)
    if show:
        plt.show()

def show(save=False, f=None):
    if not save:
        plt.show()
    global  PLT_CNT
    if save:
        plt.savefig('./imgs/%s' % str(f))
        PLT_CNT += 1
        plt.clf()