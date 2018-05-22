"""

Author: Xisen Jin
Date: 2018.05.20
Purpose: Signal Plotter. 

"""
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm

PLT_CNT = 0

def plot_frame(frames,frame_len=1,name='frames',where='111',show=False, sep=None):
    x = np.arange(0, len(frames), frame_len)
    plt.subplot(where)
    plt.title(name)
    plt.plot(x, frames)
    if sep:
        for c in sep:
            plt.axvline(c)
    if show:
        plt.show()

def plot_mfcc(mfcc):
    fig, ax = plt.subplots()
    mfcc_data = np.swapaxes(mfcc, 0, 1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    ax.set_title('MFCC')
    plt.show()

def show(save=False, f=None):
    if not save:
        plt.show()
    global  PLT_CNT
    if save:
        plt.savefig('./imgs/%s' % str(f))
        PLT_CNT += 1
        plt.clf()