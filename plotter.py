"""

Author: Xisen Jin
Date: 2018.05.20
Purpose: Signal Plotter. 

"""
from matplotlib import pyplot as plt
import numpy as np

PLT_CNT = 0

def plot_frame(frames,frame_len=1,name='frames',where='111',show=False, sep=None, filename=None):
    x = np.arange(0, len(frames), frame_len)
    plt.subplot(where)
    plt.title(name)
    plt.plot(x, frames)
    if sep:
        for c in sep:
            plt.axvline(c)
    if show:
        plt.show(f=filename)

def show(save=False, f=None):
    #plt.show()
    global  PLT_CNT
    if save:
        plt.savefig('./imgs/%s' % str(f))
        PLT_CNT += 1
        plt.clf()