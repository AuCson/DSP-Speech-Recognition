"""

Author: Xisen Jin
Date: 2018.05.20
Purpose: Signal Plotter. 

"""
from matplotlib import pyplot as plt
import numpy as np

def plot_frame(frames,frame_len=1,name='frames',where='111',show=True):
    x = np.arange(0, len(frames), frame_len)
    plt.subplot(where)
    plt.title(name)
    plt.plot(x, frames)
    if show:
        plt.show()