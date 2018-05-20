"""

Author: Xisen Jin
Date: 2018.05.20
Purpose: Endpoint detection algorithms

"""
from sigproc import *
from plotter import plot_frame
import numpy as np
from reader import Reader

def basic_endpoint_detection(frames):
    """
    Basic endpoint detection based on energy and zpr. 
    :param frames: list of 1-d numpy array
    :return: 
    """
    energy = get_amplitude(frames, 'dirc')
    energy = [np.mean(_) for _ in energy]
    plot_frame(energy,where='212')


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
    return energy

if __name__ == '__main__':
    r = Reader()
    itr = r.iterator(r.train)
    for (sig, rate),label in itr:
        plot_frame(sig,where='211',show=False)
        print(label)
        frames = to_frames(sig, rate)
        basic_endpoint_detection(frames)