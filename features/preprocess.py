"""

Author: AuCson
Date: 2018.05.20
Preprocess the signal

"""

import numpy as np

def preemphasis(signal, coeff=0.95):
    """
    preemphasis on the input signal.
    
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])