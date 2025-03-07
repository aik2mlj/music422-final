"""

Music 422  Marina Bosi

window.py -- Defines functions to window an array of discrete-time data samples

-----------------------------------------------------------------------
© 2009-2025 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------


"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np
import scipy


### Problem 1.d ###
def SineWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray sine-windowed
    Sine window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    ### YOUR CODE STARTS HERE ###
    N = len(dataSampleArray)
    w = np.sin(np.pi * (np.arange(N) + 0.5) / N)
    return w * dataSampleArray
    ### YOUR CODE ENDS HERE ###


def HanningWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray Hanning-windowed
    Hann window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    ### YOUR CODE STARTS HERE ###
    N = len(dataSampleArray)
    w = 0.5 * (1 - np.cos(2 * np.pi * (np.arange(N) + 0.5) / N))
    return w * dataSampleArray
    ### YOUR CODE ENDS HERE ###


### Problem 1.d - OPTIONAL ###
def KBDWindow(dataSampleArray, alpha=4.0):
    """
    Returns a copy of the dataSampleArray KBD-windowed
    KBD window is defined following the KDB Window handout in the
        Canvas Files/Assignments/HW3 folder
    """

    ### YOUR CODE STARTS HERE ###
    N = len(dataSampleArray)
    N_half = N // 2
    numerator = (
        np.pi * alpha * np.sqrt(1.0 - np.pow((2 * np.arange(N_half + 1)) / (N_half + 1) - 1, 2))
    )
    numerator = scipy.special.i0(numerator)
    denominator = scipy.special.i0(np.pi * alpha)
    w_kb = numerator / denominator
    w_kb_cumsum = np.cumsum(w_kb)
    w_kbd_half = np.sqrt(w_kb_cumsum[:-1] / w_kb_cumsum[-1])
    w_kbd = np.concat((w_kbd_half, np.flip(w_kbd_half)))
    return w_kbd * dataSampleArray
    ### YOUR CODE ENDS HERE ###


# -----------------------------------------------------------------------------

# Testing code
if __name__ == "__main__":
    ### YOUR TESTING CODE STARTS HERE ###
    # see plots.ipynb for 1.f)
    pass
    ### YOUR TESTING CODE ENDS HERE ###
