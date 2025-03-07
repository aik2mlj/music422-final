"""
Music 422 Marina Bosi

- mdct.py -- Computes a reasonably fast MDCT/IMDCT using the FFT/IFFT

-----------------------------------------------------------------------
© 2009-2025 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------

"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np
import scipy
import time


### Problem 1.a ###
def MDCTslow(data, a, b, isInverse=False):
    """
    Slow MDCT algorithm for window length a+b following pp. 130 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    and where the 2/N factor is included in the forward transform instead of inverse.
    a: left half-window length
    b: right half-window length
    """

    ### YOUR CODE STARTS HERE ###
    N = a + b
    n0 = (b + 1) / 2
    n_plus_n0 = np.arange(0, N) + n0
    k_plus_half = np.arange(0, N // 2) + 0.5
    cos_matrix = np.cos((2 * np.pi / N) * np.outer(n_plus_n0, k_plus_half))  # (N,N/2)

    if isInverse:
        summand = cos_matrix * data
        x = 2 * np.sum(summand, axis=1)
        return x
    else:
        summand = cos_matrix.T * data
        X = (2 / N) * np.sum(summand, axis=1)
        return X
    ### YOUR CODE ENDS HERE ###


### Problem 1.c ###
def MDCT(data, a, b, isInverse=False):
    """
    Fast MDCT algorithm for window length a+b following pp. 141-143 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    and where the 2/N factor is included in forward transform instead of inverse.
    a: left half-window length
    b: right half-window length
    """

    ### YOUR CODE STARTS HERE ###
    N = a + b
    n0 = (b + 1) / 2
    n_array = np.arange(N)
    k_array = np.arange(N // 2)

    if isInverse:
        data_pre_twd = np.concat((data, -np.flip(data)))
        data_pre_twd = data_pre_twd * np.exp(1j * 2 * np.pi * n_array * n0 / N)
        data_ifft = scipy.fft.ifft(data_pre_twd)
        data_post_twd = np.real(data_ifft * np.exp(1j * np.pi * (n_array + n0) / N)) * N

    else:
        data_pre_twd = data * np.exp(-1j * np.pi * n_array / N)
        data_fft = scipy.fft.fft(data_pre_twd)
        data_post_twd = np.real(
            (2 / N) * data_fft[: N // 2] * np.exp(-1j * 2 * np.pi * n0 * (k_array + 0.5) / N)
        )

    return data_post_twd
    ### YOUR CODE ENDS HERE ###


def IMDCT(data, a, b):
    ### YOUR CODE STARTS HERE ###
    return MDCT(data, a, b, isInverse=True)
    ### YOUR CODE ENDS HERE ###


# -----------------------------------------------------------------------------


def compare_time():
    print("Comparing the execution time of MDCT/IMDCT implementations...")
    x = np.random.uniform(-1, 1, size=(2048,))

    start_t = time.perf_counter()
    y = MDCTslow(x, 1024, 1024)
    mdct_slow_dur = time.perf_counter() - start_t
    print("Slow MDCT:\t", mdct_slow_dur)

    start_t = time.perf_counter()
    MDCT(x, 1024, 1024)
    mdct_dur = time.perf_counter() - start_t
    print("FFT MDCT:\t", mdct_dur)

    start_t = time.perf_counter()
    MDCTslow(y, 1024, 1024, isInverse=True)
    imdct_slow_dur = time.perf_counter() - start_t
    print("Slow IMDCT:\t", imdct_slow_dur)

    start_t = time.perf_counter()
    IMDCT(y, 1024, 1024)
    imdct_dur = time.perf_counter() - start_t
    print("FFT IMDCT:\t", imdct_dur)

    print("Speedup ratio:", (imdct_slow_dur + mdct_slow_dur) / (mdct_dur + imdct_dur))


# Testing code
if __name__ == "__main__":
    ### YOUR TESTING CODE STARTS HERE ###
    xs = [0, 1, 2, 3, 4, 4, 4, 4, 3, 1, -1, -3]
    a = 4
    b = 4
    N = a + b
    xs_pad = np.concat((xs, np.zeros(b)))
    xs_pad = np.concat((np.zeros(a), xs_pad))
    # print(xs)

    idx = a  # starting from 4
    last_half_recon = np.zeros(N // 2)
    output = []
    while idx + b <= len(xs_pad):
        x = xs_pad[idx - a : idx + b]
        print(x)

        mdct_x = MDCTslow(x, a, b, isInverse=False)
        mdct_x_fft = MDCT(x, a, b, isInverse=False)
        assert np.allclose(mdct_x, mdct_x_fft, atol=1e-10)
        x_prime = MDCTslow(mdct_x, a, b, isInverse=True)
        x_prime_fft = IMDCT(mdct_x, a, b)
        assert np.allclose(x_prime, x_prime_fft, atol=1e-10)

        x_prime /= 2.0
        x_recon = last_half_recon + x_prime[: N // 2]
        last_half_recon = x_prime[N // 2 :]
        if np.allclose(x_recon, x[: N // 2], atol=1e-10):
            print("reconstruction success!")
        else:
            print("recon:", x_recon)
            print("wrong!")
        output.append(x_recon)
        idx += b
    output = np.concat(output[1:])
    assert np.allclose(output, xs, atol=1e-10)

    compare_time()
    ### YOUR TESTING CODE ENDS HERE ###
