"""
psychoac.py -- masking models implementation

-----------------------------------------------------------------------
(c) 2011-2025 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np
import scipy
from window import HanningWindow
from rotation import apply_rotation


def intensity_from_DFT(X, gain_window=3 / 8):
    # get the intensity from a DFT result
    # window defaults to hanning window
    N = len(X)
    return 4.0 / (N * N * gain_window) * np.pow(np.abs(X), 2)


def intensity_from_MDCT(X, gain_window=1 / 2):
    # get the intensity from a MDCT result
    # window defaults to sine / KBD window
    return 2.0 / (gain_window) * np.pow(np.abs(X), 2)


def SPL(intensity):
    """
    Returns the SPL corresponding to intensity
    """
    # This does not take care of the 2/N in MDCT
    return np.maximum(96 + 10.0 * np.log10(intensity + np.finfo(float).eps), -30.0)


def Intensity(spl):
    """
    Returns the intensity for SPL spl
    """
    # This does not take care of the 2/N in MDCT
    return np.power(10, (spl - 96) / 10.0)


def Thresh(f):
    """Returns the threshold in quiet measured in SPL at frequency f (in Hz)"""

    f_floored = np.maximum(f, 20.0)
    f_khz = f_floored / 1000.0
    return (
        3.64 * np.pow(f_khz, -0.8)
        - 6.5 * np.exp(-0.6 * np.pow(f_khz - 3.3, 2))
        + 0.001 * np.pow(f_khz, 4)
    )


def Bark(f):
    """Returns the bark-scale frequency for input frequency f (in Hz)"""
    f_khz = np.array(f) / 1000.0
    return 13 * np.arctan(0.76 * f_khz) + 3.5 * np.arctan(np.pow(f_khz / 7.5, 2))


def BMLD_correction(freq):
    """
    Returns the BMLD correction (in dB) for a given frequency. Used for side channel maskers.
    At low frequencies (<250 Hz), the binaural masking level difference can be up to ~15 dB.
    For frequencies above 1500 Hz and below 4khz, the effect is stabilised (2.5 dB correction).
    For intermediate frequencies, we apply a linear interpolation.
    """
    if freq < 0:
        raise ValueError("Frequency must be positive.")
    if freq <= 250:
        return 15.0
    elif freq <= 1500:
        # Linearly interpolate between 15 dB and 2.5 dB
        return 15.0 - (15.0 - 2.5) * (freq - 250) / (1500 - 250)
    elif freq <= 4000:
        return 2.5
    else:
        return 0.0


class Masker:
    """
    a Masker whose masking curve drops linearly in Bark beyond 0.5 Bark from the
    masker frequency
    """

    def __init__(self, f, SPL, isTonal=True, isSide=False):
        """
        initialized with the frequency and SPL of a masker and whether or not
        it is Tonal
        """
        self.f = f
        self.z = Bark(f)
        self.spl = SPL
        self.isTonal = isTonal
        self.delta = 16 if isTonal else 6

        # if the masker is a side tone, apply BMLD correction
        if isSide:
            self.delta += BMLD_correction(f)

    def IntensityAtFreq(self, freq):
        """The intensity at frequency freq"""
        return self.IntensityAtBark(Bark(freq))

    def vIntensityAtFreq(self, freqVec):
        """The intensity at frequency freq"""
        return self.vIntensityAtBark(Bark(freqVec))

    def IntensityAtBark(self, z):
        """The intensity at Bark location z"""
        spl = self.spl - self.delta
        dz = z - self.z
        if dz < -0.5:
            spl += -27 * (abs(dz) - 0.5)
        elif dz > 0.5:
            spl += (-27 + 0.367 * max(self.spl - 40.0, 0.0)) * (abs(dz) - 0.5)
        return Intensity(spl)

    def vIntensityAtBark(self, zVec):
        """The intensity at vector of Bark locations zVec"""
        splVec = np.zeros_like(zVec)
        splVec += self.spl - self.delta
        dzVec = zVec - self.z
        splVec += np.where(dzVec < -0.5, -27 * (np.abs(dzVec) - 0.5), 0)
        splVec += np.where(
            dzVec > 0.5,
            (-27 + 0.367 * max(self.spl - 40.0, 0.0)) * (np.abs(dzVec) - 0.5),
            0,
        )
        return Intensity(splVec)


# Default data for 25 scale factor bands based on the traditional 25 critical bands
cbFreqLimits = [100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480,
           1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]  # fmt: skip


def AssignMDCTLinesFromFreqLimits(nMDCTLines, sampleRate, flimit=cbFreqLimits):
    """
    Assigns MDCT lines to scale factor bands for given sample rate and number
    of MDCT lines using predefined frequency band cutoffs passed as an array
    in flimit (in units of Hz). If flimit isn't passed it uses the traditional
    25 Zwicker & Fastl critical bands as scale factor bands.
    """
    freqVec = np.arange(nMDCTLines) * sampleRate / (2 * nMDCTLines)
    indices = np.searchsorted(flimit, freqVec, side="right")  # left-close, right-open
    counts = np.bincount(indices, minlength=len(flimit) + 1)
    return counts


class ScaleFactorBands:
    """
    A set of scale factor bands (each of which will share a scale factor and a
    mantissa bit allocation) and associated MDCT line mappings.

    Instances know the number of bands nBands; the upper and lower limits for
    each band lowerLimit[i in range(nBands)], upperLimit[i in range(nBands)];
    and the number of lines in each band nLines[i in range(nBands)]
    """

    def __init__(self, nLines):
        """
        Assigns MDCT lines to scale factor bands based on a vector of the number
        of lines in each band
        nlines: return value of AssignMDCTLinesFromFreqLimits
        """
        self.nBands = len(nLines)
        self.lowerLine = np.concat(([0], np.cumsum(nLines)[:-1]))
        self.upperLine = np.cumsum(nLines) - 1
        self.nLines = nLines
        assert (self.nLines == self.upperLine - self.lowerLine + 1).any()


def identifyMaskers(data, sampleRate, sfBands):
    """
    Identify the maskers from given time-domain samples data
    return: (tonal_maskers, noise_maskers), each a `np.array` of (frequency, intensity)
    """
    N = len(data)
    # Hanning window first
    data_windowed = HanningWindow(data)
    # FFT
    spectrum = scipy.fft.rfft(data_windowed)[:-1]

    return identifyMaskers_from_spectrum(spectrum, N, sampleRate, sfBands)


def getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Return Masked Threshold evaluated at MDCT lines.

    Used by CalcSMR, but can also be called from outside this module, which may
    be helpful when debugging the bit allocation code.
    """
    N = len(data)
    # indentify the maskers
    tonal_maskers, noise_maskers = identifyMaskers(data, sampleRate, sfBands)
    freqs = np.arange(N // 2) / N * sampleRate
    # combine all the masking curves and threshold in quiet to get the masked threshold
    quiet_thresh = Thresh(freqs)
    masked_thresh = quiet_thresh
    for tm_f, tm_intensity in tonal_maskers:
        tm_spl = SPL(tm_intensity)
        masker = Masker(tm_f, tm_spl, isTonal=True)
        masker_intensity = masker.vIntensityAtFreq(freqs)
        masking_curve = SPL(masker_intensity)
        masked_thresh = np.maximum(masked_thresh, masking_curve)  # alpha=inf
    for ns_f, ns_intensity in noise_maskers:
        ns_spl = SPL(ns_intensity)
        masker = Masker(ns_f, ns_spl, isTonal=False)
        masker_intensity = masker.vIntensityAtFreq(freqs)
        masking_curve = SPL(masker_intensity)
        masked_thresh = np.maximum(masked_thresh, masking_curve)  # alpha=inf
    return masked_thresh


def CalcSMRs(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Set SMR for each critical band in sfBands.

    Arguments:
                data:       is an array of N time domain samples
                MDCTdata:   is an array of N/2 MDCT frequency coefficients for the time domain samples
                            in data; note that the MDCT coefficients have been scaled up by a factor
                            of 2^MDCTscale
                MDCTscale:  corresponds to an overall scale factor 2^MDCTscale for the set of MDCT
                            frequency coefficients
                sampleRate: is the sampling rate of the time domain samples
                sfBands:    points to information about which MDCT frequency lines
                            are in which scale factor band

    Returns:
                SMR[sfBands.nBands] is the maximum signal-to-mask ratio in each
                                    scale factor band

    Logic:
                Performs an FFT of data[N] and identifies tonal and noise maskers.
                Combines their relative masking curves and the hearing threshold
                to calculate the overall masked threshold at the MDCT frequency locations.
                                Then determines the maximum signal-to-mask ratio within
                each critical band and returns that result in the SMR[] array.
    """
    masked_thresh = getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands)
    MDCTdata_origin = MDCTdata / (2**MDCTscale)
    MDCTspl = SPL(intensity_from_MDCT(MDCTdata_origin))
    SMR_all = MDCTspl - masked_thresh
    # record the maximum SMR in each sfBand
    SMRs = []
    for lower_l, upper_l in zip(sfBands.lowerLine, sfBands.upperLine):
        SMR_max = np.max(SMR_all[lower_l : upper_l + 1])
        SMRs.append(SMR_max)
    return np.array(SMRs)


def identifyMaskers_from_spectrum(spectrum, N, sampleRate, sfBands):
    freqs = np.arange(N // 2) / N * sampleRate
    # get intensity & SPL
    intensity = intensity_from_DFT(spectrum)

    # get peak indices where potentially tonal maskers locate
    peak_indices = scipy.signal.argrelextrema(intensity, np.greater, order=1)[0]
    # peak_indices = [i for i in peak_indices if intensity[i] > np.mean(intensity)]

    # get tonal maskers
    tonal_maskers = []
    for p in peak_indices:
        # aggregate the intensity values across the peak
        intensity_agg = intensity[p - 1] + intensity[p] + intensity[p + 1]
        # center of mass interpolation for peak frequency estimation
        freq_peak = (
            sampleRate
            / N
            * ((p - 1) * intensity[p - 1] + p * intensity[p] + (p + 1) * intensity[p + 1])
            / intensity_agg
        )
        tonal_maskers.append((freq_peak, intensity_agg))

    # get noise maskers: within each critical band, sum up the intensity excluding the tonal ones.
    noise_maskers = []
    for lower_l, upper_l in zip(sfBands.lowerLine, sfBands.upperLine):
        freq_indices = np.arange(lower_l, upper_l + 1)
        noise_indices = freq_indices
        # remove tonal indices from the noise indices
        for peak in peak_indices:
            noise_indices = noise_indices[noise_indices != peak]
        if len(noise_indices) == 0:
            continue
        # sum up the intensity
        noise_intensity = np.sum(intensity[noise_indices])
        # the center frequency is the geometric mean of this critical band
        noise_freq = np.exp(np.mean(np.log(np.maximum(1, freqs[freq_indices]))))
        noise_maskers.append((noise_freq, noise_intensity))

    return tonal_maskers, noise_maskers


def identifyMaskers_MS(data_LR, sampleRate, sfBands, psi_array):
    """
    Identify the maskers from given time-domain samples data
    return: (tonal_maskers_MS, noise_maskers_MS), each a list of `np.array` of (frequency, intensity)
    """
    N = len(data_LR[0])

    spectrum_LR = []
    for iCh in range(2):
        # Hanning window first
        data_windowed = HanningWindow(data_LR[iCh])
        # FFT
        spectrum_LR.append(scipy.fft.rfft(data_windowed)[:-1])

    # rotation: to MS
    spectrum_M, spectrum_S = apply_rotation(spectrum_LR[0], spectrum_LR[1], psi_array, sfBands)

    tonal_maskers_M, noise_maskers_M = identifyMaskers_from_spectrum(
        spectrum_M, N, sampleRate, sfBands
    )
    tonal_maskers_S, noise_maskers_S = identifyMaskers_from_spectrum(
        spectrum_S, N, sampleRate, sfBands
    )
    return ([tonal_maskers_M, tonal_maskers_S], [noise_maskers_M, noise_maskers_S])


def getMaskedThreshold_MS(data_LR, sampleRate, sfBands, psi_array):
    """
    Return Masked Threshold evaluated at MDCT lines.

    Used by CalcSMR_MS, but can also be called from outside this module, which may
    be helpful when debugging the bit allocation code.
    """
    N = len(data_LR[0])
    # indentify the maskers
    tonal_maskers_MS, noise_maskers_MS = identifyMaskers_MS(data_LR, sampleRate, sfBands, psi_array)
    freqs = np.arange(N // 2) / N * sampleRate
    # combine all the masking curves and threshold in quiet to get the masked threshold
    quiet_thresh = Thresh(freqs)
    masked_thresh_MS = []
    for iCh in range(2):
        masked_thresh = quiet_thresh
        for tm_f, tm_intensity in tonal_maskers_MS[iCh]:
            tm_spl = SPL(tm_intensity)
            masker = Masker(
                tm_f, tm_spl, isTonal=True, isSide=(iCh == 1)
            )  # apply BMLD correction if is side
            masker_intensity = masker.vIntensityAtFreq(freqs)
            masking_curve = SPL(masker_intensity)
            masked_thresh = np.maximum(masked_thresh, masking_curve)  # alpha=inf
        for ns_f, ns_intensity in noise_maskers_MS[iCh]:
            ns_spl = SPL(ns_intensity)
            masker = Masker(
                ns_f, ns_spl, isTonal=False, isSide=(iCh == 1)
            )  # apply BMLD correction if is side
            masker_intensity = masker.vIntensityAtFreq(freqs)
            masking_curve = SPL(masker_intensity)
            masked_thresh = np.maximum(masked_thresh, masking_curve)  # alpha=inf
        masked_thresh_MS.append(masked_thresh)
    return masked_thresh_MS


def CalcSMRs_MS(data_LR, MDCTdata_MS, MDCTscale_MS, sampleRate, sfBands, psi_array):
    """
    Set SMR for each critical band in sfBands.

    Arguments:
                data:       [2] of an array of N time domain samples
                MDCTdata:   [2] of an array of N/2 MDCT frequency coefficients for the time domain samples
                            in data; note that the MDCT coefficients have been scaled up by a factor
                            of 2^MDCTscale
                MDCTscale:  [2] corresponding to an overall scale factor 2^MDCTscale for the set of MDCT
                            frequency coefficients
                sampleRate: is the sampling rate of the time domain samples
                sfBands:    points to information about which MDCT frequency lines
                            are in which scale factor band
                psi_array:  array of rotational anglesm, used for masking calculations.

    Returns:
        SMRs_MS:       [2] of arrays, each containing the maximum signal-to-mask ratio in each
                       scale factor band for the corresponding channel.
    Logic:
                Performs an FFT of data[N] and identifies tonal and noise maskers.
                Combines their relative masking curves and the hearing threshold
                to calculate the overall masked threshold at the MDCT frequency locations.
                                Then determines the maximum signal-to-mask ratio within
                each critical band and returns that result in the SMR[] array.
    """
    masked_thresh_MS = getMaskedThreshold_MS(data_LR, sampleRate, sfBands, psi_array)
    SMRs_MS = []
    for iCh in range(2):
        MDCTdata_origin = MDCTdata_MS[iCh] / (2 ** MDCTscale_MS[iCh])
        MDCTspl = SPL(intensity_from_MDCT(MDCTdata_origin))
        SMR_all = MDCTspl - masked_thresh_MS[iCh]
        # record the maximum SMR in each sfBand
        SMRs = []
        for lower_l, upper_l in zip(sfBands.lowerLine, sfBands.upperLine):
            SMR_max = np.max(SMR_all[lower_l : upper_l + 1])
            SMRs.append(SMR_max)
        SMRs_MS.append(np.array(SMRs))
    return SMRs_MS


# -----------------------------------------------------------------------------

# Testing code
if __name__ == "__main__":
    # 1.d)
    fls = [0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480,
           1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000]  # fmt: skip
    np.set_printoptions(precision=3)
    print(Bark(fls))

    # 1.f)
    nMDCTLines = 512
    sampleRate = 48000
    nLines = AssignMDCTLinesFromFreqLimits(nMDCTLines, sampleRate)
    scaleFactorBands = ScaleFactorBands(nLines)
    print("nBands =", scaleFactorBands.nBands)
    print("lowerLine =", scaleFactorBands.lowerLine)
    print("upperLine =", scaleFactorBands.upperLine)
    print("nLines =", scaleFactorBands.nLines)
