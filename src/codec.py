"""
codec.py -- The actual encode/decode functions for the perceptual audio codec

-----------------------------------------------------------------------
© 2019-2025 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np  # used for arrays

# used by Encode and Decode
from window import (
    HanningWindow,
    SineWindow,
)  # current window used for MDCT -- implement KB-derived?
from mdct import MDCT, IMDCT  # fast MDCT implementation (uses numpy FFT)
from quantize import *  # using vectorized versions (to use normal versions, uncomment lines 18,67 below defining vMantissa and vDequantize)
from rotation import rotational_ms, inverse_rotational_ms, apply_rotation
import scipy.fft  # used for FFTs in the side chain

# used only by Encode
from psychoac import CalcSMRs, CalcSMRs_MS  # calculates SMRs for each scale factor band
from bitalloc import BitAlloc  # allocates bits to scale factor bands given SMRs


def Decode(
    scaleFactor_MS, bitAlloc_MS, mantissa_MS, overallScaleFactor_MS, psi_array, codingParams
):
    """Reconstitutes a stereo-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object"""

    mdctLines_MS = []
    halfN = codingParams.nMDCTLines
    for scaleFactor, bitAlloc, mantissa, overallScaleFactor in zip(
        scaleFactor_MS, bitAlloc_MS, mantissa_MS, overallScaleFactor_MS
    ):
        rescaleLevel = 1.0 * (1 << overallScaleFactor)

        # reconstitute the first halfN MDCT lines of this channel from the stored data
        mdctLine = np.zeros(halfN, dtype=np.float64)
        iMant = 0
        for iBand in range(codingParams.sfBands.nBands):
            nLines = codingParams.sfBands.nLines[iBand]
            if bitAlloc[iBand]:
                mdctLine[iMant : (iMant + nLines)] = vDequantize(
                    scaleFactor[iBand],
                    mantissa[iMant : (iMant + nLines)],
                    codingParams.nScaleBits,
                    bitAlloc[iBand],
                )
            iMant += nLines
        mdctLine /= rescaleLevel  # put overall gain back to original level
        mdctLines_MS.append(mdctLine)

    # rotate back to L and R channels
    mdctLines_L, mdctLines_R = inverse_rotational_ms(
        mdctLines_MS[0], mdctLines_MS[1], psi_array, codingParams.sfBands
    )

    # IMDCT and window the data
    data_L = SineWindow(IMDCT(mdctLines_L, halfN, halfN))
    data_R = SineWindow(IMDCT(mdctLines_R, halfN, halfN))

    return (data_L, data_R)


def Encode(data, codingParams):
    """
    Encodes a stereo block of signed-fraction data based on the parameters in a PACFile object
    Returns:
        tuple: A tuple containing the following encoded data:
            - scaleFactor (list[np.ndarray]): Scale factors for each scale factor band in the mid and side channels.
            - bitAlloc (list[np.ndarray]): Bit allocations for each scale factor band in the mid and side channels.
            - mantissa (list[np.ndarray]): Quantized mantissas for each scale factor band in the mid and side channels.
            - overallScaleFactor (list[int]): Overall scale factors for the mid and side channels.
            - psi_array (np.ndarray): Array of rotational angles used for M/S processing.

    Notes:
        - The function assumes a binaural scenario with exactly two channels.
        - The MDCT, SMR calculation, and bit allocation are performed separately for the mid and side channels.
        - The function uses psychoacoustic models to optimize bit allocation for perceptual quality.
    """

    scaleFactor = []
    bitAlloc = []
    mantissa = []
    overallScaleFactor = []

    # We now assume the binaural scenario, i.e., two channels
    assert codingParams.nChannels == 2

    # prepare various constants
    halfN = codingParams.nMDCTLines
    N = 2 * halfN
    nScaleBits = codingParams.nScaleBits
    maxMantBits = 1 << codingParams.nMantSizeBits
    if maxMantBits > 16:
        maxMantBits = 16  # to make sure we don't ever overflow mantissa holders
    sfBands = codingParams.sfBands

    # compute target mantissa bit budget for this block of halfN MDCT mantissas
    bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
    bitBudget -= nScaleBits * (
        sfBands.nBands + 1
    )  # less scale factor bits (including overall scale factor)
    bitBudget -= codingParams.nMantSizeBits * sfBands.nBands  # less mantissa bit allocation bits
    bitBudget -= (
        codingParams.nPsiBits * sfBands.nBands // 2
    )  # less rotational angle bits (split between channels)

    # get MDCT lines
    mdctTimeSamples_L = SineWindow(data[0])
    mdctLines_L = MDCT(mdctTimeSamples_L, halfN, halfN)[:halfN]
    mdctTimeSamples_R = SineWindow(data[1])
    mdctLines_R = MDCT(mdctTimeSamples_R, halfN, halfN)[:halfN]

    # get FFT lines for angle calculation
    fftTimeSamples_L = HanningWindow(data[0])
    fftLines_L = scipy.fft.rfft(fftTimeSamples_L)[:halfN]
    fftTimeSamples_R = HanningWindow(data[1])
    fftLines_R = scipy.fft.rfft(fftTimeSamples_R)[:halfN]

    # calculate rotaional M/S
    psi_array, mdctLines_M, mdctLines_S = rotational_ms(
        mdctLines_L, mdctLines_R, fftLines_L, fftLines_R, sfBands
    )
    mdctLines_MS = [mdctLines_M, mdctLines_S]

    # compute overall scale factor for M/S block and boost mdctLines using it
    for iCh in range(2):
        maxLine = np.max(np.abs(mdctLines_MS[iCh]))
        overallScale = ScaleFactor(maxLine, nScaleBits)  # leading zeroes don't depend on nMantBits
        mdctLines_MS[iCh] *= 1 << overallScale
        overallScaleFactor.append(overallScale)

    # compute SMRs (take psi_array into account)
    SMRs_MS = CalcSMRs_MS(
        data, mdctLines_MS, overallScaleFactor, codingParams.sampleRate, sfBands, psi_array
    )

    # get coded results
    if codingParams.useML:
        raise NotImplementedError("ML bit allocation not implemented yet")
    else:
        # Split bitBudget unevenly?
        # Yes, in order to decrease the average block squared error,
        # The bitBudge should be split according to the SMR value of these two channels
        SMR_sum_MS = [np.sum(np.array(SMRs_MS[iCh]) * sfBands.nLines) for iCh in range(2)]
        SMR_sum_avg = 0.5 * (SMR_sum_MS[0] + SMR_sum_MS[1])
        bitBudget_MS = [bitBudget + 0.1661 * (SMR_sum_MS[iCh] - SMR_sum_avg) for iCh in range(2)]
        if bitBudget_MS[1] < 0:
            # incase side channel has negative bit budget
            bitBudget_MS[0] += bitBudget_MS[1]
            bitBudget_MS[1] = 0
        # bitBudget_MS = [bitBudget * 2, 0]
        # print(bitBudget_MS)
        for iCh in range(2):
            (s, b, m) = getCoded_from_SMR(
                N,
                bitBudget_MS[iCh],
                maxMantBits,
                sfBands,
                SMRs_MS[iCh],
                mdctLines_MS[iCh],
                nScaleBits,
            )
            scaleFactor.append(s)
            bitAlloc.append(b)
            mantissa.append(m)

    # return results bundled over channels
    return (scaleFactor, bitAlloc, mantissa, overallScaleFactor, psi_array)


def getCoded_from_SMR(N, bitBudget, maxMantBits, sfBands, SMRs, mdctLines, nScaleBits):
    # perform bit allocation using SMR results
    bitAlloc = BitAlloc(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs)

    # given the bit allocations, quantize the mdct lines in each band
    scaleFactor = np.empty(sfBands.nBands, dtype=np.int32)
    nMant = N // 2
    for iBand in range(sfBands.nBands):
        if not bitAlloc[iBand]:
            nMant -= sfBands.nLines[iBand]  # account for mantissas not being transmitted
    mantissa = np.empty(nMant, dtype=np.int32)
    iMant = 0
    for iBand in range(sfBands.nBands):
        lowLine = sfBands.lowerLine[iBand]
        highLine = (
            sfBands.upperLine[iBand] + 1
        )  # extra value is because slices don't include last value
        nLines = sfBands.nLines[iBand]
        scaleLine = np.max(np.abs(mdctLines[lowLine:highLine]))
        scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iBand])
        if bitAlloc[iBand]:
            mantissa[iMant : iMant + nLines] = vMantissa(
                mdctLines[lowLine:highLine],
                scaleFactor[iBand],
                nScaleBits,
                bitAlloc[iBand],
            )
            iMant += nLines
    # end of loop over scale factor bands

    # return results
    return (scaleFactor, bitAlloc, mantissa)


def EncodeSingleChannel(data, codingParams):
    """Encodes a single-channel block of signed-fraction data based on the parameters in a PACFile object"""

    # prepare various constants
    halfN = codingParams.nMDCTLines
    N = 2 * halfN
    nScaleBits = codingParams.nScaleBits
    maxMantBits = (
        1 << codingParams.nMantSizeBits
    )  # 1 isn't an allowed bit allocation so n size bits counts up to 2^n
    if maxMantBits > 16:
        maxMantBits = 16  # to make sure we don't ever overflow mantissa holders
    sfBands = codingParams.sfBands
    # vectorizing the Mantissa function call
    #    vMantissa = np.vectorize(Mantissa)

    # compute target mantissa bit budget for this block of halfN MDCT mantissas
    bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
    bitBudget -= nScaleBits * (
        sfBands.nBands + 1
    )  # less scale factor bits (including overall scale factor)
    bitBudget -= codingParams.nMantSizeBits * sfBands.nBands  # less mantissa bit allocation bits

    # window data for side chain FFT and also window and compute MDCT
    timeSamples = data
    mdctTimeSamples = SineWindow(data)
    mdctLines = MDCT(mdctTimeSamples, halfN, halfN)[:halfN]

    # compute overall scale factor for this block and boost mdctLines using it
    maxLine = np.max(np.abs(mdctLines))
    overallScale = ScaleFactor(maxLine, nScaleBits)  # leading zeroes don't depend on nMantBits
    mdctLines *= 1 << overallScale

    # compute the mantissa bit allocations
    # compute SMRs in side chain FFT
    SMRs = CalcSMRs(timeSamples, mdctLines, overallScale, codingParams.sampleRate, sfBands)
    # perform bit allocation using SMR results
    bitAlloc = BitAlloc(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs)

    # given the bit allocations, quantize the mdct lines in each band
    scaleFactor = np.empty(sfBands.nBands, dtype=np.int32)
    nMant = halfN
    for iBand in range(sfBands.nBands):
        if not bitAlloc[iBand]:
            nMant -= sfBands.nLines[iBand]  # account for mantissas not being transmitted
    mantissa = np.empty(nMant, dtype=np.int32)
    iMant = 0
    for iBand in range(sfBands.nBands):
        lowLine = sfBands.lowerLine[iBand]
        highLine = (
            sfBands.upperLine[iBand] + 1
        )  # extra value is because slices don't include last value
        nLines = sfBands.nLines[iBand]
        scaleLine = np.max(np.abs(mdctLines[lowLine:highLine]))
        scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iBand])
        if bitAlloc[iBand]:
            mantissa[iMant : iMant + nLines] = vMantissa(
                mdctLines[lowLine:highLine],
                scaleFactor[iBand],
                nScaleBits,
                bitAlloc[iBand],
            )
            iMant += nLines
    # end of loop over scale factor bands

    # return results
    return (scaleFactor, bitAlloc, mantissa, overallScale)
