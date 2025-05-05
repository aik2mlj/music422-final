"""
Calculate the rotation matrix for M/S split.
"""

import numpy as np
from quantize import vQuantizeUniform, vDequantizeUniform


def calc_rotation_angles(fftLines_L, fftLines_R, sfBands):
    """
    Calculate the rotation angles for each frequency subband for the given MDCT lines.
    """
    # for each pair of subbands from fftLines_L and fftLines_R, calculate the covariance matrix
    # and store it in a list
    cov_matrices = []

    rotation_angles = np.zeros(sfBands.nBands)
    for iBand in range(sfBands.nBands):
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[iBand] + 1
        nLines = sfBands.nLines[iBand]
        # get the absolute value of fftLines for the current subband
        subband_L = np.abs(fftLines_L[lowLine:highLine])
        subband_R = np.abs(fftLines_R[lowLine:highLine])
        # calculate the covariance matrix
        cov_matrix = np.cov(np.vstack((subband_L, subband_R)))
        cov_matrices.append(cov_matrix)
        # calculate the rotation angle, use arctan2 to avoid division by zero
        rotation_angle = 0.5 * np.arctan2(
            cov_matrix[0, 1] + cov_matrix[1, 0], cov_matrix[0, 0] - cov_matrix[1, 1]
        )
        # rotation_angle = 0.5 * np.arctan((cov_matrix[0, 1] + cov_matrix[1, 0]) / (cov_matrix[0, 0] - cov_matrix[1, 1]))
        rotation_angles[iBand] = rotation_angle

    return rotation_angles


def quantize_rotation(rotation_angles, nPsiBits=4):
    """
    Uniformly quantize the rotation angles using nPsiBits-bit precision.
    """
    # normalize the rotation angles to the range [-pi/2, pi/2]
    rotation_angles = np.array(rotation_angles) / (np.pi / 2)

    # quantize the rotation angles
    return vQuantizeUniform(rotation_angles, nPsiBits)


def apply_rotation(mdctLines_L, mdctLines_R, quantized_rotation_angles, sfBands, nPsiBits=4):
    """
    Apply the rotation to the MDCT lines.
    Return mdctLines_M and mdctLines_S.
    """
    psi_array = vDequantizeUniform(quantized_rotation_angles, nPsiBits) * (np.pi / 2)
    mdctLines_M = np.zeros_like(mdctLines_L)
    mdctLines_S = np.zeros_like(mdctLines_L)

    # apply rotation to each subband
    for iBand in range(len(psi_array)):
        rotation_angle = psi_array[iBand]
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[iBand] + 1
        nLines = sfBands.nLines[iBand]
        mdctLines_L_subband = mdctLines_L[lowLine:highLine]
        mdctLines_R_subband = mdctLines_R[lowLine:highLine]

        # calculate rotation matrix elements
        cos_psi = np.cos(rotation_angle)
        sin_psi = np.sin(rotation_angle)

        # apply rotation
        mdctLines_M_subband = cos_psi * mdctLines_L_subband + sin_psi * mdctLines_R_subband
        mdctLines_S_subband = -sin_psi * mdctLines_L_subband + cos_psi * mdctLines_R_subband

        # update the MDCT lines
        mdctLines_M[lowLine:highLine] = mdctLines_M_subband
        mdctLines_S[lowLine:highLine] = mdctLines_S_subband

    return mdctLines_M, mdctLines_S


def apply_inverse_rotation(
    mdctLines_M, mdctLines_S, quantized_rotation_angles, sfBands, nPsiBits=4
):
    """
    Apply the inverse rotation to the MDCT lines.
    Return mdctLines_L and mdctLines_R.
    """
    psi_array = vDequantizeUniform(np.array(quantized_rotation_angles), nPsiBits) * (np.pi / 2)
    mdctLines_L = np.zeros_like(mdctLines_M)
    mdctLines_R = np.zeros_like(mdctLines_M)

    # apply inverse rotation to each subband
    for iBand in range(len(psi_array)):
        rotation_angle = psi_array[iBand]
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[iBand] + 1
        nLines = sfBands.nLines[iBand]
        mdctLines_M_subband = mdctLines_M[lowLine:highLine]
        mdctLines_S_subband = mdctLines_S[lowLine:highLine]

        # calculate inverse rotation matrix elements
        cos_psi = np.cos(rotation_angle)
        sin_psi = np.sin(rotation_angle)

        # apply inverse rotation
        mdctLines_L_subband = cos_psi * mdctLines_M_subband - sin_psi * mdctLines_S_subband
        mdctLines_R_subband = sin_psi * mdctLines_M_subband + cos_psi * mdctLines_S_subband

        # update the MDCT lines
        mdctLines_L[lowLine:highLine] = mdctLines_L_subband
        mdctLines_R[lowLine:highLine] = mdctLines_R_subband

    return mdctLines_L, mdctLines_R


def rotational_ms(mdctLines_L, mdctLines_R, fftLines_L, fftLines_R, sfBands, isVanilla=False):
    """
    Split the MDCT lines into M and S components using rotational M/S.
    Now use fftLines for rotation angles calculation.
    """
    # sanity check with vanilla M/S
    if isVanilla:
        psi_array_vanilla = quantize_rotation(np.full(sfBands.nBands, np.pi / 4), nPsiBits=4)
        mdctLines_M_vanilla, mdctLines_S_vanilla = apply_rotation(
            mdctLines_L, mdctLines_R, psi_array_vanilla, sfBands
        )
        return psi_array_vanilla, mdctLines_M_vanilla, mdctLines_S_vanilla

    psi_array = quantize_rotation(calc_rotation_angles(fftLines_L, fftLines_R, sfBands))
    mdctLines_M, mdctLines_S = apply_rotation(mdctLines_L, mdctLines_R, psi_array, sfBands)
    return psi_array, mdctLines_M, mdctLines_S


def inverse_rotational_ms(mdctLines_M, mdctLines_S, psi_array, sfBands):
    """
    Restore the L and R components from the M and S components using inverse rotational M/S.
    """
    mdctLines_L, mdctLines_R = apply_inverse_rotation(mdctLines_M, mdctLines_S, psi_array, sfBands)
    return mdctLines_L, mdctLines_R


if __name__ == "__main__":
    # test the rotational_ms and inverse function to see if we get back the original signal
    from psychoac import ScaleFactorBands, AssignMDCTLinesFromFreqLimits

    def sum_intensity(mdctLines):
        return np.sum(mdctLines**2)

    N = 1024
    nLines = AssignMDCTLinesFromFreqLimits(N // 2, 48000)
    # randomize the MDCT lines between 0 and 1
    mdctLines_L = np.concatenate(
        (
            np.random.rand(100) * 0.1,
            np.random.rand(100) * 0.1,
            np.random.rand(100) * 0.1,
            np.random.rand(212),
        )
    )
    mdctLines_R = np.concatenate(
        (
            np.random.rand(100),
            np.random.rand(100) * 0.9,
            np.random.rand(100),
            np.random.rand(212) * 0.1,
        )
    )
    print("mdctLines_L:", mdctLines_L[0:10])
    print("mdctLines_R:", mdctLines_R[0:10])
    print("sum_intensity(mdctLines_L):", sum_intensity(mdctLines_L))
    print("sum_intensity(mdctLines_R):", sum_intensity(mdctLines_R))

    # test rotation
    sfBands = ScaleFactorBands(nLines)
    psi_array, mdctLines_M, mdctLines_S = rotational_ms(mdctLines_L, mdctLines_R, sfBands)
    print("psi_array:", psi_array)
    print("M channel:", mdctLines_M[0:10])
    print("S channel:", mdctLines_S[0:10])
    print("sum_intensity(mdctLines_M):", sum_intensity(mdctLines_M))
    print("sum_intensity(mdctLines_S):", sum_intensity(mdctLines_S))

    # test vanilla rotation
    psi_array_vanilla, mdctLines_M_vanilla, mdctLines_S_vanilla = rotational_ms(
        mdctLines_L, mdctLines_R, sfBands, isVanilla=True
    )
    print("psi_array_vanilla:", psi_array_vanilla)
    print("M channel vanilla:", mdctLines_M_vanilla[0:10])
    print("S channel vanilla:", mdctLines_S_vanilla[0:10])
    print("sum_intensity(mdctLines_M_vanilla):", sum_intensity(mdctLines_M_vanilla))
    print("sum_intensity(mdctLines_S_vanilla):", sum_intensity(mdctLines_S_vanilla))

    mdctLines_L_restored, mdctLines_R_restored = inverse_rotational_ms(
        mdctLines_M, mdctLines_S, psi_array, sfBands
    )
    print("L channel restored:", mdctLines_L_restored[0:10])
    print("R channel restored:", mdctLines_R_restored[0:10])
    print("sum_intensity(mdctLines_L_restored):", sum_intensity(mdctLines_L_restored))
    print("sum_intensity(mdctLines_R_restored):", sum_intensity(mdctLines_R_restored))

    print("M/S energy ratio:", sum_intensity(mdctLines_M) / sum_intensity(mdctLines_S))
    print("L/R energy ratio:", sum_intensity(mdctLines_L) / sum_intensity(mdctLines_R))
    print(
        "M/S energy ratio vanilla:",
        sum_intensity(mdctLines_M_vanilla) / sum_intensity(mdctLines_S_vanilla),
    )
    # assert if sum of intensities M is greater than sum of intensities S
    assert sum_intensity(mdctLines_M) > sum_intensity(mdctLines_S)
