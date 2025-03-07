"""
Music 422 - Marina Bosi

quantize.py -- routines to quantize and dequantize floating point values
between -1.0 and 1.0 ("signed fractions")

-----------------------------------------------------------------------
© 2009-2025 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np


### Problem 1.a.i ###
def QuantizeUniform(aNum, nBits):
    """
    Uniformly quantize signed fraction aNum with nBits
    """
    # Notes:
    # The overload level of the quantizer should be 1.0

    aQuantizedNum = 0

    ### YOUR CODE STARTS HERE ###
    if aNum < 0:
        aQuantizedNum = 1 << (nBits - 1)  # sign bit

    if abs(aNum) >= 1:
        aQuantizedNum |= (1 << (nBits - 1)) - 1
    else:
        aQuantizedNum |= int((((1 << nBits) - 1) * abs(aNum) + 1) / 2)
    ### YOUR CODE ENDS HERE ###

    return aQuantizedNum


### Problem 1.a.i ###
def DequantizeUniform(aQuantizedNum, nBits):
    """
    Uniformly dequantizes nBits-long number aQuantizedNum into a signed fraction
    """

    ### YOUR CODE STARTS HERE ###
    sign = -1 if (aQuantizedNum >> (nBits - 1)) == 1 else 1
    code = aQuantizedNum & ((1 << (nBits - 1)) - 1)
    aNum = sign * 2 * code / ((1 << nBits) - 1)
    ### YOUR CODE ENDS HERE ###

    return aNum


### Problem 1.a.ii ###
def vQuantizeUniform(aNumVec, nBits):
    """
    Uniformly quantize vector aNumberVec of signed fractions with nBits
    """

    # Notes:
    # Make sure to vectorize properly your function as specified in the homework instructions

    ### YOUR CODE STARTS HERE ###
    # Compute the sign bit
    sign = (aNumVec < 0).astype(int) << (nBits - 1)

    # Compute the magnitude quantization
    abs_aNumArray = np.abs(aNumVec)
    code = np.where(
        abs_aNumArray >= 1,
        (1 << (nBits - 1)) - 1,  # overload
        np.floor(((1 << nBits) - 1) * abs_aNumArray / 2 + 0.5).astype(int),
    )

    aQuantizedNumVec = sign | code
    ### YOUR CODE ENDS HERE ###

    return aQuantizedNumVec


### Problem 1.a.ii ###
def vDequantizeUniform(aQuantizedNumVec, nBits):
    """
    Uniformly dequantizes vector of nBits-long numbers aQuantizedNumVec into vector of signed fractions
    """

    ### YOUR CODE STARTS HERE ###
    if nBits == 0:
        exit(0)
    sign = np.where((aQuantizedNumVec >> (nBits - 1)), -1, 1)
    code = aQuantizedNumVec & ((1 << (nBits - 1)) - 1)
    aNumVec = sign * 2 * code / ((1 << nBits) - 1)

    ### YOUR CODE ENDS HERE ###

    return aNumVec


### Problem 1.b ###
def ScaleFactor(aNum, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point scale factor for a signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    # Notes:
    # The scale factor should be the number of leading zeros

    ### YOUR CODE STARTS HERE ###
    # get full quantization
    nBits = (1 << nScaleBits) - 1 + nMantBits
    aQuantizedNum = QuantizeUniform(aNum, nBits)

    # compute number of leading zeros
    testBit = 1 << (nBits - 2)
    nLeadingZeros = 0
    while not (aQuantizedNum & testBit) and testBit > 0:
        nLeadingZeros += 1
        testBit >>= 1

    if nLeadingZeros < (1 << nScaleBits) - 1:
        scale = nLeadingZeros
    else:
        scale = (1 << nScaleBits) - 1
    ### YOUR CODE ENDS HERE ###

    return scale


### Problem 1.b ###
def MantissaFP(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point mantissa for a signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """

    ### YOUR CODE STARTS HERE ###
    # get full quantization
    maxScale = (1 << nScaleBits) - 1
    nBits = maxScale + nMantBits
    aQuantizedNum = QuantizeUniform(aNum, nBits)

    mantissa = (aNum < 0) << (nMantBits - 1)  # sign bit
    codeBits_getter = (1 << (nMantBits - 1)) - 1
    if scale == maxScale:
        mantissa |= aQuantizedNum & codeBits_getter
    else:
        mantissa |= (aQuantizedNum >> (maxScale - scale - 1)) & codeBits_getter
    ### YOUR CODE ENDS HERE ###

    return mantissa


### Problem 1.b ###
def DequantizeFP(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a signed fraction for floating-point scale and mantissa given specified scale and mantissa bits
    """

    ### YOUR CODE STARTS HERE ###
    maxScale = (1 << nScaleBits) - 1
    nBits = maxScale + nMantBits
    aQuantizedNum = mantissa >> (nMantBits - 1) << (nBits - 1)  # sign bit
    codeBits_getter = (1 << (nMantBits - 1)) - 1
    if scale == maxScale:
        aQuantizedNum |= mantissa & codeBits_getter
    else:
        aQuantizedNum |= (
            1 << (nBits - 2 - scale)  # leading 1
            | (mantissa & codeBits_getter) << (nBits - 1 - scale - nMantBits)
        )
        if maxScale - scale - 2 >= 0:
            aQuantizedNum |= 1 << (maxScale - scale - 2)  # guessing bit

    aNum = DequantizeUniform(aQuantizedNum, nBits)
    ### YOUR CODE ENDS HERE ###

    return aNum


### Problem 1.c.i ###
def Mantissa(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the block floating-point mantissa for a signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """

    ### YOUR CODE STARTS HERE ###
    # get full quantization
    maxScale = (1 << nScaleBits) - 1
    nBits = maxScale + nMantBits
    aQuantizedNum = QuantizeUniform(aNum, nBits)

    mantissa = (aNum < 0) << (nMantBits - 1)  # sign bit
    codeBits_getter = (1 << (nMantBits - 1)) - 1
    mantissa |= (aQuantizedNum >> (maxScale - scale)) & codeBits_getter
    ### YOUR CODE ENDS HERE ###

    return mantissa


### Problem 1.c.i ###
def Dequantize(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a signed fraction for block floating-point scale and mantissa given specified scale and mantissa bits
    """

    ### YOUR CODE STARTS HERE ###
    maxScale = (1 << nScaleBits) - 1
    nBits = maxScale + nMantBits
    aQuantizedNum = mantissa >> (nMantBits - 1) << (nBits - 1)  # sign bit
    codeBits_getter = (1 << (nMantBits - 1)) - 1
    if scale == maxScale:
        aQuantizedNum |= mantissa & codeBits_getter
    else:
        aQuantizedNum |= (mantissa & codeBits_getter) << (nBits - scale - nMantBits)
        if maxScale - scale - 1 >= 0 and (mantissa & codeBits_getter):
            aQuantizedNum |= 1 << (maxScale - scale - 1)  # guessing bit

    aNum = DequantizeUniform(aQuantizedNum, nBits)
    ### YOUR CODE ENDS HERE ###

    return aNum


### Problem 1.c.ii ###
def vMantissa(aNumVec, scale, nScaleBits=3, nMantBits=5):
    """
    Return a vector of block floating-point mantissas for a vector of signed fractions aNum given nScaleBits scale bits and nMantBits mantissa bits
    """

    ### YOUR CODE STARTS HERE ###
    # get full quantization
    maxScale = (1 << nScaleBits) - 1
    nBits = maxScale + nMantBits
    aQuantizedNumVec = vQuantizeUniform(aNumVec, nBits)

    mantissaVec = (aNumVec < 0).astype(int) << (nMantBits - 1)
    codeBits_getter = (1 << (nMantBits - 1)) - 1
    mantissaVec |= (aQuantizedNumVec >> (maxScale - scale)) & codeBits_getter
    ### YOUR CODE ENDS HERE ###

    return mantissaVec


### Problem 1.c.ii ###
def vDequantize(scale, mantissaVec, nScaleBits=3, nMantBits=5):
    """
    Returns a vector of signed fractions for block floating-point scale and vector of block floating-point mantissas given specified scale and mantissa bits
    """

    ### YOUR CODE STARTS HERE ###
    maxScale = (1 << nScaleBits) - 1
    nBits = maxScale + nMantBits
    aQuantizedNumVec = mantissaVec >> (nMantBits - 1) << (nBits - 1)  # sign bit
    codeBits_getter = (1 << (nMantBits - 1)) - 1
    if scale == maxScale:
        aQuantizedNumVec |= mantissaVec & codeBits_getter
    else:
        aQuantizedNumVec |= (mantissaVec & codeBits_getter) << (nBits - scale - nMantBits)
        if maxScale - scale - 1 >= 0:
            aQuantizedNumVec = np.where(
                mantissaVec & codeBits_getter,
                aQuantizedNumVec | 1 << (maxScale - scale - 1),
                aQuantizedNumVec,
            )
    aNumVec = vDequantizeUniform(aQuantizedNumVec, nBits)
    ### YOUR CODE ENDS HERE ###

    return aNumVec


# -----------------------------------------------------------------------------

# Testing code
if __name__ == "__main__":
    ### YOUR TESTING CODE STARTS HERE ###
    xs = np.array(
        [
            -0.99,
            -0.38,
            -0.10,
            -0.01,
            -0.001,
            0.0,
            0.05,
            0.28,
            0.65,
            0.97,
            1.0,
        ]
    )
    print("12 bit binary")
    for y in vQuantizeUniform(xs, 12):
        print(f"{y:012b}")
    print("====================")
    print("8 bit midread")
    for y in vDequantizeUniform(vQuantizeUniform(xs, 8), 8):
        print(f"{y:.5f}")
    print("====================")
    print("12 bit midread")
    for y in vDequantizeUniform(vQuantizeUniform(xs, 12), 12):
        print(f"{y:.5f}")
    print("====================")
    print("3s5m FP")
    for x in xs:
        scale = ScaleFactor(x)
        mantissa = MantissaFP(x, scale)
        y = DequantizeFP(scale, mantissa)
        print(f"{y:.5f}")
    print("====================")
    print("3s5m BFP N=1")
    for x in xs:
        scale = ScaleFactor(x)
        mantissa = Mantissa(x, scale)
        y = Dequantize(scale, mantissa)
        print(f"{y:.5f}")
    ### YOUR TESTING CODE ENDS HERE ###
