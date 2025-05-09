"""
pacfile.py -- Defines a PACFile class to handle reading and writing audio
data to an audio file holding data compressed using an MDCT-based perceptual audio
coding algorithm.  The MDCT lines of each audio channel are grouped into bands,
each sharing a single scaleFactor and bit allocation that are used to block-
floating point quantize those lines.  This class is a subclass of AudioFile.

-----------------------------------------------------------------------
© 2019-2025 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------

See the documentation of the AudioFile class for general use of the AudioFile
class.

Notes on reading and decoding PAC files:

    The OpenFileForReading() function returns a CodedParams object containing:

        nChannels = the number of audio channels
        sampleRate = the sample rate of the audio samples
        numSamples = the total number of samples in the file for each channel
        nMDCTLines = half the MDCT block size (block switching not supported)
        nSamplesPerBlock = MDCTLines (but a name that PCM files look for)
        nScaleBits = the number of bits storing scale factors
        nMantSizeBits = the number of bits storing mantissa bit allocations
        sfBands = a ScaleFactorBands object
        overlapAndAdd = decoded data from the prior block (initially all zeros)

    The returned ScaleFactorBands object, sfBands, contains an allocation of
    the MDCT lines into groups that share a single scale factor and mantissa bit
    allocation.  sfBands has the following attributes available:

        nBands = the total number of scale factor bands
        nLines[iBand] = the number of MDCT lines in scale factor band iBand
        lowerLine[iBand] = the first MDCT line in scale factor band iBand
        upperLine[iBand] = the last MDCT line in scale factor band iBand


Notes on encoding and writing PAC files:

    When writing to a PACFile the CodingParams object passed to OpenForWriting()
    should have the following attributes set:

        nChannels = the number of audio channels
        sampleRate = the sample rate of the audio samples
        numSamples = the total number of samples in the file for each channel
        nMDCTLines = half the MDCT block size (format does not support block switching)
        nSamplesPerBlock = MDCTLines (but a name that PCM files look for)
        nScaleBits = the number of bits storing scale factors
        nMantSizeBits = the number of bits storing mantissa bit allocations
        targetBitsPerSample = the target encoding bit rate in units of bits per sample

    The first three attributes (nChannels, sampleRate, and numSamples) are
    typically added by the original data source (e.g. a PCMFile object) but
    numSamples may need to be extended to account for the MDCT coding delay of
    nMDCTLines and any zero-padding done in the final data block

    OpenForWriting() will add the following attributes to be used during the encoding
    process carried out in WriteDataBlock():

        sfBands = a ScaleFactorBands object
        priorBlock = the prior block of audio data (initially all zeros)

    The passed ScaleFactorBands object, sfBands, contains an allocation of
    the MDCT lines into groups that share a single scale factor and mantissa bit
    allocation.  sfBands has the following attributes available:

        nBands = the total number of scale factor bands
        nLines[iBand] = the number of MDCT lines in scale factor band iBand
        lowerLine[iBand] = the first MDCT line in scale factor band iBand
        upperLine[iBand] = the last MDCT line in scale factor band iBand

Description of the PAC File Format:

    Header:

        tag                 4 byte file tag equal to "PAC "
        sampleRate          little-endian unsigned long ("<L" format in struct)
        nChannels           little-endian unsigned short("<H" format in struct)
        numSamples          little-endian unsigned long ("<L" format in struct)
        nMDCTLines          little-endian unsigned long ("<L" format in struct)
        nScaleBits          little-endian unsigned short("<H" format in struct)
        nMantSizeBits       little-endian unsigned short("<H" format in struct)
        nSFBands            little-endian unsigned long ("<L" format in struct)
        for iBand in range(nSFBands):
            nLines[iBand]   little-endian unsigned short("<H" format in struct)

    Each Data Block:  (reads data blocks until end of file hit)

        for iCh in range(nChannels):
            nBytes          little-endian unsigned long ("<L" format in struct)
            as bits packed into an array of nBytes bytes:
                overallScale[iCh]                       nScaleBits bits
                for iBand in range(nSFBands):
                    scaleFactor[iCh][iBand]             nScaleBits bits
                    bitAlloc[iCh][iBand]                nMantSizeBits bits
                    if bitAlloc[iCh][iBand]:
                        for m in nLines[iBand]:
                            mantissa[iCh][iBand][m]     bitAlloc[iCh][iBand]+1 bits
                <extra custom data bits as long as space is included in nBytes>

"""

from audiofile import *  # base class
from bitpack import *  # class for packing data into an array of bytes where each item's number of bits is specified
import codec  # module where the actual PAC coding functions reside(this module only specifies the PAC file format)
from psychoac import (
    ScaleFactorBands,
    AssignMDCTLinesFromFreqLimits,
)  # defines the grouping of MDCT lines into scale factor bands

import numpy as np  # to allow conversion of data blocks to numpy's array object
import time

MAX16BITS = 32767


class PACFile(AudioFile):
    """
    Handlers for a perceptually coded audio file I am encoding/decoding
    """

    # a file tag to recognize PAC coded files
    tag = b"PAC "

    def ReadFileHeader(self):
        """
        Reads the PAC file header from a just-opened PAC file and uses it to set
        object attributes.  File pointer ends at start of data portion.
        """
        # check file header tag to make sure it is the right kind of file
        tag = self.fp.read(4)
        if tag != self.tag:
            raise RuntimeError("Tried to read a non-PAC file into a PACFile object")
        # use struct.unpack() to load up all the header data
        (sampleRate, nChannels, numSamples, nMDCTLines, nScaleBits, nMantSizeBits) = unpack(
            "<LHLLHH", self.fp.read(calcsize("<LHLLHH"))
        )
        nBands = unpack("<L", self.fp.read(calcsize("<L")))[0]
        nLines = unpack("<" + str(nBands) + "H", self.fp.read(calcsize("<" + str(nBands) + "H")))
        sfBands = ScaleFactorBands(nLines)
        # load up a CodingParams object with the header data
        myParams = CodingParams()
        myParams.sampleRate = sampleRate
        myParams.nChannels = nChannels
        myParams.numSamples = numSamples
        myParams.nMDCTLines = myParams.nSamplesPerBlock = nMDCTLines
        myParams.nScaleBits = nScaleBits
        myParams.nMantSizeBits = nMantSizeBits
        # add in scale factor band information
        myParams.sfBands = sfBands
        # start w/o all zeroes as data from prior block to overlap-and-add for output
        overlapAndAdd = []
        for iCh in range(nChannels):
            overlapAndAdd.append(np.zeros(nMDCTLines, dtype=np.float64))
        myParams.overlapAndAdd = overlapAndAdd

        # new ones for M/S and neural network
        (useML, nPsiBits) = unpack("<HH", self.fp.read(calcsize("<HH")))
        myParams.useML = useML
        myParams.nPsiBits = nPsiBits

        return myParams

    def ReadDataBlock(self, codingParams):
        """
        Reads a block of coded data from a PACFile object that has already
        executed OpenForReading() and returns those samples as reconstituted
        signed-fraction data
        """
        # TODO: change reading strategy for M/S coding
        # loop over channels (whose coded data are stored separately) and read in each data block
        data = []
        scaleFactor_MS = []
        bitAlloc_MS = []
        mantissa_MS = []
        overallScaleFactor_MS = []
        for iCh in range(codingParams.nChannels):
            # read in string containing the number of bytes of data for this channel (but check if at end of file!)
            s = self.fp.read(calcsize("<L"))  # will be empty if at end of file
            if not s:
                # hit last block, see if final overlap and add needs returning, else return nothing
                if codingParams.overlapAndAdd:
                    overlapAndAdd = codingParams.overlapAndAdd
                    codingParams.overlapAndAdd = (
                        0  # setting it to zero so next pass will just return
                    )
                    return overlapAndAdd
                else:
                    return
            # not at end of file, get nBytes from the string we just read
            nBytes = unpack("<L", s)[0]  # read it as a little-endian unsigned long
            # read the nBytes of data into a PackedBits object to unpack
            pb = PackedBits()
            pb.SetPackedData(
                self.fp.read(nBytes)
            )  # PackedBits function SetPackedData() converts strings to internally-held array of bytes
            if pb.nBytes < nBytes:
                raise "Only read a partial block of coded PACFile data"

            # extract the data from the PackedBits object
            overallScaleFactor = pb.ReadBits(codingParams.nScaleBits)  # overall scale factor
            scaleFactor = []
            bitAlloc = []
            mantissa = np.zeros(codingParams.nMDCTLines, np.int32)  # start w/ all mantissas zero
            for iBand in range(
                codingParams.sfBands.nBands
            ):  # loop over each scale factor band to pack its data
                ba = pb.ReadBits(codingParams.nMantSizeBits)
                if ba:
                    ba += 1  # no bit allocation of 1 so ba of 2 and up stored as one less
                bitAlloc.append(ba)  # bit allocation for this band
                scaleFactor.append(
                    pb.ReadBits(codingParams.nScaleBits)
                )  # scale factor for this band
                if bitAlloc[iBand]:
                    # if bits allocated, extract those mantissas and put in correct location in matnissa array
                    m = np.empty(codingParams.sfBands.nLines[iBand], np.int32)
                    for j in range(codingParams.sfBands.nLines[iBand]):
                        m[j] = pb.ReadBits(
                            bitAlloc[iBand]
                        )  # mantissas for this band (if bit allocation non-zero) and bit alloc <>1 so encoded as 1 lower than actual allocation
                    mantissa[
                        codingParams.sfBands.lowerLine[iBand] : (
                            codingParams.sfBands.upperLine[iBand] + 1
                        )
                    ] = m
            scaleFactor_MS.append(scaleFactor)
            bitAlloc_MS.append(bitAlloc)
            mantissa_MS.append(mantissa)
            overallScaleFactor_MS.append(overallScaleFactor)
            # done unpacking data (end loop over scale factor bands)

        # read psi_array for M/S rotation
        nBits = codingParams.nPsiBits * codingParams.sfBands.nBands
        if nBits % BYTESIZE == 0:
            nBytes = nBits // BYTESIZE
        else:
            nBytes = nBits // BYTESIZE + 1
        pb = PackedBits()
        pb.SetPackedData(self.fp.read(nBytes))
        psi_array = []
        for iBand in range(codingParams.sfBands.nBands):
            psi_array.append(pb.ReadBits(codingParams.nPsiBits))

        # decode the time-domain data for this block
        decodedData_LR = self.Decode(
            scaleFactor_MS, bitAlloc_MS, mantissa_MS, overallScaleFactor_MS, psi_array, codingParams
        )
        # overlap-and-add the decoded data for each channel
        for iCh in range(codingParams.nChannels):
            data.append(np.array([], dtype=np.float64))  # add location for this channel's data
            decodedData = decodedData_LR[iCh]
            # (DECODE HERE) decode the unpacked data for this channel, overlap-and-add first half, and append it to the data array (saving other half for next overlap-and-add)
            data[iCh] = np.concatenate(
                (
                    data[iCh],
                    np.add(
                        codingParams.overlapAndAdd[iCh],
                        decodedData[: codingParams.nMDCTLines],
                    ),
                )
            )  # data[iCh] is overlap-and-added data
            codingParams.overlapAndAdd[iCh] = decodedData[
                codingParams.nMDCTLines :
            ]  # save other half for next pass

        # end loop over channels, return signed-fraction samples for this block
        return data

    def WriteFileHeader(self, codingParams):
        """
        Writes the PAC file header for a just-opened PAC file and uses codingParams
        attributes for the header data.  File pointer ends at start of data portion.
        """
        # write a header tag
        self.fp.write(self.tag)
        # make sure that the number of samples in the file is a multiple of the
        # number of MDCT half-blocksize, otherwise zero pad as needed
        if not codingParams.numSamples % codingParams.nMDCTLines:
            codingParams.numSamples += (
                codingParams.nMDCTLines - codingParams.numSamples % codingParams.nMDCTLines
            )  # zero padding for partial final PCM block
        # also add in the delay block for the second pass w/ the last half-block
        codingParams.numSamples += (
            codingParams.nMDCTLines
        )  # due to the delay in processing the first samples on both sides of the MDCT block
        # write the coded file attributes
        self.fp.write(
            pack(
                "<LHLLHH",
                codingParams.sampleRate,
                codingParams.nChannels,
                codingParams.numSamples,
                codingParams.nMDCTLines,
                codingParams.nScaleBits,
                codingParams.nMantSizeBits,
            )
        )
        # create a ScaleFactorBand object to be used by the encoding process and write its info to header
        sfBands = ScaleFactorBands(
            AssignMDCTLinesFromFreqLimits(codingParams.nMDCTLines, codingParams.sampleRate)
        )
        codingParams.sfBands = sfBands
        self.fp.write(pack("<L", sfBands.nBands))
        self.fp.write(pack("<" + str(sfBands.nBands) + "H", *(sfBands.nLines.tolist())))

        # write M/S coding params
        self.fp.write(pack("<HH", codingParams.useML, codingParams.nPsiBits))

        # start w/o all zeroes as prior block of unencoded data for other half of MDCT block
        priorBlock = []
        for iCh in range(codingParams.nChannels):
            priorBlock.append(np.zeros(codingParams.nMDCTLines, dtype=np.float64))
        codingParams.priorBlock = priorBlock
        return

    def WriteDataBlock(self, data, codingParams):
        """
        Writes a block of signed-fraction data to a PACFile object that has
        already executed OpenForWriting()"""

        # TODO: change writing strategy for M/S coding
        # combine this block of multi-channel data w/ the prior block's to prepare for MDCTs twice as long
        fullBlockData = []
        for iCh in range(codingParams.nChannels):
            fullBlockData.append(np.concatenate((codingParams.priorBlock[iCh], data[iCh])))
        codingParams.priorBlock = data  # current pass's data is next pass's prior block data

        # (ENCODE HERE) Encode the full block of multi=channel data
        (scaleFactor, bitAlloc, mantissa, overallScaleFactor, psi_array) = self.Encode(
            fullBlockData, codingParams
        )  # returns a tuple with all the block-specific info not in the file header

        # for each channel, write the data to the output file
        for iCh in range(codingParams.nChannels):
            # determine the size of this channel's data block and write it to the output file
            nBytes = codingParams.nScaleBits  # bits for overall scale factor
            for iBand in range(
                codingParams.sfBands.nBands
            ):  # loop over each scale factor band to get its bits
                nBytes += (
                    codingParams.nMantSizeBits + codingParams.nScaleBits
                )  # mantissa bit allocation and scale factor for that sf band
                if bitAlloc[iCh][iBand]:
                    # if non-zero bit allocation for this band, add in bits for scale factor and each mantissa (0 bits means zero)
                    nBytes += (
                        bitAlloc[iCh][iBand] * codingParams.sfBands.nLines[iBand]
                    )  # no bit alloc = 1 so actuall alloc is one higher
            # end computing bits needed for this channel's data

            # CUSTOM DATA:
            # < now can add space for custom data, if desired>

            # now convert the bits to bytes (w/ extra one if spillover beyond byte boundary)
            if nBytes % BYTESIZE == 0:
                nBytes //= BYTESIZE
            else:
                nBytes = nBytes // BYTESIZE + 1
            self.fp.write(pack("<L", int(nBytes)))  # stores size as a little-endian unsigned long

            # create a PackedBits object to hold the nBytes of data for this channel/block of coded data
            pb = PackedBits()
            pb.Size(nBytes)

            # now pack the nBytes of data into the PackedBits object
            pb.WriteBits(overallScaleFactor[iCh], codingParams.nScaleBits)  # overall scale factor
            iMant = 0  # index offset in mantissa array (because mantissas w/ zero bits are omitted)
            for iBand in range(
                codingParams.sfBands.nBands
            ):  # loop over each scale factor band to pack its data
                ba = bitAlloc[iCh][iBand]
                if ba:
                    ba -= 1  # if non-zero, store as one less (since no bit allocation of 1 bits/mantissa)
                pb.WriteBits(
                    ba, codingParams.nMantSizeBits
                )  # bit allocation for this band (written as one less if non-zero)
                pb.WriteBits(
                    scaleFactor[iCh][iBand], codingParams.nScaleBits
                )  # scale factor for this band (if bit allocation non-zero)
                if bitAlloc[iCh][iBand]:
                    for j in range(codingParams.sfBands.nLines[iBand]):
                        pb.WriteBits(
                            mantissa[iCh][iMant + j], bitAlloc[iCh][iBand]
                        )  # mantissas for this band (if bit allocation non-zero) and bit alloc <>1 so is 1 higher than the number
                    iMant += codingParams.sfBands.nLines[
                        iBand
                    ]  # add to mantissa offset if we passed mantissas for this band
            # done packing (end loop over scale factor bands)

            # CUSTOM DATA:
            # < now can add in custom data if space allocated in nBytes above>

            # finally, write the data in this channel's PackedBits object to the output file
            self.fp.write(pb.GetPackedData())
        # end loop over channels, done writing coded data for all channels

        # write psi_array for M/S rotation
        nBits = codingParams.nPsiBits * codingParams.sfBands.nBands
        if nBits % BYTESIZE == 0:
            nBytes = nBits // BYTESIZE
        else:
            nBytes = nBits // BYTESIZE + 1
        pb = PackedBits()
        pb.Size(nBytes)
        for iBand in range(codingParams.sfBands.nBands):
            pb.WriteBits(psi_array[iBand], codingParams.nPsiBits)
        self.fp.write(pb.GetPackedData())

        return

    def Close(self, codingParams):
        """
        Flushes the last data block through the encoding process (if encoding)
        and closes the audio file
        """
        # determine if encoding or encoding and, if encoding, do last block
        if self.fp.mode == "wb":  # we are writing to the PACFile, must be encode
            # we are writing the coded file -- pass a block of zeros to move last data block to other side of MDCT block
            data = [
                np.zeros(codingParams.nMDCTLines, dtype=np.float64),
                np.zeros(codingParams.nMDCTLines, dtype=np.float64),
            ]
            self.WriteDataBlock(data, codingParams)
        self.fp.close()

    def Encode(self, data, codingParams):
        """
        Encodes multichannel audio data and returns a tuple containing
        the scale factors, mantissa bit allocations, quantized mantissas,
        and the overall scale factor for each channel.
        """
        # Passes encoding logic to the Encode function defined in the codec module
        return codec.Encode(data, codingParams)

    def Decode(self, scaleFactor, bitAlloc, mantissa, overallScaleFactor, psi_array, codingParams):
        """
        Decodes a single audio channel of data based on the values of its scale factors,
        bit allocations, quantized mantissas, and overall scale factor.
        """
        # Passes decoding logic to the Decode function defined in the codec module
        return codec.Decode(
            scaleFactor, bitAlloc, mantissa, overallScaleFactor, psi_array, codingParams
        )


def encode_decode(file, bitRate):
    from pcmfile import PCMFile  # to get access to WAV file handling

    elapsed = time.time()
    for Direction in ("Encode", "Decode"):
        #    for Direction in ("Decode",):

        # create the audio file objects
        if Direction == "Encode":
            print(
                "\n\tEncoding input PCM file...",
            )
            inFile = PCMFile(file)
            outFile = PACFile(f"{file.parent}/coded/{file.stem}_{bitRate}kbps.pac")
        else:  # "Decode"
            print(
                "\n\tDecoding coded PAC file...",
            )
            inFile = PACFile(f"{file.parent}/coded/{file.stem}_{bitRate}kbps.pac")
            outFile = PCMFile(f"{file.parent}/coded/{file.stem}_{bitRate}kbps.wav")
        # only difference is file names and type of AudioFile object

        # open input file
        codingParams = inFile.OpenForReading()  # (includes reading header)

        # pass parameters to the output file
        if Direction == "Encode":
            # set additional parameters that are needed for PAC file
            # (beyond those set by the PCM file on open)
            codingParams.nMDCTLines = 512
            codingParams.nScaleBits = 4
            codingParams.nMantSizeBits = 4
            codingParams.targetBitsPerSample = bitRate / codingParams.sampleRate * 1000
            # tell the PCM file how large the block size is
            codingParams.nSamplesPerBlock = codingParams.nMDCTLines

            # M/S coding
            codingParams.useML = False
            codingParams.nPsiBits = 6
        else:  # "Decode"
            # set PCM parameters (the rest is same as set by PAC file on open)
            codingParams.bitsPerSample = 16
        # only difference is in setting up the output file parameters

        # open the output file
        outFile.OpenForWriting(codingParams)  # (includes writing header)

        # Read the input file and pass its data to the output file to be written
        while True:
            data = inFile.ReadDataBlock(codingParams)
            if not data:
                break  # we hit the end of the input file
            outFile.WriteDataBlock(data, codingParams)
            print(".", end="")  # just to signal how far we've gotten to user
        # end loop over reading/writing the blocks

        # close the files
        inFile.Close(codingParams)
        outFile.Close(codingParams)
    # end of loop over Encode/Decode

    elapsed = time.time() - elapsed
    print("\nDone with Encode/Decode test\n")
    print(elapsed, " seconds elapsed")


def get_side_MDCTLines(data, codingParams):
    from window import SineWindow, HanningWindow
    from rotation import rotational_ms
    from mdct import MDCT
    import scipy.fft

    # get MDCT lines
    halfN = codingParams.nMDCTLines
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
        mdctLines_L, mdctLines_R, fftLines_L, fftLines_R, codingParams.sfBands
    )
    mdctLines_MS = [mdctLines_M, mdctLines_S]
    return mdctLines_MS


def extract_side_MDCTLines_from_wav(file):
    from pcmfile import PCMFile  # to get access to WAV file handling
    from quantize import ScaleFactor

    # create the audio file objects
    print(f"\nExtracting side MDCT from input PCM file: {file}")
    inFile = PCMFile(file)
    # outFile = PACFile(f"{file.parent}/coded/{file.stem}_{bitRate}kbps.pac")

    # open input file
    codingParams = inFile.OpenForReading()  # (includes reading header)

    codingParams.nMDCTLines = 512
    codingParams.nScaleBits = 4
    codingParams.nMantSizeBits = 4
    # codingParams.targetBitsPerSample = bitRate / codingParams.sampleRate * 1000
    # tell the PCM file how large the block size is
    codingParams.nSamplesPerBlock = codingParams.nMDCTLines

    # M/S coding
    codingParams.useML = False
    codingParams.nPsiBits = 6

    sfBands = ScaleFactorBands(
        AssignMDCTLinesFromFreqLimits(codingParams.nMDCTLines, codingParams.sampleRate)
    )
    codingParams.sfBands = sfBands
    # start w/o all zeroes as prior block of unencoded data for other half of MDCT block
    priorBlock = []
    for iCh in range(codingParams.nChannels):
        priorBlock.append(np.zeros(codingParams.nMDCTLines, dtype=np.float64))
    codingParams.priorBlock = priorBlock

    side_mdctlines_array = []
    # Read the input file and pass its data to the output file to be written
    while True:
        data = inFile.ReadDataBlock(codingParams)
        if not data:
            break  # we hit the end of the input file

        # used to be WriteDataBlock, now manually extract the side MDCT lines
        fullBlockData = []
        for iCh in range(codingParams.nChannels):
            fullBlockData.append(np.concatenate((codingParams.priorBlock[iCh], data[iCh])))
        codingParams.priorBlock = data  # current pass's data is next pass's prior block data

        mdctLines_MS = get_side_MDCTLines(fullBlockData, codingParams)

        # compute overall scale factor for M/S block and boost mdctLines using it
        overallScaleFactor = []
        for iCh in range(2):
            maxLine = np.max(np.abs(mdctLines_MS[iCh]))
            overallScale = ScaleFactor(maxLine, codingParams.nScaleBits)
            mdctLines_MS[iCh] *= 1 << overallScale
            overallScaleFactor.append(overallScale)

        side_mdctlines_array.append(mdctLines_MS[1])

    # close the files
    inFile.Close(codingParams)

    side_mdctlines_array = np.array(side_mdctlines_array)
    print(side_mdctlines_array.shape)
    return side_mdctlines_array


# -----------------------------------------------------------------------------

# Testing the full PAC coder (needs a file called "input.wav" in the code directory)
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Process a WAV file for encoding or decoding.")
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the input WAV file. If not specified, will iterate over all the WAV files in '../audio/'.",
    )
    parser.add_argument(
        "--extract_side",
        action="store_true",
    )
    args = parser.parse_args()
    bitRates = [96, 128]
    # bitRates = [128, 192]

    if args.extract_side:
        if args.input_file:
            extract_side_MDCTLines_from_wav(Path(args.input_file))
        else:
            folder_path = Path("../audio")
            side_mdctlines_array = []
            for file in folder_path.iterdir():
                if file.is_file() and file.suffix == ".wav":
                    print(file)
                    side_mdctlines_array.append(extract_side_MDCTLines_from_wav(file))
            side_mdctlines_array = np.concatenate(side_mdctlines_array, axis=0)
            # if exists "../dump.npy", concatenate the new data to it
            if Path("../dump.npy").exists():
                side_mdctlines_array = np.concatenate(
                    [np.load("../dump.npy"), side_mdctlines_array], axis=0
                )
            # write to dump.npy
            np.save("../dump.npy", side_mdctlines_array)
        exit(0)

    if args.input_file:
        for bitRate in bitRates:
            print(f"bitRate = {bitRate}")
            encode_decode(Path(args.input_file), bitRate)
        exit(0)

    folder_path = Path("../audio")
    for file in folder_path.iterdir():
        if file.is_file() and file.suffix == ".wav":
            print(file)
            for bitRate in bitRates:
                print(f"bitRate = {bitRate}")
                encode_decode(file, bitRate)
