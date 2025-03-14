The `PACFile` class is the entry for handling audio files compressed using an MDCT-based perceptual audio coding algorithm.

## Key Features

Baseline codec:

- **MDCT-based Compression**: Utilizes Modified Discrete Cosine Transform (MDCT) for audio compression.
- **Scale Factor Bands**: Groups MDCT lines into bands that share a single scale factor and bit allocation.
- **Block-Floating Point Quantization**: Quantizes MDCT lines using block-floating point techniques.

Additional features implemented:

- **Rotational M/S (Mid/Side) Coding**: Supports rotational M/S stereo coding for improved compression efficiency.
- **Noise Masker Identification**: Implemented for better psychoacoustic modeling.

## Usage

### Encoding and Decoding

To encode a WAV file into a PAC file and then decode it back to WAV:

```python
from pacfile import encode_decode

# Encode and decode a WAV file at 128 kbps
encode_decode("input.wav", 128)
```

### Extracting Side MDCT Lines for NN Training

To extract side MDCT lines from a WAV file:

```python
from pacfile import extract_side_MDCTLines_from_wav

# Extract side MDCT lines from a WAV file
side_mdctlines = extract_side_MDCTLines_from_wav("input.wav")
```

### Command Line Interface

The script can also be run from the command line:

```bash
python pacfile.py --input_file input.wav
```

For batch processing of all WAV files in a directory (defaults to `../audio/`):

```bash
python pacfile.py
```

To extract side MDCT lines from a WAV file:

```bash
python pacfile.py --input_file input.wav --extract_side
```

## File Format

### Header

Baseline codec:

- **tag**: 4-byte file tag equal to "PAC ".
- **sampleRate**: Sample rate of the audio.
- **nChannels**: Number of audio channels.
- **numSamples**: Total number of samples in the file.
- **nMDCTLines**: Half the MDCT block size.
- **nScaleBits**: Number of bits storing scale factors.
- **nMantSizeBits**: Number of bits storing mantissa bit allocations.
- **nSFBands**: Number of scale factor bands.

Additional headers:

- **useML**: Whether to switch to NN encoding and decoding.
- **nPsiBits**: Number of bits for the angle in the M/S coding.

### Data Block

Baseline codec:

- **nBytes**: Number of bytes of data for the channel.
- **overallScale**: Overall scale factor.
- **scaleFactor**: Scale factor for each band.
- **bitAlloc**: Bit allocation for each band.
- **mantissa**: Quantized mantissas for each band.

Additional data:

- **psi_array**: Array of angles for M/S coding.
