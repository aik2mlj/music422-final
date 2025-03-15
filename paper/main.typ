#import "@preview/charged-ieee:0.1.3": ieee
#import "@preview/subpar:0.2.1"
#show link: underline
#set math.mat(delim: "[")
#set math.vec(delim: "[")

#show: ieee.with(
  title: [Leveraging Rotational M/S Coding and Machine Learning in Stereo Audio Coding],
  abstract: [In stereo coding, it is essential to consider binaural correlation in order to decrease the bit rate while achieving good audio quality. Traditional M/S coding performs poorly when there isn't a significant difference between the calculated mid and side channel. We implemented a rotational M/S coding strategy where an adaptive matrixing scheme is used to ensure the significance of M/S difference. The psychoacoustic model then takes binaural unmasking effects into account, applying Binaural Masking Level Difference (BMLD) to correct the signal-to-mask ratio (SMR) calculation for the side channel. Noise masker identification is also added for more accuracy. The listening experiment shows that our coding achieves better subjective quality given the same bit rate compared to the baseline model, especially in the scenarios where the audio is extremely panned or noisy (e.g., speech). In the end, we also experimented integrating several neural networks, including Convolutional Autoencoder and Vector Quantized Variational Autoencoder (VQ-VAE) model to encode the side channel signal.],
  authors: (
    (
      name: "Si Qi Chen",
      organization: [CCRMA, Stanford University],
      email: "siqichen@ccrma.stanford.edu"
    ),
    (
      name: "Lejun Min",
      organization: [CCRMA, Stanford University],
      email: "lejun@ccrma.stanford.edu"
    ),
  ),
  index-terms: ("Perceptual audio coding", "Stereo coding", "Machine Learning"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Figure],
)


= Motivation

We start with a baseline codec implemented in #link("https://ccrma.stanford.edu/courses/422")[Music 422] _(Perceptual Audio Coding)_ mostly following the MPEG-1 standard, with components including MDCT, psychoacoustic model for masking effect detection, block floating point quantization, and bit allocation. The baseline codec encodes multichannel audio samples separately, resulting in redundancies for highly similar multichannel audio files. Therefore, the goal of this project is to remove such redundancy by applying an optimized M/S coding using a combination of rotational M/S coding @rotational_ms and a neural network to further compress the resulting side channels. Additionally, noise maskers are implemented and tweaked for more accurate masked threshold estimation. We expect to significantly improve the quality under the same compression rate for
1. highly correlated stereo audio files with high redundancies between the left and right tracks;
2. robustness and quality improvement for stereo audio files with a heavily panned or moving sound source, where standard M/S coding fails.
3. noisy scenarios like German speech.


= Methods <sec:methods>

Our codec explores extensively the application of stereo coding for low bit-rates. @sec-standard-ms and @sec-intensity-coding are introductions of relevant stereo coding methods that precede the method we used in our codec. We implemented the method of rotational M/S coding (referred to as Adaptive Matrixing in the original paper @rotational_ms), the detail of which is described in @sec-rotational-ms. To address the difference in psychoacoustic masking for stereo coding, we modified our psychoacoustic model (@sec-stereo-psychoac) and improved the noise masker (@sec-noise-masker) which was attempted but not used in the baseline coder. In @sec-machine-learning, we describe our attempts and ideas on using neural networks for more significant compression of the resulting side channels.

== Standard M/S Stereo Coding
<sec-standard-ms>

Mid/Side (M/S) stereo coding is a widely used technique in audio signal processing that enhances the efficiency of stereo audio compression. Instead of encoding the left (L) and right (R) audio channels separately, M/S stereo coding transforms the stereo signal into a sum-and-difference representation:

$ M=(L+R) / 2, quad S=(L−R) / 2 $

where the mid (M) component represents the monophonic content shared between both channels, while the side (S) component encodes the stereo width or spatial differences. Since many stereo signals exhibit strong correlation between the left and right channels, the mid signal often carries the majority of the perceptual audio information, while the side signal contains lower-energy residuals. This transformation allows for a more efficient distribution of bit allocation in perceptual audio codecs. This technique is particularly advantageous for highly correlated stereo recordings, such as orchestral music or speech signals, where the side channel remains sparse, leading to improved compression efficiency.

However, standard M/S stereo coding performs poorly when the audio signal is extremely panned to one side, as the mid and side signal will be distributed evenly in this scenario, yielding no optimization for bit allocation. Therefore, M/S coding fails to capture the monophonic content and spatial differences when the audio involves complex panning or moving sound source.

== Intensity Coding
<sec-intensity-coding>

Intensity stereo coding @intensitycoding is a perceptual audio coding technique designed to reduce bit rate by exploiting the way the human auditory system perceives stereo sound, particularly at high frequencies. Instead of encoding both the left and right channels separately, intensity coding represents stereo information using a single channel (typically a weighted sum of the original channels $M = (L+R)/2$) along with directional cues. These cues are applied as time-varying amplitude scaling factors to recreate the perceived spatial positioning of the sound.

Compared to standard M/S coding which maps pairs of vectors after time-frequency transform representing individual channels into orthogonal axis, intensity coding explores more efficient ways to represent intensity differences between the left and right stereo channels, using calculation of intensity ratios $ C_I =  R_k / L_k $ where $R_k$ and $L_k$ are averages of the intensity of MDCT lines in each critical frequency band. Discarding extra side (phase) information, intensity coding calculates an intensity channel $ I_k = L_k + R_k $ and a rotation angle $ theta_k = arctan(R_k/L_k) $ to encode a mid channel and stereo panning. The key assumption behind this method is that, at higher frequencies, the human auditory system relies more on intensity differences than phase differences for spatial localization. By discarding phase information and only encoding intensity variations, intensity stereo coding achieves significant bit-rate savings while maintaining an acceptable level of perceptual stereo separation. Intensity stereo coding eliminates phase information entirely and relies on amplitude panning to simulate stereo width, which can lead to a loss of spatial accuracy, particularly in complex stereo recordings. Therefore, it is preferred in low-bitrate scenarios where significant data reduction is required, such as speech coding or highly compressed streaming audio.


== Rotational M/S Coding (Adaptive Matrixing)
<sec-rotational-ms>

The method we implemented, rotational M/S coding, shares the same overarching goal as conventional M/S coding and Intensity coding—namely, to maximize the energy in a mid channel. Then, by assigning more bits to the mid channel, we can thereby reduce quantization noise @rotational_ms. 

For each block of time-frequency transformed samples, the two input channels are first divided into 25 sub-bands. 
Then, within each sub-band $Omega_k$, we compute the covariance matrix:

$
#strong[R]_(x x)^k = 1/(abs(Omega_k)) sum_(harpoon(x) in Omega_k) harpoon(x) harpoon(x)^T = mat(r_"LL", r_"LR"; r_"RL", r_"RR")
$
where
$ harpoon(x) = vec(x_L, x_R) $

and $x_L, x_R$ are the magnitude of each MDCT line in that sub-band. We then use the matrix $#strong[R]_(x x)^k$ to determine the optimal angle of rotation, such that the transformed mid channel aligns with the eigenvector of this covariance matrix:

$
phi_k = 1/2 arctan((r_"LR" + r_"RL") / (r_"LL" - r_"RR"))
$

$r_".."$ being the entries of $#strong[R]_(x x)^k$, and $phi_k$ the angle of rotation. This calculation using the covariance matrix takes into account the correlation and phase differences between the left and right channels, making it mathematically more robust than simply comparing the average magnitudes, as occurs in Intensity coding.

After obtaining $phi_k$, we uniformly quantize them using 4-bit midtread quantization, so that we need not to worry about bit allocation later in the process. The quantized angle $hat(phi_k)$ is then used to compute the rotation matrix:

$
#strong[H]_k = mat(cos(hat(phi_k)), sin(hat(phi_k)); -sin(hat(phi_k)), cos(hat(phi_k)))
$

which we apply to each pair of left-right MDCT values represented as coefficients, $harpoon(x) = [x_L, x_R]$. 

The resulting mid-side signals

$
vec(x_M, x_S) = #strong[H] vec(x_L, x_R)
$

are separately passed through scaling, bit allocation, and quantization stages, and finally to the bit-stream.

To restore the left and right channels in the decoder, we simply invert the rotation:

$
vec(x_L, x_R) = #strong[H]^T vec(x_M, x_S)
$

=== Why do we choose this method? 

This rotational M/S technique is advantageous because minimizing the amount of information carried by the side channels allows more aggressive compression, trusting that overall decoded quality is primarily determined by the mid channel (which aligns with the methodology of traditional M/S coding). Conventional M/S coding are not very effective for cases where left and right channels differ significantly, making it necessary to look for alternative methods for allowing aggressive compression by a neural network. 


By adapting the rotation to each frequency sub-band, we more consistently capture the bulk of the stereo information in the mid channel. Moreover, when coupled with a neural-network-based bit-allocation or post-processing stage, the method can theoretically surpass the compression efficiency of BCC (Binaural Cue Coding), which typically requires a dimensionality of at least 3 times the number of critical bands for parameter transmission @BCC.


=== Lower-Bound on Method Performance

A useful way to estimate the performance floor of this method is to assume that the side channel $x_S$ is zero. In this hypothetical case, the decoded audio would recover only the intensity differences (no correlation cues), yielding a result similar to that of Intensity coding. 

Consider the vector $harpoon(x)_mid = [x_M, 0]$, the restored left and right channels will be:


$
vec(x_L, x_R) = #strong[H]^T vec(x_M, 0) = vec(cos(hat(phi_k))*x_M, sin(hat(phi_k))*x_M),
$
which captures only the intensity variations. Hence, theoretically, this approach cannot perform much "worse" than Intensity coding, since the limiting scenario effectively defaults back to an Intensity-based representation of the stereo image (assuming bit allocation is clever enough).

== Stereo Psychoacoustic Model
<sec-stereo-psychoac>

When adapting a psychoacoustic model originally designed for single-channel (monaural) audio coding to accommodate Mid/Side (M/S) stereo coding, it's essential to incorporate considerations of binaural auditory perception, since human auditory perception differs when processing sounds binaurally (with both ears) compared to monaurally. Especially, binaural masking refers to the phenomenon where the brain integrates auditory information from both ears to detect sounds more effectively in noisy environments. This integration can improve the detection threshold of a signal when there are differences in interaural phase or time delays between the signal and the noise, a phenomenon known as the Binaural Masking Level Difference (BMLD).

The BMLD can provide up to a 15 dB improvement at low frequencies (around 250 Hz) and decreases with increasing frequency. According to the result in @BMLD, the BMLD for pure tones in broadband noise reaches a maximum value of about 15 dB at 250 Hz and progressively declines to 2-3 dB at 1500 Hz. The BMLD then stabilises at 2-3 dB for all higher frequencies, up to at least 4 kHz. We implemented this correction to our side channel frequency lines after MDCT, ensuring that the calculation of signal-to-mask ratio (SMR) takes binaural unmasking effect into consideration.

== Noise Maskers in the Psychoacoustic Model
<sec-noise-masker>

In addition to the tonal maskers we implemented in the baseline model, noise maskers are added to take noise masking effects into consideration. Different from the tonal maskers where the peak masking threshold drops $Delta = 6$ from the sound pressure level (SPL) of the tonal peak, noise maskers has a drop of $Delta=16$, yielding a larger SMR value. In each sub-band, after the peak detection searching for tonal peaks, the remaining MDCT lines are counted as noise lines. The SPL of the noise signal is thus calculated from their summed-up intensity. The center frequency of the noise masker is the geometric mean of the frequencies within this sub-band. Therefore, each sub-band provides a noise masker.

When integrating all the masking thresholds and threshold in quiet into the masked threshold, $alpha = oo$ is used, meaning that the combined masked threshold is the maximum value of all the thresholds along the sub-bands.

== Machine Learning Attempts for Learning side Channel Representation
<sec-machine-learning>

In this section, we will outline our exploration of machine learning strategies for representing the side-channel MDCT coefficients in stereo audio coding. Note that due to the limitation on data and time, we did not implement the neural network in our final codec used for the listening test. However, we will present some preliminary results and evaluations based on our experiments so far, and will discuss in the conclusion some potential future works on this method.

This approach draws inspiration from the Binaural Cue Coding (BCC) framework @BCC, that is, to encode only one monophonic channel while using lower-dimensional parametric representations for positional information. These parameters—namely, the Inter-Channel Time Difference (ICTD), Inter-Channel Level Difference (ICLD), and Inter-Channel Correlation (ICC)—are computed from the MDCT coefficients of the left and right channels using a time–frequency analysis that mimics critical-band filtering and perceptual auditory models. 

Although BCC achieves high compression rates, it is known to introduce perceptible artifacts @rotational_ms. Therefore, we are motivated to investigate the potential of further compressing the side-channel information using autoencoder architectures, hoping that the neural network could learn meaningful representation $Z_S$ for the stereo differences between the left and right channels.

After implementing rotational M/S coding, we experimented with encoding the side channel on a per-block basis using several neural networks, including a simple Multi-Layer Perceptron (MLP) Autoencoder, a Convolutional Autoencoder, and a Vector Quantized Variational Autoencoder (VQ-VAE). 

In terms of loss design, the MLP and Convolutional Autoencoder are trained on the Mean-Square Error loss:
$
L = L_"MSE" = 1/N sum_(i=1)^N (y_i - hat(y)_i)^2
$


which evaluates how close the reconstructed signal $X'_S$ is to the original input $X_S$. 

For the loss of the VQ-VAE, two more terms are added according to @Gosthipaty2024: the reconstruction loss for the difference between the encoded vector and the vector-quantized representation of that vector in the codebook, $L_"VQ"$, also calculated using mean-square error, and a commitment loss, $L_"commit"$, that enforces the encoder network to commit to a specific representation in the codebook:

$
L = L_"MSE" + L_"VQ" + L_"commit"
$


#figure(
  image("figs/nn.drawio.png", width: 60%),
  caption: [Neural network training diagram.],
) <fig-nn>

In short, the most important loss for our case is how well the model reconstructs the input side signal. However, it is possible to explore more varieties such as perceptual loss used in many neural codec research. 


= Codec Implementation

Compared to the baseline codec, we add rotational M/S coding and corresponding correction to the psychoacoustic model. We then provide two versions of codec design: with machine learning model, and without.

#figure(
  image("figs/codec.drawio.png"),
  caption: [Encoder _without_ machine learning module (NN) in audio codec design.],
) <fig-codec>

#figure(
  image("figs/codec_nn.drawio.png"),
  caption: [Encoder _with_ machine learning module (NN) in audio codec design.],
) <fig-codec-nn>

== Rotational M/S Coding in Codec Design

As shown in @fig-codec, after MDCT transformation and scale factor grouping, the left and right frequency components $X_L$ and $X_R$ are used to calculate the covariance matrix $R_(x x)^k$, which yields the angles of rotation $phi_k$. Then, the quantized angles $hat(phi)_k$ is calculated and used for data partitioning.

To cooperate the quantized angles of rotation into the data block, we added a parameter _nPhiBits_ (defaults to 4) in the _PAC_ file header, and write the array $hat(phi)_k$ after the mantissa bits are stored.

== Psychoacoustic Model and Bit Allocation

Notedly, we also fed the array $hat(phi)_k$ to the perceptual model as in @fig-codec. This is due to that the frequency lines calculated by FFT need to be transformed with the covariance matrix accordingly, so that the SMR values correctly correspond to the rotational M/S channels.

As stated in @sec-stereo-psychoac, a BMLD correction is applied to the side channel masker identification. From the calculated SMR values, we allocate the bit budget between M and S channels based on their respective masking thresholds and perceptual importance. The mid channel carries more critical information for overall sound quality, thus being distributed more bits. The split is calculated according to this equation:
$ R_"ch"^"opt" = R_"total"/2 + ln(10)/(20ln(2)) ("SMR"_"ch" - 1/2 "SMR"_"total") $
where "ch" stands for M/S channels, $R$ stands for distributed bit budget. $"SMR"_"ch"$ is the summation of SMR values of each sub-band in a channel, and $"SMR"_"total"$ is the addition of $"SMR"_"ch"$ of two channels. The equation is a special case of the equation in terms of each sub-band's SMR in page 217, @textbook. Specially, when this equation gives a negative bit budget for the side channel, we correct it to zero and adjust the mid channel budget accordingly.

== Neural Network Integration

The integration of the neural network is still work-in-progress. We briefly explain our concept here for the continuing future work. First, a parameter _useML_ is added in the _PAC_ file header to indicate whether to switch to machine learning method in encoding and decoding. As illustrated in @fig-codec-nn, the neural network extracts the latent representation $Z_S$ for each MDCT block of the side channel $X_S$. Then, according to the model types, we plan to choose different strategy to store the latent representation $Z_S$.

=== Non-quantized $Z_S$

For MLP and Convolutional Autoencoder, the encoded latent representation $Z_S$ is an array of floating point values. Therefore, it needs further quantization to be written to the _PAC_ data blocks. Our primary trial will be using a floating point quantization with fixed scale factor and mantissa bits.

=== Quantized $Z_S$

VQ-VAE has the advantage that the latent representation $Z_S$ is chosen from a fixed-length codebook, which means that instead of the floating point latent values, the index of the chosen latent from the codebook can be directly passed to data partitioning without quantization. From the perspective of model design, this approach aligns better with the purpose of involving neural network in the codec.

= Experiments and Results

== Critical Listening Materials

In the listening test, we selected 5 critical samples, four of which are from Sound Quality Assessment Materials (SQAM) and one of which a custom-made sample.
- _castanets_: A fast-paced playing recording of castanets, for critical assessment of percussive sounds.
- _glockenspiel_: A series of glockenspiel (bell) notes in high pitches, for assessment of high frequency and transient sounds.
- _harpsichord_: A tonal instrument with a rich spectral timbre, for assessment of highly tonal signals.
- _spgm_: A segment of German male speech, for assessment of human voice (especially noisy, consonant-rich pronunciation).
- _pannedCello_: A custom-made sample where a segment of cello recording is panned around, for assessment of heavily panned and moving sounds.

== Experiment Implementation

We implemented an ITU-R BS.1116-based impairment test, which is a subjective assessment method designed to evaluate impairments in audio systems. We employed a double-blind triple-stimulus with hidden reference approach, where listeners are presented with three stimuli: the original reference signal, the compressed signal, and a hidden reference identical to the original. Listeners are tasked with identifying the hidden reference and rating the impairments of the test signal compared to the reference using a continuous five-grade impairment scale ranging from 5.0 (imperceptible) to 1.0 (very annoying). We refer readers to Chapter 10 in @textbook for more details.

The experiment is conducted using monitoring headphone or at the Ballroom and Studio D at CCRMA, Stanford. The participants are either experienced in the domain of audio coding, or lightly trained before the experiment.

Two models are tested during the listening test: the proposed codec _without_ neural network module, and the baseline codec. The proposed codec _with_ neural network module is not included due to the insufficient learning results shown at @sec-nn-result as well as the limited timeline.

#subpar.grid(
  figure(image("figs/128kbps_ours.png")), <128kbps_ours>,
  figure(image("figs/128kbps_bl.png")), <128kbps_bl>,
  figure(image("figs/96kbps_ours.png")), <96kbps_ours>,
  figure(image("figs/96kbps_bl.png")), <96kbps_bl>,
  caption: [The listening test results under bit rate of 128 kbps and 96 kbps. Tested on the proposed codec without neural network module (ours) and the baseline codec (bl). The y-axis indicates the subjective difference grade (SDG) of the tested codec.],
  placement: top,
  scope: "parent",
  columns: (1fr, 1fr),
  label: <fig-results>,
)

== Result Analysis

As shown in @fig-results, it can be seen that overall, our codec with rotational M/S coding and noise masking slightly outperforms the baseline codec at low bit-rate (96 kbps). Notably, the results for _spgm_, the German speaker sample, is significantly better. We conclude that this is due to the implementation of noise maskers. On the _pannedCello_ sample that is specially designed to assess binaural coding, our proposed codec achieves much better result at 96kbps, suggesting that our rotational M/S coding does improve stereo coding on heavily panned or moving sounds. The improvements are consistent with our hypothesis, since the advantage of saving more bits are more apparent for low bit-rates.

The performance on the other three samples (_castanets_, _glockenspiel_, _harpsichord_) mildly degrades. First, we argue that these materials are not the targets for our improvement on the codec, since they do not show significant L/R difference. Also, admittedly, while introducing rotational M/S coding can optimize bit allocation among the mid and side channels, it may also introduce "rotational" artifact after applying the quantization. Finally, As for the percussive samples like _castanets_ and _glockenspiel_, we speculate that due to the less bit allocation for the side channel, the pre-echo becomes thus more hearable than the baseline model, yielding worse subjective quality.

== Dataset Construction for Machine Learning

The training and validation dataset is made on the samples of Sound Quality Assessment Materials (SQAM). We extracted the MDCT values of the side channel from these samples, and stored them on a `.npy` file. Given a block size of 1024, the extracted MDCT values are 1-d array of length 512. In total, the dataset contains 227931 blocks, each of 512 MDCT values. We recognize that this is a limited dataset that may not be able to fully optimize the model, but it can be a trial for proving of concept.


== Neural Network Training Results
<sec-nn-result>

We trained the three neural networks to encode the 512-dimensional input vector consisted of the 512 MDCT Lines from the side channel into a 64-dimensional embedding in the latent space. For the VQ-VAE, we chose a codebook with 64-dimensional latent vectors and a total number of embeddings of 1024. 

We include the loss curve of first 100 epochs for all three models here for comparison. Out of the three models, the Convolutional Autoencoder has shown potential to achieving a trainable loss curve, indicating that we could potentially train it on a larger dataset to achieve better results. We will provide brief description of our experiments for each method below.

=== MLP Autoencoder


#figure(
  image("figs/mlp.png"),
  caption: [Training and Validation loss for the first 100 epochs of training the MLP Autoencoder. lr=0.001] ,
) <fig-mlpLoss>

As shown in @fig-mlpLoss, the training and validation loss of the MLP Autoencoder did not improve much for the first 100 epochs (less than 0.02). It seems to have difficulty learning good representations for the side signal in a compressed latent space.  


=== Convolutional Autoencoder

#figure(
  image("figs/conv.png"),
  caption: [Training and Validation loss for the first 100 epochs of training the Convolutional Autoencoder, lr=0.001],
) <fig-convLoss>

@fig-convLoss shows that the Convolutional Autoencoder is the most promising out of the three neural networks, with the first 30 epochs showing significant improvement in reconstruction loss (an improvement of more than 0.2 in ~20 epochs, also consistent with training loss). However, after the 30th epoch, the validation loss started to increase while the training loss keeps decreasing, suggesting that the network is likely over-fitting. To address this issue in future works, we could try increasing the number training data, allowing the network to learn from more diverse inputs.

=== VQ-VAE

#figure(
  image("figs/vqvae.png"),
  caption: [VQ-VAE training reconstruction loss over the first 83 epochs]
) <fig-vqvae.png>



After extensive hyperparameter tuning and loss function exploration for the VQ-VAE, we have not yet achieved successful training of this architecture. Nonetheless, we remain optimistic that a more carefully designed loss term could enable effective training, and that this model is worth exploring given the potential for significant bit-rate reduction from the integrated codebook layer.

= Conclusions

In conclusion, the listening test results show that our implementation of rotational M/S coding efficiently allocates bits under low bit-rate setting, yielding better performance on heavily panned or moving sounds; adding noise maskers achieves better quality on noisy samples like German male speech. 

For the trial on machine learning, our proof-of-concept experiments indicate that NN-based methods, despite current limitations, show promise. Both the Convolutional Autoencoder and VQ-VAE architectures might benefit from further investigation. Future work should consider incorporating perceptual loss for improved loss calculation, refining bit-allocation strategies for neural network integration, and exploring sequential modeling over multiple blocks to potentially enhance compression performance at the cost of encoding-decoding speed.

