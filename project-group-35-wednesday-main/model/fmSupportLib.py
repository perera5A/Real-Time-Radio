#
# Comp Eng 3DY4 (Computer Systems Integration Project)
#
# Department of Electrical and Computer Engineering
# McMaster University
# Ontario, Canada
#

import numpy as np
import math, cmath
from scipy import signal

import matplotlib.pyplot as plt

#
# you should use the demodulator based on arctan given below as a reference
#
# in order to implement your OWN FM demodulator without the arctan function,
# a very good and to-the-point description is given by Richard Lyons at:
#
# https://www.embedded.com/dsp-tricks-frequency-demodulation-algorithms/
#
# the demodulator boils down to implementing equation (13-117) from above, where
# the derivatives are nothing else but differences between consecutive samples
#
# needless to say, you should not jump directly to equation (13-117)
# rather try first to understand the entire thought process based on calculus
# identities, like derivative of the arctan function or derivatives of ratios
#

# def getSamples(input_signal, SPS, prev_samples=None):
#     if prev_samples is None:
#         prev_samples = []

#     # Concatenate previous samples with the new input signal
#     extended_signal = np.concatenate((prev_samples, input_signal))
    
#     samples = []
#     indices = []
#     offset = 0
#     block_count = 0
#     start_idx = SPS // 2  # Starting sample index

#     for i in range(start_idx, len(extended_signal) - SPS, SPS):
#         if block_count % 100 == 0:
#             if i + offset + 1 < len(extended_signal) and abs(extended_signal[i + offset + 1]) > abs(extended_signal[i + offset]):
#                 offset = min(SPS // 2 - 1, offset + 1)
#             elif i + offset - 1 >= 0 and abs(extended_signal[i + offset - 1]) > abs(extended_signal[i + offset]):
#                 offset = max(-(SPS // 2 - 1), offset - 1)
        
#         samples.append(extended_signal[i + offset])
#         indices.append(i + offset)
#         block_count += 1

#     # Determine leftover samples that were not processed
#     last_processed_idx = indices[-1] + offset if indices else start_idx
#     last_processed_idx = last_processed_idx + (SPS // 2 - offset)
#     leftover_samples = extended_signal[last_processed_idx + 1:]  # Get samples after the last processed index

#     return samples, indices, leftover_samples

def getSamples(input_signal, SPS, prev_samples=None):
    offset = 0
    if prev_samples is None:
        prev_samples = []

    # Concatenate previous samples with the new input signal
    extended_signal = np.concatenate((prev_samples, input_signal))
    
    samples = []
    indices = []
    
    block_count = 0
    start_idx = SPS // 2  # Starting sample index
    num_blocks = len(extended_signal) // 26
    
    for block in range(num_blocks):
        # For the first block, find the max absolute value and set the offset
        if block == 0:
            first_block = extended_signal[start_idx + block*26 : start_idx + (block+1)*26]
            max_index = np.argmax(np.abs(first_block))  # Index of max absolute value
            offset = max_index - SPS // 2  # Set offset based on max index
            offset = np.clip(offset, -5, 5)  # Cap the offset between -5 and 5
        
        # Add the sample with the current offset
        samples.append(extended_signal[start_idx + block*26 + offset])
        indices.append(start_idx + block*26 + offset)
    
    # Handle leftover samples if any
    if len(extended_signal) % 26 != 0:
        prev_samples = extended_signal[num_blocks*26:]

    return samples, indices, prev_samples



	
	

def adaptiveSampling(input_signal, SPS):
    samples = []
    sample_indices = []
    block_size = SPS
    total_blocks = len(input_signal) // block_size
    max_index = SPS // 2

    for block_num in range(total_blocks):
        start = block_num * block_size
        end = start + block_size
        block = input_signal[start:end]

        if len(block) < block_size:
            break  # Ignore incomplete block at the end

        if block_num % 200 == 0:
            if abs(input_signal[max_index + 1]) > abs(input_signal[max_index]):
                max_index = min(max_index + 1, 25)  # Ensure max_index does not exceed 26
            elif abs(input_signal[max_index - 1]) > abs(input_signal[max_index]):
                max_index = max(max_index - 1, 0)   # Ensure max_index does not go below 0
        # Sample at the max_index_in_block of the current block
        sample_index = start + max_index
        samples.append(input_signal[sample_index])
        sample_indices.append(sample_index)

    return samples, sample_indices


def manchesterDecoding(input_signal, prev_sample=None, start=1):
    decoded_bits = []
    if prev_sample is not None:
        input_signal = np.concatenate((np.array([prev_sample]), input_signal))

    # Determine correct `start` only on the first call
    if start == 1:
        error1 = 0
        error2 = 0

        # Test assuming input_signal[0] is the first sample
        for i in range(0, len(input_signal) - 1, 2):
            if not ((input_signal[i] > 0 and input_signal[i + 1] < 0) or 
                    (input_signal[i] < 0 and input_signal[i + 1] > 0)):
                error1 += 1

        # Test assuming input_signal[0] is the second sample
        for i in range(1, len(input_signal) - 1, 2):
            if not ((input_signal[i] > 0 and input_signal[i + 1] < 0) or 
                    (input_signal[i] < 0 and input_signal[i + 1] > 0)):
                error2 += 1

        # Choose the start that results in fewer errors
        start = 1 if error1 > error2 else 0

    # Decode Manchester encoding using the chosen start
    for i in range(start, len(input_signal) - 1, 2):
        first_sample = input_signal[i]
        second_sample = input_signal[i + 1]

        if first_sample > 0 and second_sample < 0:
            decoded_bits.append(1)
        elif first_sample < 0 and second_sample > 0:
            decoded_bits.append(0)
        else:
            decoded_bits.append(1)

    # Determine leftover sample for the next function call
    prev_sample = input_signal[-1] if (len(input_signal) - start) % 2 != 0 else None
    start = 0

    return decoded_bits, prev_sample, start




def differentialDecoding(input_bits, state=0):
	previous_bit = state
	output_bits = []

	for i in range(0, len(input_bits)):
		output_bits.append(previous_bit ^ input_bits[i])
		previous_bit = input_bits[i]

	return output_bits, previous_bit


def plot_constellation(i_vals, q_vals):
    # Ensure both arrays have the same length
    if len(i_vals) != len(q_vals):
        raise ValueError("i_vals and q_vals must have the same length")

    # Plot the constellation diagram
    plt.scatter(i_vals, q_vals, alpha=0.5)
    plt.title("Pseudo Constellation Diagram")
    plt.xlabel("In-Phase (i) Values")
    plt.ylabel("Quadrature (q) Values")
    plt.grid(True)
    plt.show()

def process_delay_block(input_block, state_block):

	output_block = np.concatenate((state_block, input_block[:(len(input_block) - len(state_block))]))
	state_block = input_block[-len(state_block):]

	return output_block, state_block

def blockwise_resample(current_block, prev_samples, U, D, coeffs):
    # Concatenate previous tail and current input
    extended_input = np.concatenate((prev_samples, current_block))
    
    # Resample
    output_block = signal.resample_poly(extended_input, up=U, down=D, window=coeffs)
    
    # Save tail for next block
    new_prev_samples = current_block[-(len(coeffs) - 1):]

    return output_block, new_prev_samples

def blockwise_resample_poly(x, state, up, down, h, axis=0):
    """
    Blockwise version of resample_poly using upfirdn with state carryover.

    Parameters
    ----------
    x : ndarray
        Input signal block.
    state : ndarray
        Filter delay line state from previous block.
    up : int
        Upsampling factor.
    down : int
        Downsampling factor.
    h : ndarray
        FIR filter coefficients (already scaled by up).
    axis : int, optional
        Axis along which to apply resampling.

    Returns
    -------
    y : ndarray
        Resampled output block.
    new_state : ndarray
        Updated state to carry into next block.
    """
    # Concatenate old state with current input block
    if state.size > 0:  # Check if state is not empty
        x_padded = np.concatenate((state, x))
    else:
        x_padded = x

    # Resample with polyphase filtering
    y_full = signal.upfirdn(h, x_padded, up=up, down=down, axis=axis)

    # Save new state for next block (last len(h)-1 input samples)
    num_state = len(h) - 1
    new_state = x_padded[-num_state:] if len(x_padded) >= num_state else x_padded

    return y_full, new_state



def convolveFIR_block_polyphase(x, h, down, up, prev_samples):
    h_size = len(h)
    x_size = len(x)
    y_size = (x_size * up) // down
    overlap = len(prev_samples)

    y = [0.0] * y_size

    for i in range(y_size):
        temp = i * down
        phase = (down * i) % up
        fix = (temp - phase) // up

        for k in range(phase, h_size, up):
            index = temp - k
            if index >= 0:
                if 0 <= fix < x_size:
                    y[i] += h[k] * x[fix]
            else:
                prev_index = overlap + fix
                if 0 <= prev_index < overlap:
                    y[i] += h[k] * prev_samples[prev_index]
            fix -= 1

    # Update prev_samples for next block processing
    num_samples_to_save = math.ceil((h_size - 1) / up)
    start_index = max(x_size - num_samples_to_save, 0)

    # Ensure that prev_samples holds only the number of required samples
    if len(prev_samples) < num_samples_to_save:
        prev_samples.extend([0.0] * (num_samples_to_save - len(prev_samples)))  # pad with zeros if needed

    # Save the relevant portion of x into prev_samples
    prev_samples[:] = x[-len(prev_samples)]

    return y




def upsampler(upsample_size, input_signal):
	output_signal = np.zeros(len(input_signal) * upsample_size, dtype = int)
	for i in range(len(input_signal)):
		output_signal[i * upsample_size] = input_signal[i]
	return output_signal

# use the four quadrant arctan function for phase detect between a pair of
# IQ samples; then unwrap the phase and take its derivative to FM demodulate
def fmDemodArctan(I, Q, prev_phase = 0.0):

	# the default prev_phase phase is assumed to be zero, however
	# take note in block processing it must be explicitly controlled

	# empty vector to store the demodulated samples
	fm_demod = np.empty(len(I))

	# iterate through each of the I and Q pairs
	for k in range(len(I)):

		# use the atan2 function (four quadrant version) to detect angle between
		# the imaginary part (quadrature Q) and the real part (in-phase I)
		current_phase = math.atan2(Q[k], I[k])

		# we need to unwrap the angle obtained in radians through arctan2
		# to deal with the case when the change between consecutive angles
		# is greater than Pi radians (unwrap brings it back between -Pi to Pi)
		[prev_phase, current_phase] = np.unwrap([prev_phase, current_phase])

		# take the derivative of the phase
		fm_demod[k] = current_phase - prev_phase

		# save the state of the current phase
		# to compute the next derivative
		prev_phase = current_phase

	# return both the demodulated samples as well as the last phase
	# (the last phase is needed to enable continuity for block processing)
	return fm_demod, prev_phase

# custom function for DFT that can be used by the PSD estimate
def DFT(x):

	# number of samples
	N = len(x)

	# frequency bins
	Xf = np.zeros(N, dtype='complex')

	# iterate through all frequency bins/samples
	for m in range(N):
		for k in range(N):
			Xf[m] += x[k] * cmath.exp(1j * 2 * math.pi * ((-k) * m) / N)

	# return the vector that holds the frequency bins
	return Xf

# custom function to estimate PSD based on the Bartlett method
# this is less accurate than the Welch method used in some packages
# however, as the visual inspections confirm, the estimate gives
# the user a "reasonably good" view of the power spectrum
def estimatePSD(samples, NFFT, Fs):

	# rename the NFFT argument (notation consistent with matplotlib.psd)
	# to freq_bins (i.e., frequency bins for which we compute the spectrum)
	freq_bins = NFFT
	# frequency increment (or resolution of the frequency bins)
	df = Fs / freq_bins

	# create the frequency vector to be used on the X axis
	# for plotting the PSD on the Y axis (only positive freq)
	freq = np.arange(0, Fs / 2, df)

	# design the Hann window used to smoothen the discrete data in order
	# to reduce the spectral leakage after the Fourier transform
	hann = np.empty(freq_bins)
	for i in range(len(hann)):
		hann[i] = 0.5 * (1 - math.cos(2 * math.pi * i / (freq_bins - 1)))

	# create an empty list where the PSD for each segment is computed
	psd_list = []

	# samples should be a multiple of frequency bins, so
	# the number of segments used for estimation is an integer
	# note: for this to work you must provide an argument for the
	# number of frequency bins not greater than the number of samples!
	no_segments = int(math.floor(len(samples) / float(freq_bins)))

	# iterate through all the segments
	for k in range(no_segments):

		# apply the hann window (using pointwise multiplication)
		# before computing the Fourier transform on a segment
		windowed_samples = samples[k * freq_bins:(k + 1) * freq_bins] * hann

		# compute the Fourier transform using the built-in FFT from numpy
		Xf = np.fft.fft(windowed_samples, freq_bins)

		# since input is real, we keep only the positive half of the spectrum
		# however, we will also add the signal energy of negative frequencies
		# to have a better and more accurate PSD estimate when plotting
		Xf = Xf[0:int(freq_bins / 2)] # keep only positive freq bins
		psd_seg = (1 / (Fs * freq_bins / 2)) * (abs(Xf)**2) # compute signal power
		psd_seg = 2 * psd_seg # add the energy from the negative freq bins

		# append to the list where PSD for each segment is stored
		# in sequential order (first segment, followed by the second one, ...)
		psd_list.extend(psd_seg)

	# iterate through all the frequency bins (positive freq only)
	# from all segments and average them (one bin at a time ...)
	psd_seg = np.zeros(int(freq_bins / 2))
	for k in range(int(freq_bins / 2)):
		# iterate through all the segments
		for l in range(no_segments):
			psd_seg[k] += psd_list[k + l * int(freq_bins / 2)]
		# compute the estimate for each bin
		psd_seg[k] = psd_seg[k] / no_segments

	# translate to the decibel (dB) scale
	psd_est = np.zeros(int(freq_bins / 2))
	for k in range(int(freq_bins / 2)):
		psd_est[k] = 10 * math.log10(psd_seg[k])

	# the frequency vector and PSD estimate
	return freq, psd_est

# custom function to format the plotting of the PSD
def fmPlotPSD(ax, samples, Fs, height, title):

	# adjust grid lines as needed
	x_major_interval = (Fs / 12)
	x_minor_interval = (Fs / 12) / 4
	y_major_interval = 20
	x_epsilon = 1e-3
	# adjust x/y range as needed
	x_max = x_epsilon + Fs / 2
	x_min = 0
	y_max = 10
	y_min = y_max - 100 * height

	ax.psd(samples, NFFT=512, Fs=Fs)
	#
	# below is the custom PSD estimate, which is based on the Bartlett method
	# it is less accurate than the PSD from matplotlib, however it is sufficient
	# to help us visualize the power spectra on the acquired/filtered data
	#
	# freq, my_psd = estimatePSD(samples, NFFT=512, Fs=Fs)
	# ax.plot(freq, my_psd)
	#
	ax.set_xlim([x_min, x_max])
	ax.set_ylim([y_min, y_max])
	ax.set_xticks(np.arange(x_min, x_max, x_major_interval))
	ax.set_xticks(np.arange(x_min, x_max, x_minor_interval), minor=True)
	ax.set_yticks(np.arange(y_min, y_max, y_major_interval))
	ax.grid(which='major', alpha=0.75)
	ax.grid(which='minor', alpha=0.25)
	ax.set_xlabel('Frequency (kHz)')
	ax.set_ylabel('PSD (db/Hz)')
	ax.set_title(title)

##############################################################
# New code as part of benchmarking/testing and the project
##############################################################

# custom function to estimate PSD using the matrix approach
def matrixPSD(samples, NFFT, Fs):

	freq_bins = NFFT
	df = Fs / freq_bins
	freq = np.arange(0, Fs / 2, df)
	no_segments = int(math.floor(len(samples) / float(freq_bins)))

	# generate the DFT matrix for the given size N
	dft_matrix = np.empty((freq_bins, freq_bins), dtype='complex')
	for m in range(freq_bins):
		for k in range(freq_bins):
			dft_matrix[m, k] = cmath.exp(1j * 2 * math.pi * ((-k) * m) / freq_bins)

	# generate the Hann window for the given size N
	hann_window = np.empty(freq_bins, dtype='float')
	for i in range(freq_bins):
		hann_window[i] = 0.5 * (1 - math.cos(2 * math.pi * i / (freq_bins - 1)))

	# apply Hann window and perform matrix multiplication using nested loops
	Xf = np.zeros((no_segments, freq_bins), dtype='complex')
	for seg in range(no_segments):
		for m in range(freq_bins):
			for k in range(freq_bins):
				Xf[seg][m] += samples[seg * freq_bins + k] * hann_window[k] * dft_matrix[m][k]

	# compute power, keep only positive frequencies, average across segments, and convert to dB
	psd_est = np.zeros(int(freq_bins / 2))  # same as (freq_bins // 2)
	for m in range(freq_bins // 2):
		sum_power = 0.0
		for seg in range(no_segments):
			sum_power += (1 / ((Fs / 2) * (freq_bins / 2))) * (abs(Xf[seg][m]) ** 2)
		psd_est[m] += 10 * math.log10(sum_power / no_segments)

	return freq, psd_est

# function to unit test PSD estimation
def psdUnitTest(min=-1, max=1, Fs=1e3, size=1024, NFFT=128):

	# generate random samples for testing
	samples = np.random.uniform(low=min, high=max, size=size)

	# calculate reference PSD
	freq_ref, psd_ref = estimatePSD(samples, NFFT, Fs)

	# calculate PSD using the matrix-based function
	freq_mat, psd_mat = matrixPSD(samples, NFFT, Fs)

	# check if all the values are close within the given tolerance
	if not np.allclose(freq_ref, freq_mat, atol=1e-4):
		print("Comparison between reference frequency vectors fails")

	if not np.allclose(psd_ref, psd_mat, atol=1e-4):
		print("Comparison between reference estimate PSD and matrix PSD fails")
		print("Reference PSD:", psd_ref)
		print("Matrix PSD   :", psd_mat)
		print("Maximum difference:", np.max(np.abs(psd_ref - psd_mat)))
	else:
		print(f"Unit test for matrix PSD transform passed.")

if __name__ == "__main__":

	'''
	# this unit test (when uncommented) will confirm that
	# estimate PSD and matrix PSD are equivalent to each other
	psdUnitTest()
	'''

	# do nothing when this module is launched on its own
	pass
