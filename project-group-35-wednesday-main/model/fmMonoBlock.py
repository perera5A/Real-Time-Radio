#
# Comp Eng 3DY4 (Computer Systems Integration Project)
#
# Copyright by Nicola Nicolici
# Department of Electrical and Computer Engineering
# McMaster University
# Ontario, Canada
#

import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy as np
import math

# use fmDemodArctan and fmPlotPSD
from fmSupportLib import fmDemodArctan, fmPlotPSD
# for take-home add your functions
# testing mode 0
rf_Fs = 2.4e6
rf_Fc = 100e3
rf_taps = 101
rf_decim = 10

# add other settings for audio, like filter taps, ...
audio_Fs = 48e3
audio_decim = 5
audio_up = 1
audio_taps = 101
audio_Fc = 16e3

# flag that keeps track if your code is running for
# in-lab (il_vs_th = 0) vs takehome (il_vs_th = 1)
il_vs_th = 1

def low_pass_filter(fc, fs, N_taps):
	
	norm_cutoff = fc / (fs / 2)
	middle_index = (N_taps - 1) / 2
	h = np.zeros(N_taps)

	for i in range(0, N_taps - 1):
		if (i == ((N_taps - 1) / 2)):
			h[i] = norm_cutoff
		else:
			h[i] = (norm_cutoff * 
				np.sin(np.pi * norm_cutoff * (i - middle_index)) /
				(np.pi * norm_cutoff * (i - middle_index)))
		
		h[i] *= 1/2 - 1/2 * np.cos(2 * np.pi * i / (N_taps - 1))

	return h

def resample(impulse_response, input_block, state, decimation=1, upsampling=1):
    output_block = np.zeros(int(len(input_block) * upsampling / decimation))
    filter_length = len(impulse_response) // upsampling

    for output_index in range(len(output_block)):
        for filter_index in range(filter_length):
            impulse_index = filter_index * upsampling + (output_index * decimation) % upsampling
            input_index = (output_index * decimation - impulse_index) // upsampling

            if input_index >= 0:
                output_block[output_index] += impulse_response[impulse_index] * input_block[input_index]
            else:
                output_block[output_index] += impulse_response[impulse_index] * state[input_index]
    
    state = input_block[-(len(impulse_response) - 1):]
    return output_block, state

def fmDemodArctanFast(in_phase, quadrature, prev_phase=0.0, epsilon=1e-10):
   
    # Initialize the output array for the demodulated signal
    fm_demod = np.empty(len(in_phase))

    # Iterate through the signal samples
    for k in range(len(in_phase)):
        # Calculate the differences between consecutive samples (for dI and dQ)
        diff_in_phase = in_phase[k] - in_phase[k - 1] if k > 0 else 0
        diff_quadrature = quadrature[k] - quadrature[k - 1] if k > 0 else 0
        
        # Compute the numerator and denominator for the phase change calculation
        numerator = (in_phase[k] * diff_quadrature) - (quadrature[k] * diff_in_phase)
        denominator = (in_phase[k]**2) + (quadrature[k]**2) + epsilon  # Adding epsilon to avoid division by zero
        
        # Calculate the phase change (dTheta)
        phase_change = numerator / denominator
        
        # Update the current phase using the previous phase and the phase change
        current_phase = prev_phase + phase_change
        
        # Unwrap the phase to handle phase discontinuities
        prev_phase, current_phase = np.unwrap([prev_phase, current_phase])

        # Store the phase change (FM demodulation) in the output array
        fm_demod[k] = phase_change

        # Update the previous phase for the next iteration
        prev_phase = current_phase

    return fm_demod, prev_phase

if __name__ == "__main__":

    # read the raw IQ data from the recorded file
    # IQ data is assumed to be in 8-bits unsigned (and interleaved)
    in_fname = "../data/samples3.raw"
    raw_data = np.fromfile(in_fname, dtype='uint8')
    print("Read raw RF data from \"" + in_fname + "\" in unsigned 8-bit format")
    # IQ data is normalized between -1 and +1 in 32-bit float format
    iq_data = (np.float32(raw_data) - 128.0)/128.0
    print("Reformatted raw RF data to 32-bit float format (" + str(iq_data.size * iq_data.itemsize) + " bytes)")

    # coefficients for the front-end low-pass filter
    rf_coeff = signal.firwin(rf_taps, rf_Fc/(rf_Fs/2), window=('hann'))

    # coefficients for the filter to extract mono audio
    if il_vs_th == 0:
        # to be updated by you during the in-lab session based on firwin
        # same principle  as for rf_coeff (but different arguments, of course)
        audio_coeff = signal.firwin(audio_taps, audio_Fc, window=('hann'), fs=rf_Fs/rf_decim)
    else:
        # to be updated by you for the takehome exercise
        # with your own code for impulse response generation
        #audio_coeff = audio_up*filter_coeff(audio_Fc, rf_Fs/rf_decim*audio_up, audio_taps*audio_up)
        audio_coeff = audio_up*low_pass_filter(audio_Fc, rf_Fs/rf_decim*audio_up, audio_taps*audio_up)

    # set up the subfigures for plotting
    subfig_height = np.array([0.8, 2, 1.6]) # relative heights of the subfigures
    plt.rc('figure', figsize=(7.5, 7.5))    # the size of the entire figure
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, gridspec_kw={'height_ratios': subfig_height})
    fig.subplots_adjust(hspace = .6)

    # select a block_size that is a multiple of KB
    # and a multiple of decimation factors
    block_size = 1024 * 10 * 5 * 2
    block_count = 0

    # states needed for continuity in block processing
    state_i_lpf_100k = np.zeros(rf_taps-1)
    state_q_lpf_100k = np.zeros(rf_taps-1)
    state_audio = np.zeros(audio_up*audio_taps-1)
    state_phase = 0

    # add state as needed for the mono channel filter

    # audio buffer that stores all the audio blocks
    audio_data = np.array([]) # used to concatenate filtered blocks (audio data)

    # if the number of samples in the last block is less than the block size
    # it is fine to ignore the last few samples from the raw IQ file
    while (block_count+1)*block_size < len(iq_data):

        # if you wish to have shorter runtimes while troubleshooting
        # you can control the above loop exit condition as you see fit
        print('Processing block ' + str(block_count))

        # filter to extract the FM channel (I samples are even, Q samples are odd)
        i_filt, state_i_lpf_100k = signal.lfilter(rf_coeff, 1.0, \
                iq_data[(block_count)*block_size:(block_count+1)*block_size:2],
                zi=state_i_lpf_100k)
        q_filt, state_q_lpf_100k = signal.lfilter(rf_coeff, 1.0, \
                iq_data[(block_count)*block_size+1:(block_count+1)*block_size:2],
                zi=state_q_lpf_100k)

        # downsample the I/Q data from the FM channel
        i_ds = i_filt[::rf_decim]
        q_ds = q_filt[::rf_decim]

        # FM demodulator
        if il_vs_th == 0:
            # already given to you for the in-lab
            # take particular notice of the "special" state-saving
            fm_demod, state_phase = fmDemodArctan(i_ds, q_ds, state_phase)
        else:
            # you will need to implement your own FM demodulation based on:
            # https://www.embedded.com/dsp-tricks-frequency-demodulation-algorithms/
            # see more comments on fmSupportLib.py - take particular notice that
            # you MUST have also "custom" state-saving for your own FM demodulator
            fm_demod, state_phase = fmDemodArctanFast(i_ds, q_ds, state_phase)

        # extract the mono audio data through filtering
        if il_vs_th == 0:
            # to be updated by you during the in-lab session based on lfilter
            # same principle as for i_filt or q_filt (but different arguments)
            audio_filt, state_audio = signal.lfilter(audio_coeff, 1.0, \
                fm_demod, zi=state_audio)
        else:
            # to be updated by you for the takehome exercise
            # with your own code for BLOCK convolution
            audio_filt, state_audio = resample(audio_coeff, fm_demod, state_audio, audio_decim, audio_up)

        # downsample audio data
        # to be updated by you during in-lab (same code for takehome)
        audio_block = audio_filt

        # concatenate the most recently processed audio_block
        # to the previous blocks stored already in audio_data
        #

        audio_data = np.concatenate((audio_data, audio_block))

		# to save runtime, select the range of blocks to log data
		# this includes both saving binary files and plotting PSD
        if block_count >= 10 and block_count < 12:

            # plot PSD of selected block after FM demodulation
            # (for easier visualization purposes we divide Fs by 1e3 to imply the kHz units on the x-axis)
            # (this scales the y axis of the PSD, but not the relative strength of different frequencies)
            ax0.clear()
            fmPlotPSD(ax0, fm_demod, (rf_Fs / rf_decim) / 1e3, subfig_height[0], \
            	'Demodulated FM (block ' + str(block_count) + ')')
			# output binary file name (where samples are written from Python)
            fm_demod_fname = "../data/fm_demod_" + str(block_count) + ".bin"
            # create binary file where each sample is a 32-bit float
            fm_demod.astype('float32').tofile(fm_demod_fname)

            ax1.clear()
            fmPlotPSD(ax1, audio_filt, (rf_Fs/rf_decim) / 1e3, subfig_height[1], 'Extracted')
            ax2.clear()
            fmPlotPSD(ax2, audio_block, audio_Fs / 1e3, subfig_height[2], 'Downsampled')

			# save figure to file
            fig.savefig("../data/fmMonoBlock" + str(block_count) + ".png")

        block_count += 1

    print('Finished processing all the blocks from the recorded I/Q samples')

    # write audio data to file
    out_fname = "../data/fmMonoBlock.wav"
    wavfile.write(out_fname, int(audio_Fs), np.int16((audio_data/2)*32767))
    print("Written audio samples to \"" + out_fname + "\" in signed 16-bit format")

    # uncomment assuming you wish to show some plots
    plt.show()
