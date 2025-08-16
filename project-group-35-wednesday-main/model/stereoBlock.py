import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
import math

# use fmDemodArctan from support library
from fmSupportLib import fmDemodArctan, estimatePSD, process_delay_block
from fmPll import fmPll

# the radio-frequency (RF) sampling rate
rf_Fs = 2.4e6

# the cutoff frequency to extract the FM channel from raw IQ data
rf_Fc = 100e3

# the number of taps for the low-pass filter
rf_taps = 101

# decimation rate for reducing sampling rate at the intermediate frequency (IF)
rf_decim = 10

# FFT size for PSD estimation
NFFT = 1024  

# intermediate sampling frequency
if_Fs = 2.4e5

if_taps = 101

# Audio frequency
audio_Fs = 2.4e5 / 5



if __name__ == "__main__":

    # read the raw IQ data
    in_fname = "../data/stereo_l0_r9.raw"
    
    raw_data = np.fromfile(in_fname, dtype='uint8')
    print("Read raw RF data from \"" + in_fname + "\" in unsigned 8-bit format")

    # normalize raw IQ data to 64-bit float (-1 to +1)
    iq_data = (np.float64(raw_data) - 128.0) / 128.0
    print("Reformatted raw RF data to 64-bit double format (" + str(iq_data.size * iq_data.itemsize) + " bytes)")


    # Front end ------------------------------------------------------------

    # coefficients for filters
    rf_coeff = signal.firwin(rf_taps, rf_Fc / (rf_Fs / 2), window=('hann'))
    carrier_coeff = signal.firwin(if_taps, [18.5e3 / (if_Fs / 2), 19.5e3 / (if_Fs / 2)], pass_zero=False, window=('hann'))
    channel_coeff = signal.firwin(if_taps, [22e3 / (if_Fs / 2), 54e3 / (if_Fs / 2)], pass_zero=False, window=('hann'))
    mono_coeff = signal.firwin(if_taps, 16e3 / (if_Fs / 2), window=('hann'))

    block_size = 1024 * rf_decim * 5 * 2 
    block_count = 0

    audio_blocks = []  

    # states needed for continuity in block processing
    state_i = np.zeros(rf_taps - 1)
    state_q = np.zeros(rf_taps - 1)
    state_phase = 0

    # States for continuity across blocks
    state_carrier = np.zeros(if_taps - 1)
    state_channel = np.zeros(if_taps - 1)
    state_filtered_mixed = np.zeros(if_taps - 1)
    state_filtered_mono = np.zeros(if_taps - 1)

    state_delay_block = np.zeros(int((if_taps - 1) / 2))

    pll_state = {"integrator": 0.0, "phaseEst": 0.0, "trigOffset": 0, "feedbackI": 1.0, "feedbackQ": 0.0, "ncoOut0": 1.0}



    while (block_count + 1) * block_size < len(iq_data):

      print(f"Processing block {block_count + 1}")

      # filter to extract the FM channel
      i_filt, state_i = signal.lfilter(rf_coeff, 1.0, \
      iq_data[block_count * block_size:(block_count + 1) * block_size:2],
      zi=state_i)
      q_filt, state_q = signal.lfilter(rf_coeff, 1.0, \
      iq_data[block_count * block_size + 1:(block_count + 1) * block_size:2],
      zi=state_q)

      # downsample the FM channel
      i_ds = i_filt[::rf_decim]
      q_ds = q_filt[::rf_decim]

      # FM demodulation
      fm_demod, _ = fmDemodArctan(i_ds, q_ds)

      # Back End ----------------------------------------------------

      # Carrier wave extraction ----------------------------

      carrier_data, state_carrier = signal.lfilter(carrier_coeff, 1.0, fm_demod, zi=state_carrier)

      nco_out, pll_state = fmPll(carrier_data, 19e3, if_Fs, 2.0, state=pll_state)

      # Channel data extraction -----------------------------

      channel_data, state_channel = signal.lfilter(channel_coeff, 1.0, fm_demod, zi=state_channel)

      # Mixer
      np_channel_data = np.array(channel_data)
      np_nco_out = np.array(nco_out)
      mixed_data = np_channel_data * np_nco_out[:len(np_channel_data)] * 2

      # Mono extraction

      filtered_mixed_data, state_filtered_mixed = signal.lfilter(mono_coeff, 1.0, mixed_data, zi=state_filtered_mixed)

      delayed_fm_demod, state_delay_block = process_delay_block(fm_demod, state_delay_block)

      filtered_mono_data, state_filtered_mono = signal.lfilter(mono_coeff, 1.0, delayed_fm_demod, zi=state_filtered_mono)

      downsampled_mixed_data = filtered_mixed_data[::5]
      downsampled_mono_data = filtered_mono_data[::5]

      left_audio_data = (downsampled_mono_data + downsampled_mixed_data) / 2
      right_audio_data = (downsampled_mono_data - downsampled_mixed_data) / 2

      # Combine into stereo: shape = (num_samples, 2)
      stereo_data = np.stack((left_audio_data, right_audio_data), axis=-1)

      audio_blocks.append(stereo_data)

      block_count += 1

    # Scale to 16-bit signed integers
    audio_data = np.vstack(audio_blocks)
    stereo_data_int16 = np.int16(audio_data * 32767)

    # Write to WAV
    out_fname = "fmStereoBlock.wav"
    wavfile.write(out_fname, int(audio_Fs), stereo_data_int16)

    print(f"Written stereo audio to \"{out_fname}\" in signed 16-bit format")











    
