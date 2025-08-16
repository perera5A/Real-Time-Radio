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

    # coefficients for front-end low-pass filter
    rf_coeff = signal.firwin(rf_taps, rf_Fc / (rf_Fs / 2), window=('hann'))

    # filter to extract the FM channel
    i_filt = signal.lfilter(rf_coeff, 1.0, iq_data[0::2])
    q_filt = signal.lfilter(rf_coeff, 1.0, iq_data[1::2])

    # downsample the FM channel
    i_ds = i_filt[::rf_decim]
    q_ds = q_filt[::rf_decim]

    # FM demodulation
    fm_demod, _ = fmDemodArctan(i_ds, q_ds)

    # Back End ----------------------------------------------------

    # Carrier wave extraction ----------------------------
    carrier_coeff = signal.firwin(if_taps, [18.5e3 / (if_Fs / 2), 19.5e3 / (if_Fs / 2)], pass_zero=False, window=('hann'))

    carrier_data = signal.lfilter(carrier_coeff, 1.0, fm_demod)


    nco_out, state = fmPll(carrier_data, 19e3, if_Fs, 2.0)

    # Channel data extraction -----------------------------
    channel_coeff = signal.firwin(if_taps, [22e3 / (if_Fs / 2), 54e3 / (if_Fs / 2)], pass_zero=False, window=('hann'))

    channel_data = signal.lfilter(channel_coeff, 1.0, fm_demod)

    # Mixer
    np_channel_data = np.array(channel_data)
    
    # Will give error when PLL is in block mode 
    np_nco_out = np.array(nco_out)

    mixed_data = np_channel_data * np_nco_out[:len(np_channel_data)] * 2

    # Mono extraction
    mono_coeff = signal.firwin(if_taps, 16e3 / (if_Fs / 2), window=('hann'))

    filtered_mixed_data = signal.lfilter(mono_coeff, 1.0, mixed_data)

    # delay mono data
    delay_samples = int((if_taps - 1) / 2)
    delay_buffer = np.zeros(delay_samples)

    delayed_fm_demod, delay_buffer = process_delay_block(fm_demod, delay_buffer)

    filtered_mono_data = signal.lfilter(mono_coeff, 1.0, delayed_fm_demod)

    downsampled_mixed_data = filtered_mixed_data[::5]
    downsampled_mono_data = filtered_mono_data[::5]

    left_audio_data = (downsampled_mono_data + downsampled_mixed_data) / 2
    right_audio_data = (downsampled_mono_data - downsampled_mixed_data) / 2

        # Combine into stereo: shape = (num_samples, 2)
    stereo_data = np.stack((left_audio_data, right_audio_data), axis=-1)

    # Scale to 16-bit signed integers
    stereo_data_int16 = np.int16(stereo_data * 32767)

    # Write to WAV
    out_fname = "fmStereo.wav"
    wavfile.write(out_fname, int(audio_Fs), stereo_data_int16)

    print(f"Written stereo audio to \"{out_fname}\" in signed 16-bit format")

