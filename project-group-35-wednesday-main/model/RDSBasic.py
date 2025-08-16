import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
import math

# use fmDemodArctan from support library
from fmSupportLib import fmDemodArctan, estimatePSD, process_delay_block, convolveFIR_block_polyphase, plot_constellation, manchesterDecoding, getSamples, adaptiveSampling, differentialDecoding
from fmPll import fmPll, fmPllwithIQ
from fmRRC import impulseResponseRootRaisedCosine
from fmParityMatrix import ParityCheck

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

pll_state = {"integrator": 0.0, "phaseEst": 0.0, "trigOffset": 0, "feedbackI": 1.0, "feedbackQ": 0.0, "ncoOutI": 1.0, "ncoOutQ": 0.0}



if __name__ == "__main__":

    # read the raw IQ data
    in_fname = "C:/McMaster/2025_Winter/3DY4/project-group-35-wednesday/data/samples3.raw"
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

    # Channel extraction ------------------------------------------

    RDS_channel_coeff = signal.firwin(if_taps, [54e3 / (if_Fs / 2), 60e3 / (if_Fs / 2)], pass_zero=False, window=('hann'))

    RDS_channel_data = signal.lfilter(RDS_channel_coeff, 1.0, fm_demod)

    # Carrier recovery --------------------------------------------
    # Nonlinear squaring
    RDS_channel_data = np.array(RDS_channel_data)
    RDS_channel_data_squared = RDS_channel_data * RDS_channel_data

    RDS_squared_coeff = signal.firwin(if_taps, [113.5e3 / (if_Fs / 2), 114.5e3 / (if_Fs / 2)], pass_zero=False, window=('hann'))

    RDS_carrier_data = signal.lfilter(RDS_squared_coeff, 1.0, RDS_channel_data_squared)

    # plt.psd(RDS_carrier_data[1000:1500], NFFT=1024, Fs=if_Fs)
    # plt.show()
    # exit()

    # freq, psd_data = estimatePSD(RDS_carrier_data, 1024, if_Fs)

    # plt.plot(freq, psd_data, label='Q component', color='red')
    # plt.title('RDS carrier')
    # plt.xlabel('Samples')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.show()

    # PLL locking
    nco_outI, nco_outQ, pll_state = fmPllwithIQ(RDS_carrier_data, 114e3, if_Fs, 0.5, 0, 0.0045, state = pll_state)

    plt.plot(nco_outI[10000:10100])
    plt.show()

    # All pass filtering
    state_block = np.zeros(int((if_taps - 1) / 2))
    RDS_channel_delayed, state = process_delay_block(RDS_channel_data, state_block)

    # Mixer -----------------------------------------------------
    nco_outI = np.array(nco_outI)
    nco_outQ = np.array(nco_outQ)
    RDS_channel_delayed = np.array(RDS_channel_delayed)

    mixed_RDS_dataI = nco_outI[:len(RDS_channel_delayed)] * RDS_channel_delayed * 2
    mixed_RDS_dataQ = nco_outQ[:len(RDS_channel_delayed)] * RDS_channel_delayed * 2

    # LPF 
    mixer_coeff = signal.firwin(if_taps, 3e3 / (if_Fs / 2),  window=('hann'))

    filtered_mixed_RDS_dataI = signal.lfilter(mixer_coeff, 1.0, mixed_RDS_dataI)
    filtered_mixed_RDS_dataQ = signal.lfilter(mixer_coeff, 1.0, mixed_RDS_dataQ)

    # Rational resampler
    SPS = 26
    U = 247
    D = 960
    resampler_coeff = signal.firwin(if_taps * U, 61750/2, window=('hann'), fs = if_Fs * U)
    resampler_dataI = signal.resample_poly(filtered_mixed_RDS_dataI, up=U, down=D, window=resampler_coeff)
    resampler_dataQ = signal.resample_poly(filtered_mixed_RDS_dataQ, up=U, down=D, window=resampler_coeff)

    # # Take the first 1000 samples
    # samples_to_plot = 5000

    # plt.figure(figsize=(10, 5))
    # plt.plot(filtered_mixed_RDS_dataI[515000-1000:515000-1000+samples_to_plot], label="I Channel", color='b')
    # plt.plot(filtered_mixed_RDS_dataQ[515000-1000:515000-1000+samples_to_plot], label="Q Channel", color='r', alpha=0.7)

    # plt.xlabel("Sample Index")
    # plt.ylabel("Amplitude")
    # plt.title("Filtered Mixed RDS Data (First 1000 Samples)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # RRC
    RRC_coeff = impulseResponseRootRaisedCosine(61750, if_taps)
    RRC_dataI = signal.lfilter(RRC_coeff, 1.0, resampler_dataI)
    RRC_dataQ = signal.lfilter(RRC_coeff, 1.0, resampler_dataQ)

    RRC_dataI = np.array(RRC_dataI)
    RRC_dataQ = np.array(RRC_dataQ)

    normalized_I_RRC = RRC_dataI / max(np.abs(RRC_dataI))
    normalized_Q_RRC = RRC_dataQ / max(np.abs(RRC_dataI))

    # plt.figure(figsize=(8,5))
    # plt.plot(normalized_I_RRC[2000:2500], label='I component', color='blue')
    # plt.plot(normalized_Q_RRC[2000:2500], label='Q component', color='red')
    # plt.title('I and Q data')
    # plt.xlabel('Samples')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.show()

    # samplesI, indicesI = adaptiveSampling(normalized_I_RRC, SPS)
    # samplesQ, indicesQ = adaptiveSampling(normalized_Q_RRC, SPS)

    samplesI, indicesI, samples = getSamples(normalized_I_RRC, SPS)
    samplesQ, indicesQ, samples = getSamples(normalized_Q_RRC, SPS)

    # plt.scatter(samplesI, samplesQ, s=10)
    # plt.show()
    start = 1
    decoded_bits, prev_sample, start = manchesterDecoding(samplesI, start=start)

    bit_stream, prev_bits = differentialDecoding(decoded_bits)

    message, synchronized, prev_bits = ParityCheck(bit_stream)


    

    # # Scale to 16-bit signed integers
    # RDS_data_int16 = np.int16(RDS_data * 32767)

    # # Write to WAV
    # out_fname = "fmStereo.wav"
    # wavfile.write(out_fname, int(audio_Fs), RDS_data_int16)

    # print(f"Written stereo audio to \"{out_fname}\" in signed 16-bit format")











    
