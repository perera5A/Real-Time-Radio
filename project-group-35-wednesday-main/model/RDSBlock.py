import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
import math
import time

# use fmDemodArctan from support library
from fmSupportLib import fmDemodArctan, blockwise_resample_poly, estimatePSD, process_delay_block, convolveFIR_block_polyphase, plot_constellation, manchesterDecoding,differentialDecoding,getSamples, blockwise_resample
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





if __name__ == "__main__":

    # read the raw IQ data
    in_fname = "C:/McMaster/2025_Winter/3DY4/project-group-35-wednesday/data/samples3.raw"

    raw_data = np.fromfile(in_fname, dtype='uint8')
    print("Read raw RF data from \"" + in_fname + "\" in unsigned 8-bit format")

    # normalize raw IQ data to 64-bit float (-1 to +1)
    iq_data = (np.float64(raw_data) - 128.0) / 128.0
    print("Reformatted raw RF data to 64-bit double format (" + str(iq_data.size * iq_data.itemsize) + " bytes)")

    block_size = 1024 * rf_decim * 5 * 2 
    block_count = 0

    audio_blocks = []
    # lists for debugging and plotting
    I_const =[]
    Q_const = []

    I_mixed = []
    Q_mixed = []

    I_resample = []
    Q_resample = []

    I_norm =[]
    Q_norm = []

    SPS = 26
    U = 247
    D = 960

    # states needed for block processing
    state_i = np.zeros(rf_taps - 1)
    state_q = np.zeros(rf_taps - 1)
    state_phase = 0

    state_carrier = np.zeros(if_taps - 1)
    state_channel = np.zeros(if_taps - 1)
    state_filtered_mixedI = np.zeros(if_taps - 1)
    state_filtered_mixedQ = np.zeros(if_taps - 1)
    state_filtered_mono = np.zeros(if_taps - 1)
    state_delay_block = np.zeros(int((if_taps - 1) / 2))
    state_resamplerI = np.zeros(if_taps - 1)
    state_resamplerQ = np.zeros(if_taps- 1)
    state_RRC_I = np.zeros(if_taps- 1)
    state_RRC_Q = np.zeros(if_taps- 1)
    state_samplesI = []
    state_samplesQ = []
    synchronized = False
    prev_sample_manchester = None
    previous_bit = 0
    start = 1
    previous_bits = None

    # filter coefficients
    rf_coeff = signal.firwin(rf_taps, rf_Fc / (rf_Fs / 2), window=('hann')) # for front end LPF
    RDS_channel_coeff = signal.firwin(if_taps, [54e3 / (if_Fs / 2), 60e3 / (if_Fs / 2)], pass_zero=False, window=('hann'))
    RDS_squared_coeff = signal.firwin(if_taps, [113.5e3 / (if_Fs / 2), 114.5e3 / (if_Fs / 2)], pass_zero=False, window=('hann'))
    mixer_coeff = signal.firwin(if_taps, 3e3 / (if_Fs / 2),  window=('hann'))
    resampler_coeff = signal.firwin(if_taps * U, 61750 / 2, fs = 240000 * U, window =('hann'))*U
    RRC_coeff = impulseResponseRootRaisedCosine(61750, if_taps)

    pll_state = {"integrator": 0.0, "phaseEst": 0.0, "trigOffset": 0, "feedbackI": 1.0, "feedbackQ": 0.0, "ncoOutI": 1.0, "ncoOutQ": 0.0}

    while (block_count + 1) * block_size < len(iq_data) and (block_count + 1) < 50:

        # Front end ------------------------------------------------------------
        print("Processing Block " +  str(block_count))

        # get the current time in seconds since the epoch
        # seconds = time.time_ns()

        # print("Seconds since", seconds)	

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

        # seconds = time.time_ns()

        # print("Seconds since", seconds)	

        # FM demodulation
        fm_demod, _ = fmDemodArctan(i_ds, q_ds)

        # Back End ----------------------------------------------------

        # Channel extraction ------------------------------------------

        # BPF
        RDS_channel_data, state_channel = signal.lfilter(RDS_channel_coeff, 1.0, fm_demod, zi=state_channel)

        # Carrier recovery --------------------------------------------
        # Nonlinear squaring
        RDS_channel_data = np.array(RDS_channel_data)
        RDS_channel_data_squared = RDS_channel_data * RDS_channel_data

        RDS_carrier_data, state_carrier = signal.lfilter(RDS_squared_coeff, 1.0, RDS_channel_data_squared, zi=state_carrier)

        # PLL locking
        nco_outI, nco_outQ, pll_state = fmPllwithIQ(RDS_carrier_data, 114e3, if_Fs, 0.5, 0, 0.0045, state = pll_state)

        # All pass filtering
        RDS_channel_delayed, state_delay_block = process_delay_block(RDS_channel_data, state_delay_block)

        # Mixer -----------------------------------------------------
        nco_outI = np.array(nco_outI)
        nco_outQ = np.array(nco_outQ)
        RDS_channel_delayed = np.array(RDS_channel_delayed)

        mixed_RDS_dataI = nco_outI[:len(RDS_channel_delayed)] * RDS_channel_delayed * 2
        mixed_RDS_dataQ = nco_outQ[:len(RDS_channel_delayed)] * RDS_channel_delayed * 2

        # LPF 
        filtered_mixed_RDS_dataI, state_filtered_mixedI = signal.lfilter(mixer_coeff, 1.0, mixed_RDS_dataI, zi=state_filtered_mixedI)
        filtered_mixed_RDS_dataQ, state_filtered_mixedQ = signal.lfilter(mixer_coeff, 1.0, mixed_RDS_dataQ, zi=state_filtered_mixedQ)
        #assert len(filtered_mixed_RDS_dataI) == len(filtered_mixed_RDS_dataQ), "I/Q mismatch detected!"

        I_mixed.extend(filtered_mixed_RDS_dataI)
        Q_mixed.extend(filtered_mixed_RDS_dataQ)

        # Rational resampler
        resampler_dataI, state_resamplerI = blockwise_resample_poly(filtered_mixed_RDS_dataI, state_resamplerI, U, D, resampler_coeff)
        resampler_dataQ, state_resamplerQ = blockwise_resample_poly(filtered_mixed_RDS_dataQ, state_resamplerQ, U, D, resampler_coeff)

        # print(f"Block {block_count} — Resampler I max: {np.max(np.abs(resampler_dataI)):.4f}")
        # print(f"Block {block_count} — Resampler Q max: {np.max(np.abs(resampler_dataQ)):.4f}")
        # print(f"Resampled lengths: I={len(resampler_dataI)}, Q={len(resampler_dataQ)}")

        I_resample.extend(resampler_dataI)
        
        Q_resample.extend(resampler_dataQ)
        # RRC
        RRC_dataI, state_RRC_I = signal.lfilter(RRC_coeff, 1.0, resampler_dataI, zi=state_RRC_I)
        RRC_dataQ, state_RRC_Q = signal.lfilter(RRC_coeff, 1.0, resampler_dataQ, zi=state_RRC_Q)

        RRC_dataI = np.array(RRC_dataI)
        RRC_dataQ = np.array(RRC_dataQ)

        # Manchester decoding
        normalized_I_RRC = RRC_dataI / max(np.abs(RRC_dataI))
        normalized_Q_RRC = RRC_dataQ / max(np.abs(RRC_dataI))
            

        I_norm.extend(normalized_I_RRC)
        Q_norm.extend(normalized_Q_RRC)

        # I_norm.extend(RRC_dataI)
        # Q_norm.extend(RRC_dataQ)

        sams = np.array(state_samplesI)

        samplesI, indicesI, state_samplesI = getSamples(normalized_I_RRC, SPS, state_samplesI)
        samplesQ, indicesQ, state_samplesQ = getSamples(normalized_Q_RRC, SPS, state_samplesQ)

        norms = np.concatenate((sams, normalized_I_RRC))

        # # Plot the I samples
        # if block_count == 27:
        #     plt.figure(figsize=(10, 5))
        #     plt.plot(norms[:1000], label="Normalized I (RRC)", alpha=0.6)  # Plot full I signal
        #     plt.scatter(indicesI, samplesI, color='red', label="Sampled I", zorder=3)  # Mark sampled points
        #     plt.xlabel("Sample Index")
        #     plt.ylabel("Amplitude")
        #     plt.title("I Channel Samples from Normalized RRC")
        #     plt.legend()
        #     plt.grid(True)
        #     plt.show()
        #     break

        #samplesI, samplesQ, indices, state_samplesI, state_samplesQ = getSamples(normalized_I_RRC, normalized_Q_RRC, SPS, state_samplesI, state_samplesQ)



        I_const.extend(samplesI)
        Q_const.extend(samplesQ)

        if block_count < 10:
            block_count += 1
            continue

        decoded_bits, prev_sample, start = manchesterDecoding(samplesI, start=start)
        bit_stream, previous_bit = differentialDecoding(decoded_bits, previous_bit)
        
        message, synchronized, previous_bits = ParityCheck(bit_stream, synchronized, previous_bits)


        # if block_count == 10:
            # plt.figure(figsize=(8,5))
            # plt.plot(I_mixed[0:1000], label='I component', color='blue')
            # plt.plot(Q_mixed[0:1000], label='Q component', color='red')
            # plt.title('filtered mixed normal RDS')
            # plt.xlabel('Samples')
            # plt.ylabel('Amplitude')
            # plt.legend()
            # plt.show()

            # plt.figure(figsize=(8,5))
            # plt.plot(I_resample[0:1000], label='I component', color='blue')
            # plt.plot(Q_resample[0:1000], label='Q component', color='red')
            # plt.title('resampler data block')
            # plt.xlabel('Samples')
            # plt.ylabel('Amplitude')
            # plt.legend()
            # plt.show()

            # plt.figure(figsize=(8,5))
            # plt.plot(I_norm[0:1000], label='I component', color='blue')
            # plt.plot(Q_norm[0:1000], label='Q component', color='red')
            # plt.title('I and Q data')
            # plt.xlabel('Samples')
            # plt.ylabel('Amplitude')
            # plt.legend()
            # plt.show()
            # print(f"the i data is {len(samplesI)}")
            # print(f"the q data is {len(samplesQ)}")
            # plt.scatter(samplesI[0:1000], samplesQ[0:1000], s=10)
            # plt.show()

            


            

        # decoded_bits, prev_sample_manchester, start = manchesterDecoding(samplesI, prev_sample_manchester, start )

        # bit_stream, previous_bit = differentialDecoding(decoded_bits, previous_bit)

        block_count += 1
    
    # print(f"the i data is {len(I_const)}")
    # print(f"the q data is {len(Q_const)}")
    # plt.scatter(I_const, Q_const, s=10)
    # plt.show()
    # plt.scatter(I_const, Q_const[:-1], s=10)
    # plt.show()
    # # Scale to 16-bit signed integers
    # RDS_data_int16 = np.int16(RDS_data * 32767)

    # # Write to WAV
    # out_fname = "fmStereo.wav"
    # wavfile.write(out_fname, int(audio_Fs), RDS_data_int16)

    # print(f"Written stereo audio to \"{out_fname}\" in signed 16-bit format")
