#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <memory>
#include "dy4.h"
#include "filter.h"
#include "fourier.h"
#include "genfunc.h"
#include "iofunc.h"
#include "logfunc.h"
#include "demod.h"
#include "readStdinBlockData.h"
#include "PLL.h"
#include "RDSsupport.h"
#define NUM_TAPS 301

// Mono Commands
// cat ../data/iq_samples.raw | ./project 0 M| aplay -c 1 -f S16_LE -r 48000
// cat ../model/_960.raw | ./project 1 M| aplay -c 1 -f S16_LE -r 40000
// cat ../data/iq_samples.raw | ./project 2 M| aplay -c 1 -f S16_LE -r 44100
// cat ../model/_1152.raw | ./project 3 M| aplay -c 1 -f S16_LE -r 44100

// Stereo Commands 
// cat ../data/stereo_l0_r9.raw | ./project 0 S| aplay -c 2 -f S16_LE -r 48000
// cat ../model/_960.raw | ./project 1 S| aplay -c 2 -f S16_LE -r 40000
// cat ../data/stereo_l0_r9.raw | ./project 2 S| aplay -c 2 -f S16_LE -r 44100
// cat ../model/_1152.raw | ./project 3 S| aplay -c 2 -f S16_LE -r 44100

// Thread-safe queue for passing fm_demod data
class ThreadSafeQueue {
    private:
        std::queue<std::shared_ptr<std::vector<real>>> queue_;
        std::mutex mtx_;
        std::condition_variable cv_;
        bool finished_ = false;


    public:
        void push(std::shared_ptr<std::vector<real>> data) {  // Pass by value, not moved
            std::lock_guard<std::mutex> lock(mtx_);
            queue_.push(data);  // Copy the shared_ptr, incrementing ref count
            cv_.notify_one();
        }

        bool pop(std::shared_ptr<std::vector<real>>& data) {
            std::unique_lock<std::mutex> lock(mtx_);
            while (queue_.empty() && !finished_) {
                cv_.wait(lock);
            }
            if (queue_.empty() && finished_) {
                return false;
            }
            data = std::move(queue_.front());  // Move out of queue to consumer
            queue_.pop();
            return true;
        }

        void set_finished() {
            std::lock_guard<std::mutex> lock(mtx_);
            finished_ = true;
            cv_.notify_all();
        }
};


// Producer thread: Data acquisition, I/Q splitting, and FM demodulation
void RF_FrontEnd_Thread(ThreadSafeQueue &audio_queue,ThreadSafeQueue &rds_queue, int BLOCK_SIZE, int DecmIQ, std::vector<real> &h_IQ, std::vector<real> &prev_data_i, std::vector<real> &prev_data_q, real &prev_i, real &prev_q)
{
    // Add ThreadSafeQueue &rds_queue
    std::vector<real> block_data(BLOCK_SIZE);

    std::vector<real> i_data(BLOCK_SIZE / 2);

    std::vector<real> q_data(BLOCK_SIZE / 2);

    std::vector<real> i_downsampled(BLOCK_SIZE / (2 * DecmIQ));

    std::vector<real> q_downsampled(BLOCK_SIZE / (2 * DecmIQ));

    std::vector<real> fm_demod(BLOCK_SIZE / DecmIQ);

    std::chrono::duration<double, std::milli> accumulatedTime_I(0);
    std::chrono::duration<double, std::milli> accumulatedTime_Q(0);

    std::chrono::duration<double, std::milli> accumulatedTime_demod(0);

    for (unsigned int block_id = 0;; block_id++)
    {
        readStdinBlockData(BLOCK_SIZE, block_id, block_data);
        if (std::cin.rdstate() != 0)
        {
            audio_queue.set_finished();
            rds_queue.set_finished();
            break; // End of input
        }

        // Split data into I and Q
        for (size_t i = 0, j = 0; i < BLOCK_SIZE; i += 2, j++)
        {
            i_data[j] = block_data[i];
            q_data[j] = block_data[i + 1];
        }

        // Apply FIR filter with state saving and downsampling
        
        auto I_start_time = std::chrono::high_resolution_clock::now();

        convolveFIR_block_with_downsampling(i_downsampled, i_data, h_IQ, DecmIQ, prev_data_i);
        
        auto I_stop_time = std::chrono::high_resolution_clock::now();

        auto Q_start_time = std::chrono::high_resolution_clock::now();
            
        convolveFIR_block_with_downsampling(q_downsampled, q_data, h_IQ, DecmIQ, prev_data_q);
        auto Q_stop_time = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> I_add_run_time = I_stop_time - I_start_time;
	    std::chrono::duration<double, std::milli> Q_add_run_time = Q_stop_time - Q_start_time;
        
        accumulatedTime_I += I_add_run_time;
        accumulatedTime_Q += Q_add_run_time;
        
        // FM Demodulation
        auto fm_demod_ptr = std::make_shared<std::vector<real>>(BLOCK_SIZE / DecmIQ);

        auto demod_start_time = std::chrono::high_resolution_clock::now();
        ownfmDemod(i_downsampled, q_downsampled, prev_i, prev_q, *fm_demod_ptr);
        auto demod_stop_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> demod_run_time = demod_stop_time - demod_start_time;
        accumulatedTime_demod += demod_run_time;

        audio_queue.push(fm_demod_ptr); //push audio queue
        //std::cerr << "Pushed to audio_queue, ptr valid: " << (fm_demod_ptr != nullptr) << std::endl;
        rds_queue.push(fm_demod_ptr); //push rds queue
        //std::cerr << "Pushed to rds_queue, ptr valid: " << (fm_demod_ptr != nullptr) << std::endl;

    }
    
    std::cerr << "I filter ran for " << accumulatedTime_I.count() << " milliseconds" << "\n";
    std::cerr << "Q filter ran for " << accumulatedTime_Q.count() << " milliseconds" << "\n";
    std::cerr << "Demod ran for " << accumulatedTime_demod.count() << " milliseconds" << "\n";    
}

// Consumer thread: Mono path processing
void Audio_Thread(ThreadSafeQueue &audio_queue, int BLOCK_SIZE,
                       int mode, char outputType, int DecmIQ, 
                       std::vector<real> &h_Demod,
                       std::vector<real> &h_19kHz_BPF,
                       std::vector<real> &h_38kHz_BPF,
                       int DownsampleFactor,
                       int UpsampleFactor,
                       std::vector<real> &prev_data_mono,
                       std::vector<real> &prev_data_pilot,
                       std::vector<real> &prev_data_stereo,
                       std::vector<real> &delayed_buffer,
                       std::vector<real> &prev_data_mixer,
                       PLL_state& pll_state, int Fs)
{

    std::shared_ptr<std::vector<real>> fm_demod_ptr;
    std::vector<real> pilot_data;
    std::vector<real> nco_data;
    std::vector<real> stereo_data;
    std::vector<real> mixer_data;

    std::chrono::duration<double, std::milli> accumulatedTimeMonoFilter(0);

    std::chrono::duration<double, std::milli> accumulatedTimeStereoOutputFilter(0);
    std::chrono::duration<double, std::milli> accumulatedTimeChannelFilter(0);
    std::chrono::duration<double, std::milli> accumulatedTimeCarrierFilter(0);
    std::chrono::duration<double, std::milli> accumulatedTimePLL(0);
    std::chrono::duration<double, std::milli> accumulatedTimeMixer(0);
    std::chrono::duration<double, std::milli> accumulatedTimeAllPass(0);
    while (true)
    {

        if (!audio_queue.pop(fm_demod_ptr))
            break;

        std::vector<real> &fm_demod = *fm_demod_ptr;

        // std::cerr << "Audio thread: fm_demod.size() = " << fm_demod.size() << std::endl;

        std::vector<real> mono_data(ceil(fm_demod.size() * (UpsampleFactor / (real)DownsampleFactor)));
        std::vector<real> downsampled_stereo (BLOCK_SIZE /(DecmIQ*DownsampleFactor));
        std::vector<real> demod_delay(BLOCK_SIZE/DecmIQ);
        std::vector<real> audio_block(2*downsampled_stereo.size());
        std::vector<real> prev_data_demod(((101 * UpsampleFactor)-1), 0.0);

        if (mode == 2 || mode == 3)
        {

            if (outputType == 'S') {

                // delay demod data 
                auto all_pass_start_time = std::chrono::high_resolution_clock::now();
                allPass(fm_demod, delayed_buffer, demod_delay); 
                auto all_pass_stop_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> all_pass_time = all_pass_stop_time - all_pass_start_time;
                accumulatedTimeAllPass += all_pass_time;

                // Extract mono audio
                convolveFIR_block_polyphase(mono_data, demod_delay, h_Demod, DownsampleFactor, UpsampleFactor, prev_data_mono);
               
                /*  
                SLOW VERSION 

                std::vector<real> upsampled_fm_demod;

                upsampler(UpsampleFactor, fm_demod, upsampled_fm_demod);
                prev_data_demod.resize(NUM_TAPS*UpsampleFactor);
                convolveFIR_block_with_downsampling(downsampled_fm_demod, upsampled_fm_demod, h_Demod, DownsampleFactor, prev_data_demod);
                */

                // bandpass for 19kHz
                auto carrier_start_time = std::chrono::high_resolution_clock::now();
                convolveFIR_block_with_downsampling(pilot_data, fm_demod, h_19kHz_BPF, 1, prev_data_pilot);
                auto carrier_stop_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> carrier_time = carrier_stop_time - carrier_start_time;
                accumulatedTimeCarrierFilter += carrier_time;

                // Apply PLL to lock onto the pilot tone (19 kHz)
                auto pll_start_time = std::chrono::high_resolution_clock::now();
                fmPLL(nco_data, pilot_data, 19e3, Fs, pll_state, 2.0, 0, 0.001);
                auto pll_stop_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> pll_time = pll_stop_time - pll_start_time;
                accumulatedTimePLL += pll_time;

                // stereo channel extract
                auto channel_start_time = std::chrono::high_resolution_clock::now();
                convolveFIR_block_with_downsampling(stereo_data, fm_demod, h_38kHz_BPF, 1, prev_data_stereo);
                auto channel_stop_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> channel_time = channel_stop_time - channel_start_time;
                accumulatedTimeChannelFilter += channel_time;

                // Mix stereo_data with NCO 38 kHz signal
                // Naturally stereo_data.size() is greater than nco_data.size()
                std::vector<real> stereo(stereo_data.size());
                auto stereo_mixer_start_time = std::chrono::high_resolution_clock::now();
                for (int i=0; i<stereo_data.size(); i++) {
                stereo[i] = nco_data[i] * stereo_data[i] * 2; 
                }
                auto stereo_mixer_stop_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> stereo_mixer_time = stereo_mixer_stop_time - stereo_mixer_start_time;
                accumulatedTimeMixer += stereo_mixer_time;

                auto stereo_output_start_time = std::chrono::high_resolution_clock::now();
                convolveFIR_block_polyphase(downsampled_stereo, stereo, h_Demod, DownsampleFactor, UpsampleFactor, prev_data_mixer); 
                auto stereo_output_stop_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> stereo_output_time = stereo_output_stop_time - stereo_output_start_time;
                accumulatedTimeStereoOutputFilter += stereo_output_time;

                size_t N = std::min(mono_data.size(), downsampled_stereo.size());
                audio_block.resize(2*N);

                for (int i=0; i < N; i++) {
                    audio_block[2*i] = mono_data[i] + downsampled_stereo[i]; // L
                    audio_block[2*i+1] = mono_data[i] - downsampled_stereo[i]; // R
                }
            }

            else if (outputType == 'M') {
                auto mono_start_time = std::chrono::high_resolution_clock::now();
                convolveFIR_block_polyphase(mono_data, fm_demod, h_Demod, DownsampleFactor, UpsampleFactor, prev_data_demod);
                auto mono_stop_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> mono_run_time = mono_stop_time - mono_start_time;
                accumulatedTimeMonoFilter += mono_run_time;
            }
            
        }
        
        else if (mode == 0 || mode == 1) {

            if (outputType == 'S') {

                // delay demod data 
                auto all_pass_start_time = std::chrono::high_resolution_clock::now();
                allPass(fm_demod, delayed_buffer, demod_delay); 
                auto all_pass_stop_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> all_pass_time = all_pass_stop_time - all_pass_start_time;
                accumulatedTimeAllPass += all_pass_time;


                // Extract mono audio
                convolveFIR_block_with_downsampling(mono_data, demod_delay, h_Demod, DownsampleFactor, prev_data_mono);

                // bandpass for 19kHz
                auto carrier_start_time = std::chrono::high_resolution_clock::now();
                convolveFIR_block_with_downsampling(pilot_data, fm_demod, h_19kHz_BPF, 1, prev_data_pilot);
                auto carrier_stop_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> carrier_time = carrier_stop_time - carrier_start_time;
                accumulatedTimeCarrierFilter += carrier_time;

                // Apply PLL to lock onto the pilot tone (19 kHz)
                auto pll_start_time = std::chrono::high_resolution_clock::now();
                fmPLL(nco_data, pilot_data, 19e3, Fs, pll_state, 2.0, 0, 0.001);
                auto pll_stop_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> pll_time = pll_stop_time - pll_start_time;
                accumulatedTimePLL += pll_time;

                // stereo channel extract
                auto channel_start_time = std::chrono::high_resolution_clock::now();
                convolveFIR_block_with_downsampling(stereo_data, fm_demod, h_38kHz_BPF, 1, prev_data_stereo);
                auto channel_stop_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> channel_time = channel_stop_time - channel_start_time;
                accumulatedTimeChannelFilter += channel_time;

                // Mix stereo_data with NCO 38 kHz signal
                std::vector<real> stereo(stereo_data.size());
                auto stereo_mixer_start_time = std::chrono::high_resolution_clock::now();
                for (int i=0; i<stereo_data.size(); i++) {
                stereo[i] = nco_data[i] * stereo_data[i] * 2; 
                }
                auto stereo_mixer_stop_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> stereo_mixer_time = stereo_mixer_stop_time - stereo_mixer_start_time;
                accumulatedTimeMixer += stereo_mixer_time;

                auto stereo_output_start_time = std::chrono::high_resolution_clock::now();
                convolveFIR_block_with_downsampling(downsampled_stereo, stereo, h_Demod, DownsampleFactor, prev_data_mixer);
                auto stereo_output_stop_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> stereo_output_time = stereo_output_stop_time - stereo_output_start_time;
                accumulatedTimeStereoOutputFilter += stereo_output_time;

                for (int i=0; i<downsampled_stereo.size(); i++) {
                    audio_block[2*i] = mono_data[i] + downsampled_stereo[i];
                    audio_block[2*i+1] = mono_data[i] - downsampled_stereo[i];
                }
            }

            else if (outputType == 'M') {
                auto mono_start_time = std::chrono::high_resolution_clock::now();
                convolveFIR_block_with_downsampling(mono_data, fm_demod, h_Demod, DownsampleFactor, prev_data_mono);
                auto mono_stop_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> mono_run_time = mono_stop_time - mono_start_time;
                accumulatedTimeMonoFilter += mono_run_time;
            }
            

        }

        //////////////////////////////////////
        // Convert to audio format (short int)

        if (outputType == 'S') {
            std::vector<short int> audio_data(2*downsampled_stereo.size());
            for (size_t k = 0; k < audio_data.size(); k++)
                {

                    if (std::isnan(audio_block[k]))
                    {

                        audio_data[k] = 0; // Handle NaN values
                    }

                    else
                    {
                        audio_data[k] = static_cast<short int>(audio_block[k] * 16384);
                    }
                }

            // Write output efficiently
            fwrite(audio_data.data(), sizeof(short int), audio_data.size(), stdout);  
        }

        else if (outputType == 'M') {
            // Convert to audio format (short int)
            std::vector<short int> audio_data(mono_data.size());
            for (size_t k = 0; k < mono_data.size(); k++) {
                if (std::isnan(mono_data[k])) {
                    audio_data[k] = 0;  // Handle NaN values
                } else {
                    audio_data[k] = static_cast<short int>(mono_data[k] * 16384);
                }
            }

            // Write output efficiently
            fwrite(audio_data.data(), sizeof(short int), audio_data.size(), stdout);
        }
    }
    if (outputType == 'M') {
        std::cerr << "Mono filter ran for " << accumulatedTimeMonoFilter.count() << " milliseconds" << "\n";
    }

    if (outputType == 'S') {
        std::cerr << "Stereo output filter ran for " << accumulatedTimeStereoOutputFilter.count() << " milliseconds" << "\n";
        std::cerr << "Stereo mixer ran for " << accumulatedTimeMixer.count() << " milliseconds" << "\n";
        std::cerr << "All pass ran for " << accumulatedTimeAllPass.count() << " milliseconds" << "\n";
        std::cerr << "Carrier filter ran for " << accumulatedTimeCarrierFilter.count() << " milliseconds" << "\n";
        std::cerr << "PLL ran for " << accumulatedTimePLL.count() << " milliseconds" << "\n";
        std::cerr << "Channel filter ran for " << accumulatedTimeChannelFilter.count() << " milliseconds" << "\n";
        
    }
}
// Consumer: RDS processing for modes 0 and 2
void RDS_Thread(ThreadSafeQueue &rds_queue, int BLOCK_SIZE,
                       int mode, char outputType, int DecmIQ, 
                       std::vector<real> &h_Demod,
                       std::vector<real> &h_RDS_channel,
                       std::vector<real> &h_RDS_carrier,
                       std::vector<real> &h_RDS_downconversion,
                       std::vector<real> &h_RDS_resampler,
                       std::vector<real> &h_RRC,
                       int RDS_DownsampleFactor,
                       int RDS_UpsampleFactor,
                       int SPS,
                       std::vector<real> &prev_data_RDS_channel,
                       std::vector<real> &prev_data_RDS_carrier,
                       std::vector<real> &prev_data_RDS_downconverted,
                       std::vector<real> &prev_data_resampler,
                       std::vector<real> &delayed_buffer_RDS,
                       std::vector<real> &prev_data_mixer,
                       std::vector<real> &prev_data_RRC,
                       PLL_stateIQ& pllState_RDS, int Fs)
{

    std::shared_ptr<std::vector<real>> fm_demod_ptr;
    std::vector<real> RDS_channel_data(BLOCK_SIZE/DecmIQ);
    std::vector<real> RDS_carrier_data(RDS_channel_data.size());
    std::vector<real> RDS_channel_data_squared(RDS_channel_data.size());
    std::vector<real> RDS_mixed_and_filtered(RDS_channel_data.size());
    std::vector<real> RDS_resampler_data(RDS_channel_data.size() * (RDS_UpsampleFactor/RDS_DownsampleFactor));
    std::vector<real> RRC_data(RDS_resampler_data.size());

    std::vector<real> nco_data(RDS_channel_data.size());

    std::vector<real> RDS_delayed(BLOCK_SIZE/DecmIQ);
    
    std::vector<real> samples;
    std::vector<int> indices;

    std::vector<int> syndromes;

    std::vector<real> nco_dataI(RDS_channel_data.size());
    std::vector<real> nco_dataQ(RDS_channel_data.size());

    std::vector<real> I_mixed_and_filtered(RDS_channel_data.size());
    std::vector<real> Q_mixed_and_filtered(RDS_channel_data.size());
    
    std::vector<real> I_resampler_data(RDS_channel_data.size() * (RDS_UpsampleFactor/RDS_DownsampleFactor));
    std::vector<real> Q_resampler_data(RDS_channel_data.size() * (RDS_UpsampleFactor/RDS_DownsampleFactor));

    std::vector<real> I_RRC_data(RDS_resampler_data.size());
    std::vector<real> Q_RRC_data(RDS_resampler_data.size());

    std::vector<real> normalized_RRC_data(RDS_resampler_data.size());
    std::vector<real> normalized_I_RRC_data(RDS_resampler_data.size());
    std::vector<real> normalized_Q_RRC_data(RDS_resampler_data.size());

    int popcount; 
    
    std::vector<real> prev_data_I_downconverted(100);
    std::vector<real> prev_data_Q_downconverted(100);

    std::vector<real> prev_data_I_resampler(((101 * RDS_UpsampleFactor) - 1), 0.0);
    std::vector<real> prev_data_Q_resampler(((101 * RDS_UpsampleFactor) - 1), 0.0);
    
    std::vector<real> prev_data_I_RRC(100);
    std::vector<real> prev_data_Q_RRC(100);

    std::vector<real> mixed_RDS_data(RDS_carrier_data.size(), 0);

    std::vector<real> mixed_I_data(RDS_carrier_data.size(), 0);
    std::vector<real> mixed_Q_data(RDS_carrier_data.size(), 0);

    std::vector<real> samples_RRC;
    std::vector<real> samplesI;
    std::vector<real> samplesQ;

    std::vector<int> samples_indices;
    std::vector<int> samplesI_indices;
    std::vector<int> samplesQ_indices;

    std::vector<real> prev_samples;
    std::vector<real> prev_samplesI;
    std::vector<real> prev_samplesQ;

    std::vector<int> decoded_bits;
    real previous_sample = 0;

    std::vector<int> bit_stream;
    int previous_bit;

    std::vector<int> window(26);
    std::vector<int> prev_parity_bits;
    bool checksync = false;

    while (true) {
        
        if (!rds_queue.pop(fm_demod_ptr))
            break;
        
        popcount += 1; 

        std::vector<real> &fm_demod = *fm_demod_ptr;

       
        if (mode == 0) {
            
            // 54kHz to 60kHz Channel
            convolveFIR_block_with_downsampling(RDS_channel_data, fm_demod, h_RDS_channel, 1, prev_data_RDS_channel);
             
            //------- Carrier Recovery--------
            //Squaring
            
            for (int i = 0; i < RDS_channel_data.size() ; i++) {
                RDS_channel_data_squared[i] = RDS_channel_data[i] * RDS_channel_data[i]; 
            }
             
            //114 kHz Carrier
            convolveFIR_block_with_downsampling(RDS_carrier_data, RDS_channel_data_squared, h_RDS_carrier, 1, prev_data_RDS_carrier);

            // Set the bandwitdh of the filter 
            fmPLLIQ(nco_dataI, nco_dataQ, RDS_carrier_data, 114e3, Fs, pllState_RDS, 0.5, 0, 0.005);
                        
            // Delay the channel data
            allPass(RDS_channel_data, delayed_buffer_RDS, RDS_delayed);

            // ---------- Mixer ------------------

            for (int i = 0; i < RDS_carrier_data.size(); i++) {

                mixed_RDS_data[i] = nco_dataI[i] * RDS_delayed[i] * 2;
                mixed_I_data[i] = nco_dataI[i] * RDS_delayed[i] * 2;
                mixed_Q_data[i] = nco_dataQ[i] * RDS_delayed[i] * 2;

            }

            // 3KHz LPF
            convolveFIR_block_with_downsampling(RDS_mixed_and_filtered, mixed_RDS_data, h_RDS_downconversion, 1, prev_data_RDS_downconverted);

            convolveFIR_block_with_downsampling(I_mixed_and_filtered, mixed_I_data, h_RDS_downconversion, 1, prev_data_I_downconverted);
            convolveFIR_block_with_downsampling(Q_mixed_and_filtered, mixed_Q_data, h_RDS_downconversion, 1, prev_data_Q_downconverted);

            // Rational Resampler
            convolveFIR_block_polyphase(RDS_resampler_data, RDS_mixed_and_filtered, h_RDS_resampler, RDS_DownsampleFactor, RDS_UpsampleFactor, prev_data_resampler);

            convolveFIR_block_polyphase(I_resampler_data, I_mixed_and_filtered, h_RDS_resampler, RDS_DownsampleFactor, RDS_UpsampleFactor, prev_data_I_resampler);
            convolveFIR_block_polyphase(Q_resampler_data, Q_mixed_and_filtered, h_RDS_resampler, RDS_DownsampleFactor, RDS_UpsampleFactor, prev_data_Q_resampler);

            // RRC Filter
            convolveFIR_block_with_downsampling(RRC_data, RDS_resampler_data, h_RRC, 1, prev_data_RRC);

            convolveFIR_block_with_downsampling(I_RRC_data, I_resampler_data, h_RRC, 1, prev_data_I_RRC);
            convolveFIR_block_with_downsampling(Q_RRC_data, Q_resampler_data, h_RRC, 1, prev_data_Q_RRC);
            
            // Normalize values
            normalize(RRC_data, RRC_data, normalized_RRC_data);

            normalize(I_RRC_data, I_RRC_data, normalized_I_RRC_data);
            normalize(Q_RRC_data, I_RRC_data, normalized_Q_RRC_data);

            // Get samples 
            getSamples(normalized_RRC_data, SPS, samples_RRC, samples_indices, prev_samples);
            getSamples(normalized_I_RRC_data, SPS, samplesI, samplesI_indices, prev_samplesI);
            getSamples(normalized_Q_RRC_data, SPS, samplesQ, samplesQ_indices, prev_samplesQ);

            //Manchester outputs 0's and 1's
            manchesterDecoding(normalized_RRC_data, decoded_bits, previous_sample);

            /*
            std::cerr << "decoded_bits:";
            for(int i = 0; i < decoded_bits.size(); i++){
                std::cerr << decoded_bits[i];
            }
            */

            //Decoding outputs 0's and 1's
            differentialDecoding(decoded_bits, bit_stream, previous_bit);

            /*
            std::cerr << "bit stream:";
            for(int i = 0; i < bit_stream.size(); i++){
                std::cerr << bit_stream[i];
                std::cerr << std::endl;
            }
            */

            //Parity check does not work
            //parity_check(bit_stream, window, checksync, prev_parity_bits);

            //Debug and Plot

            /*
            std::vector<real>x_axis;
            std::vector<real>rrc_axis;
            genIndexVector(x_axis, RDS_resampler_data.size());
            genIndexVector(rrc_axis, samples_indices.size());

            logVector("pre_rrc_data", x_axis, RDS_resampler_data);
            logVector("RRC_data", x_axis, normalized_RRC_data);   

            logVector("I_sample", x_axis, normalized_I_RRC_data);
            logVector("Q_sample", x_axis, normalized_Q_RRC_data); 

            logVector("samples_rrc", samples_RRC, samplesQ); 
            
            logInt("bit_stream", rrc_axis, bit_stream);
            */
            
        }

        else if(mode == 2){
            
            // 54kHz to 60kHz Channel
            convolveFIR_block_with_downsampling(RDS_channel_data, fm_demod, h_RDS_channel, 1, prev_data_RDS_channel);
             
            //------- Carrier Recovery--------
            //Squaring
            
            for (int i = 0; i < RDS_channel_data.size() ; i++) {
                RDS_channel_data_squared[i] = RDS_channel_data[i] * RDS_channel_data[i]; 
            }
             
            //114 kHz Carrier
            convolveFIR_block_with_downsampling(RDS_carrier_data, RDS_channel_data_squared, h_RDS_carrier, 1, prev_data_RDS_carrier);
            // Lock at 114 kHz, scale down to 57 kHz

            // Set the bandwitdh of the filter 
            fmPLLIQ(nco_dataI, nco_dataQ, RDS_carrier_data, 114e3, Fs, pllState_RDS, 0.5, 0, 0.005);
                        
            // Delay the channel data
            allPass(RDS_channel_data, delayed_buffer_RDS, RDS_delayed);

            // ---------- Mixer ------------------

            for (int i = 0; i < RDS_carrier_data.size(); i++) {

                mixed_RDS_data[i] = nco_dataI[i] * RDS_delayed[i] * 2;
                mixed_I_data[i] = nco_dataI[i] * RDS_delayed[i] * 2;
                mixed_Q_data[i] = nco_dataQ[i] * RDS_delayed[i] * 2;

            }

            // 3KHz LPF
            convolveFIR_block_with_downsampling(RDS_mixed_and_filtered, mixed_RDS_data, h_RDS_downconversion, 1, prev_data_RDS_downconverted);

            convolveFIR_block_with_downsampling(I_mixed_and_filtered, mixed_I_data, h_RDS_downconversion, 1, prev_data_I_downconverted);
            convolveFIR_block_with_downsampling(Q_mixed_and_filtered, mixed_Q_data, h_RDS_downconversion, 1, prev_data_Q_downconverted);

            // Rational Resampler
            convolveFIR_block_polyphase(RDS_resampler_data, RDS_mixed_and_filtered, h_RDS_resampler, RDS_DownsampleFactor, RDS_UpsampleFactor, prev_data_resampler);

            convolveFIR_block_polyphase(I_resampler_data, I_mixed_and_filtered, h_RDS_resampler, RDS_DownsampleFactor, RDS_UpsampleFactor, prev_data_I_resampler);
            convolveFIR_block_polyphase(Q_resampler_data, Q_mixed_and_filtered, h_RDS_resampler, RDS_DownsampleFactor, RDS_UpsampleFactor, prev_data_Q_resampler);

            // RRC Filter
            convolveFIR_block_with_downsampling(RRC_data, RDS_resampler_data, h_RRC, 1, prev_data_RRC);

            convolveFIR_block_with_downsampling(I_RRC_data, I_resampler_data, h_RRC, 1, prev_data_I_RRC);
            convolveFIR_block_with_downsampling(Q_RRC_data, Q_resampler_data, h_RRC, 1, prev_data_Q_RRC);
            
            normalize(RRC_data, RRC_data, normalized_RRC_data);

            normalize(I_RRC_data, I_RRC_data, normalized_I_RRC_data);
            normalize(Q_RRC_data, I_RRC_data, normalized_Q_RRC_data);

            getSamples(normalized_RRC_data, SPS, samples_RRC, samples_indices, prev_samples);
            getSamples(normalized_I_RRC_data, SPS, samplesI, samplesI_indices, prev_samplesI);
            getSamples(normalized_Q_RRC_data, SPS, samplesQ, samplesQ_indices, prev_samplesQ);

            manchesterDecoding(normalized_RRC_data, decoded_bits, previous_sample);

            differentialDecoding(decoded_bits, bit_stream, previous_bit);

            //parity_check(bit_stream, window, checksync, prev_parity_bits);
        }
    }
}

int main(int argc, char *argv[])
{

    int mode = 0;
    char outputType = 'M'; 


    // You can input the mode and the output type 
    // For example ./project 0 S for mode 0 stereo 

    if (argc < 2) {
        std::cerr << "Operating in default mode: " << mode << " with output type: " << outputType << std::endl;
    } else if (argc == 2) {
        mode = std::atoi(argv[1]);
        if (mode < 0 || mode > 3) {
            std::cerr << "Invalid mode: " << mode << ". Please enter a number from 0 to 3." << std::endl;
            return 1;
        }
        
    } else if (argc == 3) {
        mode = std::atoi(argv[1]);
        if (mode < 0 || mode > 3) {
            std::cerr << "Invalid mode: " << mode << ". Please enter a number from 0 to 3." << std::endl;
            return 1;
        }
        outputType = toupper(argv[2][0]); // Take first character of second argument, case-insensitive
        if (outputType != 'M' && outputType != 'S') {
            std::cerr << "Invalid output type: " << outputType << ". Please enter 'M' or 'S'." << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Invalid input. Usage: ./project <mode> [M|S]" << std::endl;
        std::cerr << "Example: ./project 2 S" << std::endl;
        return 1;
    }

    std::cerr << "Operating in mode: " << mode << " with output type: " << outputType << std::endl;

    // Filter coefficients and parameters
    std::vector<real> h_IQ, h_Demod;

    int Fs, Fc, gain, BLOCK_SIZE, DecmIQ, DownsampleFactor, UpsampleFactor;

    int SPS, RDS_DownsampleFactor, RDS_UpsampleFactor;

    // Coefficents for bandpass filters
    std::vector<real> h_19kHz_BPF(NUM_TAPS);
    std::vector<real> h_38kHz_BPF(NUM_TAPS);
    
    // Coefficients for RDS
    std::vector<real> h_RDS_channel(NUM_TAPS);
    std::vector<real> h_RDS_carrier(NUM_TAPS);
    std::vector<real> h_RDS_downconversion(NUM_TAPS);
    std::vector<real> h_RDS_resampler(NUM_TAPS);
    std::vector<real> h_RRC(NUM_TAPS);

    if (mode == 0) {

        BLOCK_SIZE = 102400; 
        
        gain = 1;

        Fs = 2.4e6; 
        
        Fc = 100e3; 

        impulseResponseLPF(Fs, Fc, NUM_TAPS, h_IQ, gain);

        Fs = 240e3; 
        
        Fc = 16e3; 
        
        impulseResponseLPF(Fs, Fc, NUM_TAPS, h_Demod, gain);

        DecmIQ = 10; 
        
        DownsampleFactor = 5; 

        UpsampleFactor = 1;


        impulseResponseBPF(Fs, 18.5e3, 19.5e3, NUM_TAPS, h_19kHz_BPF);
        impulseResponseBPF(Fs, 22e3, 54e3, NUM_TAPS, h_38kHz_BPF);

        //RDS constants
        RDS_UpsampleFactor = 247;
        RDS_DownsampleFactor = 960;
        SPS = 26;

        //RDS filtering functionalities
        impulseResponseBPF(Fs, 54e3, 60e3, NUM_TAPS, h_RDS_channel);
        impulseResponseBPF(Fs, 113.5e3, 114.5e3, NUM_TAPS, h_RDS_carrier);
        impulseResponseLPF(Fs, 3e3, NUM_TAPS, h_RDS_downconversion, 1);
        impulseResponseLPF_upscale(Fs, 61750 / 2, NUM_TAPS, h_RDS_resampler, RDS_UpsampleFactor);
        impulseResponseRRC(61750, NUM_TAPS, h_RRC);
        

    } 
    
    else if (mode == 1) {

        BLOCK_SIZE = 49152; 
        
        gain = 1;
        
        Fs = 960e3; 
        
        Fc = 100e3; 
        
        impulseResponseLPF(Fs, Fc, NUM_TAPS, h_IQ, gain);
        
        Fs = 240e3; 
        
        Fc = 16e3; 
        
        impulseResponseLPF(Fs, Fc, NUM_TAPS, h_Demod, gain);
        
        DecmIQ = 4; 
        
        DownsampleFactor = 6; 
        
        UpsampleFactor = 1;

        impulseResponseBPF(Fs, 18.5e3, 19.5e3, NUM_TAPS, h_19kHz_BPF);
        impulseResponseBPF(Fs, 22e3, 54e3, NUM_TAPS, h_38kHz_BPF);
    } 
    
    else if (mode == 2) {

        BLOCK_SIZE = 102400;

        Fs = 2.4e6; Fc = 100e3; 
        
        impulseResponseLPF(Fs, Fc, NUM_TAPS, h_IQ, 1);

        Fs = 240e3; 
        
        Fc = 16e3; 
        
        DownsampleFactor = 800; 
        
        UpsampleFactor = 147;

        impulseResponseLPF_upscale(Fs, Fc, NUM_TAPS, h_Demod, UpsampleFactor);

        DecmIQ = 10;
        
        impulseResponseBPF(Fs, 18.5e3, 19.5e3, NUM_TAPS, h_19kHz_BPF);
        impulseResponseBPF(Fs, 22e3, 54e3, NUM_TAPS, h_38kHz_BPF);

        //RDS constants
        RDS_UpsampleFactor = 247;
        RDS_DownsampleFactor = 640;
        SPS = 39;

        //RDS filtering functionalities
        impulseResponseBPF(Fs, 54e3, 60e3, NUM_TAPS, h_RDS_channel);
        impulseResponseBPF(Fs, 113.5e3, 114.5e3, NUM_TAPS, h_RDS_carrier);
        impulseResponseLPF(Fs, 3e3, NUM_TAPS, h_RDS_downconversion, 1);
        impulseResponseLPF_upscale(Fs, 61750 / 2, NUM_TAPS, h_RDS_resampler, RDS_UpsampleFactor);
        impulseResponseRRC(61750, NUM_TAPS, h_RRC);
        
        
    } 

    else if (mode == 3) {

        BLOCK_SIZE = 46080;

        Fs = 1152e3; 

        Fc = 100e3; 

        impulseResponseLPF(Fs, Fc, NUM_TAPS, h_IQ, 1);

        Fs = 192e3; 

        Fc = 16e3; 

        DownsampleFactor = 640; UpsampleFactor = 147;

        impulseResponseLPF_upscale(Fs, Fc, NUM_TAPS, h_Demod, UpsampleFactor);

        DecmIQ = 6;

        impulseResponseBPF(Fs, 18.5e3, 19.5e3, NUM_TAPS, h_19kHz_BPF);
        impulseResponseBPF(Fs, 22e3, 54e3, NUM_TAPS, h_38kHz_BPF);

    }

    // PLL state
    PLL_state pllState_audio = {}; // Initialize PLL state
    PLL_stateIQ pllState_RDS = {};

    // initialise previous data
    std::vector<real> prev_data_mono(((NUM_TAPS * UpsampleFactor) - 1), 0.0);
    std::vector<real> prev_data_pilot((NUM_TAPS - 1), 0.0);
    std::vector<real> prev_data_stereo((NUM_TAPS - 1), 0.0);
    std::vector<real> prev_data_mixer(((NUM_TAPS * UpsampleFactor) - 1), 0.0);

    std::vector<real> prev_data_RDS_channel((NUM_TAPS - 1), 0.0);
    std::vector<real> prev_data_RDS_carrier((NUM_TAPS - 1), 0.0);
    std::vector<real> prev_data_RDS_downconverted((NUM_TAPS - 1), 0.0);
    std::vector<real> prev_data_resampler(((NUM_TAPS * UpsampleFactor) - 1), 0.0);
    std::vector<real> prev_data_RRC((NUM_TAPS  - 1), 0.0);

    // State variables
    real prev_i = 0, prev_q = 0;

    std::vector<real> prev_data_i(NUM_TAPS - 1, 0.0);

    std::vector<real> prev_data_q(NUM_TAPS - 1, 0.0);

    std::vector<real> delayed_buffer((NUM_TAPS-1)/2, 0.0);

    std::vector<real> delayed_buffer_RDS((NUM_TAPS-1)/2, 0.0);

    // Thread-safe queue
    ThreadSafeQueue audio_queue;
    ThreadSafeQueue rds_queue;

    // Launch threads
    std::thread frontend(RF_FrontEnd_Thread, std::ref(audio_queue),std::ref(rds_queue), BLOCK_SIZE, DecmIQ, std::ref(h_IQ), std::ref(prev_data_i), std::ref(prev_data_q), std::ref(prev_i), std::ref(prev_q));

    std::thread audio(Audio_Thread, std::ref(audio_queue), BLOCK_SIZE, mode, outputType, DecmIQ, std::ref(h_Demod), std::ref(h_19kHz_BPF), std::ref(h_38kHz_BPF), DownsampleFactor, UpsampleFactor, std::ref(prev_data_mono), std::ref(prev_data_pilot),
                      std::ref(prev_data_stereo),
                      std::ref(delayed_buffer),
                      std::ref(prev_data_mixer),
                      std::ref(pllState_audio), Fs);

    std::thread rds(RDS_Thread, std::ref(rds_queue), BLOCK_SIZE, mode, outputType, DecmIQ, std::ref(h_Demod), std::ref(h_RDS_channel), std::ref(h_RDS_carrier), std::ref(h_RDS_downconversion),
                    std::ref(h_RDS_resampler),
                    std::ref(h_RRC),
                    RDS_DownsampleFactor,
                    RDS_UpsampleFactor,
                    SPS,
                    std::ref(prev_data_RDS_channel),
                    std::ref(prev_data_RDS_carrier),
                    std::ref(prev_data_RDS_downconverted),
                    std::ref(prev_data_resampler),
                    std::ref(delayed_buffer_RDS),
                    std::ref(prev_data_mixer),
                    std::ref(prev_data_RRC),
                    std::ref(pllState_RDS), Fs);
    

    // Wait for threads to finish
    frontend.join();
    audio.join();
    rds.join();

    return 0;
}
