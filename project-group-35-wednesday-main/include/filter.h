/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#ifndef DY4_FILTER_H
#define DY4_FILTER_H

// Add headers as needed
#include <iostream>
#include <vector>

// Declaration of function prototypes

void convolveFIR_block_polyphase(std::vector<real> &y, std::vector<real> &x, std::vector<real> &h, int down, int up, std::vector<real> &prev_samples); 

void impulseResponseLPF(real Fs, real Fc, unsigned short int num_taps, std::vector<real> &h, int gain);

void impulseResponseBPF(real Fs, real F_low, real F_high, unsigned short int num_taps, std::vector<real> &h);

void impulseResponseRRC(real Fs, unsigned short int num_taps, std::vector<real> &h);

void manchesterDecoding(std::vector<float>& input_signal, int SPS);

void impulseResponseLPF_upscale(float Fs, float Fc, unsigned short int num_taps, std::vector<real> &h, unsigned int upsampling_factor);


//////////////////////////////////////////////////////////////
// New code as part of benchmarking/testing and the project
//////////////////////////////////////////////////////////////

void upsampler (int upsampleSize, std::vector<real> &inputSignal, std::vector<real> &outputSignal);

void delay_block(std::vector<real>& block, std::vector<real>& delayed_block, std::vector<real>& delay_buffer);

void convolveFIR_inefficient(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h);
void convolveFIR_reference(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h);

void convolveFIR_with_downsampling(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h, int step_size);

void convolveFIR_block_with_downsampling(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h, int step_size, std::vector<real> &prev_samples);

void convolveFIR_block_with_downsampling_with_unrolling(std::vector<real> &y, std::vector<real> &x, std::vector<real> &h, int step_size, std::vector<real> &prev_samples);


void allPass(const std::vector<real> &input_block, std::vector<real> &state_block,std::vector<real> &output_block);

#endif // DY4_FILTER_H

