/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#include "dy4.h"
#include "filter.h"
#include <cmath>

// Manchester decoding for high low differentiation
/* 
void manchesterDecoding(std::vector<float>& input_signal, int SPS) {
    std::vector<int> decoded_bits;

    for (float i = SPS / 2; i + SPS < input_signal.size(); i ++) {
        float first_sample = input_signal[i];
        float second_sample = input_signal[i + SPS];

        // Check for Manchester encoding: high-to-low (1) or low-to-high (0)
        if (first_sample > 0 && second_sample < 0) {
            decoded_bits.push_back(1);  // Transition from high to low
        } else if (first_sample < 0 && second_sample > 0) {
            decoded_bits.push_back(0);  // Transition from low to high
        } else {
            decoded_bits.push_back(-1); // Invalid transition (use -1 for None)
        }
    }

    return decoded_bits;
}
*/
// function to compute the impulse response "h" based on the sinc function
void impulseResponseLPF_upscale(float Fs, float Fc, unsigned short int num_taps, std::vector<real> &h, unsigned int upsampling_factor)
{
	// update num_taps and sampling rate to reflect upsampling factor
	num_taps *= upsampling_factor;
	Fs *= upsampling_factor;

	// allocate memory for the impulse response
	h.clear(); h.resize(num_taps, 0.0);

	float normf = Fc / (Fs/2); // Normalize cutoff
	for (int i = 0; i < num_taps; ++i) {
		if (i == (num_taps - 1) / 2) {
			h[i] = normf;
		}
		else{
			h[i] = normf * ((std::sin(PI * normf * (i - (num_taps - 1) / 2))) / (PI * normf * (i - (num_taps - 1) / 2) ));	
		}
		h[i] *= pow(std::sin(i*PI/num_taps), 2) * upsampling_factor;
	}
}


void upsampler(int upsampleSize, std::vector<real> &inputSignal, std::vector<real> &outputSignal) {
    // Resize the output signal
    outputSignal.assign(inputSignal.size() * upsampleSize, 0);  // Fills with zeros

    // Insert the input signal into the correct positions
    for (size_t i = 0; i < inputSignal.size(); ++i) {
        outputSignal[i * upsampleSize] = inputSignal[i];
    }
}



void allPass(const std::vector<real> &input_block, std::vector<real> &state_block,std::vector<real> &output_block) 
{
    output_block.clear();
    output_block.insert(output_block.end(), state_block.begin(), state_block.end());
    output_block.insert(output_block.end(), input_block.begin(), input_block.end()-state_block.size());
    int state_size = state_block.size();
    state_block.clear();
    state_block.insert(state_block.end(), input_block.end()-state_size, input_block.end());
}


void delay_block(std::vector<real>& block, std::vector<real>& delayed_block, std::vector<real>& delay_buffer) {
    // Concatenate delay_buffer and block
    std::vector<real> delayed_output;
    delayed_output.reserve(delay_buffer.size() + block.size());
    delayed_output.insert(delayed_output.end(), delay_buffer.begin(), delay_buffer.end());
    delayed_output.insert(delayed_output.end(), block.begin(), block.end());

    // Modify block to be the delayed block (first part of delayed_output)
    delayed_block.clear();
    delayed_block.resize(block.size());
    delayed_block.assign(delayed_output.begin(), delayed_output.begin() + block.size());

    // Modify delay_buffer to be the new delay buffer (remaining part of delayed_output)
    delay_buffer.assign(delayed_output.begin() + block.size(), delayed_output.end());
}




void convolveFIR_block_polyphase(std::vector<real> &y, 
                                 std::vector<real> &x, 
                                 std::vector<real> &h, 
                                 int down, int up, 
                                 std::vector<real> &prev_samples) {
    int h_size = h.size();
    int x_size = x.size();
    int y_size = (x_size * up) / down;
    int overlap = prev_samples.size();
    
    y.clear();
    y.resize(y_size, 0.0);
    
    for (int i = 0; i < y_size; i++) {
        int temp = i * down;
        int phase = (down * i) % up;
        int fix = (temp - phase) / up;

        for (int k = phase; k < h_size; k += up) {
            int index = temp - k;
            if (index >= 0) {
                y[i] += h[k] * x[fix];
            } else {
                y[i] += h[k] * prev_samples[overlap + fix];
            }
            fix--;
        }
    }

    // Update previous samples for next block processing
    int num_samples_to_save = ceil((h_size - 1) / (real)up);
    int start_index = x_size - num_samples_to_save;
    if (start_index < 0) start_index = 0;
    prev_samples.assign(x.begin() + start_index, x.end());

}





// Function to compute the impulse response "h" based on the sinc function
void impulseResponseLPF(real Fs, real Fc, unsigned short int num_taps, std::vector<real> &h, int gain)
{
	// Allocate memory for the impulse response
	h.clear();
	h.resize(num_taps, 0.0);

	 // Normalize the cutoff frequency
    real norm_cutoff = Fc / (Fs / 2);
    real middle_index = (num_taps - 1) / 2;


    // Loop over each filter tap
    for (int i = 0; i < num_taps; i++) {
        if (i == middle_index) {
            h[i] = norm_cutoff; // Handle the middle index separately
        } else {
            h[i] = (norm_cutoff * sin(M_PI * norm_cutoff * (i - middle_index)) / (M_PI * norm_cutoff * (i - middle_index)));
        }

        // Apply the window function (Hamming window)
        h[i] *= 0.5 - 0.5 * cos(2 * M_PI * i / (num_taps - 1));
    
        h[i] *= gain;
    }


    /*
    if (num_taps > 102) {
        for (int i = 0; i < (num_taps); i++) {
                h[i] *= gain;
        }
    }
   */


}

// Function to compute the impulse response "h" based on the sinc function
void impulseResponseBPF(real Fs, real F_low, real F_high, unsigned short int num_taps, std::vector<real> &h)
{
	// Allocate memory for the impulse response
	h.clear();
	h.resize(num_taps, 0.0);

	 // Normalize the cutoff frequency
    int center = (num_taps - 1) / 2;
    real F_mid = (F_low + F_high) / 2.0;
    real scale_factor = 0;

    real norm_F_low = 2 * F_low / Fs;
    real norm_F_high = 2 * F_high / Fs;
    real norm_f_mid = 2 * F_mid / Fs;

    for (int i = 0; i < num_taps; i++) {
        int n = i - center;

        if (n == 0) {
            h[i] = norm_F_high - norm_F_low;
        } else {
            h[i] = (sin(M_PI * n * norm_F_high) - sin(M_PI * n * norm_F_low)) / (M_PI * n);
        }

        // Apply the window function (Hann window)
        h[i] *= 0.5 - 0.5 * cos(M_PI * i / (center));
        scale_factor += h[i] * cos(M_PI * n * norm_f_mid);
    
    }

    for (int i = 0; i < num_taps; i++){
        h[i] /= scale_factor;
    }

}

void impulseResponseRRC(real Fs, unsigned short int num_taps, std::vector<real> &h)
{
    real T_symbol = 1/2375.0;

    real beta = 0.90;

    h.clear();
    h.resize(num_taps, 0.0);

    for (int k = 0; k < num_taps; k++) {
        real t = real((k - num_taps/2))/Fs;
        if (t == 0.0) {
            h[k] = 1.0 + beta*((4/PI)-1);
        }
        else if (t == -T_symbol/(4*beta) || t == T_symbol/(4*beta)) {
            h[k] = (beta/std::sqrt(2))*(((1+2/PI)*(std::sin(PI/(4*beta)))) + ((1-2/PI)*(std::cos(PI/(4*beta)))));
        }
        else {
            h[k] = (std::sin(PI*t*(1-beta)/T_symbol) +  \
					4*beta*(t/T_symbol)*std::cos(PI*t*(1+beta)/T_symbol))/ \
					(PI*t*(1-(4*beta*t/T_symbol)*(4*beta*t/T_symbol))/T_symbol);
        }
    }


}


//////////////////////////////////////////////////////////////
// New code as part of benchmarking/testing and the project
//////////////////////////////////////////////////////////////

void convolveFIR_inefficient(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h) {
    y.clear();
    y.resize(int(x.size()), 0.0);
    for (auto n = 0; n < (int)y.size(); n++) {
        for (auto k = 0; k < (int)x.size(); k++) {
            if ((n - k >= 0) && (n - k) < (int)h.size())
                y[n] += x[k] * h[n - k];
        }
    }
}

void convolveFIR_reference(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h) {
    y.clear();
    y.resize(int(x.size()), 0.0);

    for (auto n = 0; n < (int)y.size(); n++) {
        for (auto k = 0; k < (int)h.size(); k++) {
            if ((n - k >= 0) && (n - k) < (int)x.size())
                y[n] += h[k] * x[n - k];
        }
    }
}



void convolveFIR_with_downsampling(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h, int step_size) {
    y.clear();
    y.resize(x.size() / step_size, 0.0);

    for (auto n = 0; n < (int)y.size(); n++) {
        for (auto k = 0; k < (int)h.size(); k++) {
            int index = n * step_size - k;
            if (index >= 0 && index < (int)x.size()) {
                y[n] += h[k] * x[index];
            }
        }
    }
}

void convolveFIR_block_with_downsampling(std::vector<real> &y, 
                                         const std::vector<real> &x, 
                                         const std::vector<real> &h, 
                                         int step_size, 
                                         std::vector<real> &prev_samples) {
    int overlap = prev_samples.size();
    int x_size = x.size();
    int h_size = h.size();

    // Compute the correct y_size (accounting for downsampling factor)
    int y_size = x_size / step_size;

    y.clear();
    y.resize(y_size, 0.0);

    for (int n = 0; n < y_size; n++) {
        int x_index = n * step_size;
        for (int k = 0; k < h_size; k++) {
            int index = x_index - k;

            y[n] += h[k] * ((index >= 0) ? x[index] : prev_samples[overlap + index]);
            
        }
    }

	 // Update previous samples for next block processing
    if (x_size >= overlap) { 
        std::copy(x.end() - overlap, x.end(), prev_samples.begin());
    }

}

void convolveFIR_block_with_downsampling_with_unrolling(std::vector<real> &y, 
                                         std::vector<real> &x, 
                                         std::vector<real> &h, 
                                         int step_size, 
                                         std::vector<real> &prev_samples) {
    int overlap = prev_samples.size();
    int x_size = x.size();
    int h_size = h.size();

    // Compute the correct y_size (accounting for downsampling factor)
    int y_size = x_size / step_size;

    y.clear();
    y.resize(y_size, 0.0);

    for (int n = 0; n < y_size; n++) {
        int x_index = n * step_size;

        int kmin = 0;
        int kmax = ((int) h.size() < n + 1)? (int) h.size(): n + 1;
        int unrolled_limit = kmin + ((kmax - kmin) / 4) * 4;

        
        for (auto k = kmin; k < unrolled_limit; k += 4) {
            int index = x_index - k;

            y[n] += h[k] * ((index >= 0) ? x[index] : prev_samples[overlap + index]) 
                    + h[k+1] * ((index - 1 >= 0) ? x[index - 1] : prev_samples[overlap + index - 1])
                    + h[k+2] * ((index - 2 >= 0) ? x[index - 2] : prev_samples[overlap + index - 2])
                    + h[k+3] * ((index - 3 >= 0) ? x[index - 3] : prev_samples[overlap + index - 3]);
            
        }

        for (auto k = unrolled_limit; k < kmax; k++){
            int index = x_index - k;
            y[n] += h[k] * ((index >= 0) ? x[index] : prev_samples[overlap + index]);
        }
    }

	 // Update previous samples for next block processing
    if (x_size >= overlap) { 
        std::copy(x.end() - overlap, x.end(), prev_samples.begin());
    }

}

