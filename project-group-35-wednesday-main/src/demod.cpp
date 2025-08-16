#include "dy4.h"
#include "demod.h"

void ownfmDemod(std::vector<real> I, std::vector<real> Q, real &prev_I, real &prev_Q, std::vector<real> &fm_demod) {
    fm_demod.clear();
    fm_demod.resize(I.size());  // Fix incorrect .size syntax

    for (size_t k = 0; k < I.size(); k++) {  // Use size_t to match vector indexing
        real I_derivative = I[k] - prev_I;
        real Q_derivative = Q[k] - prev_Q;

        real denominator = I[k] * I[k] + Q[k] * Q[k];
        
        real current_phase = 0;
        if (denominator != 0) {  // Prevent division by zero
            current_phase = (I[k] * Q_derivative - Q[k] * I_derivative) / denominator;
        }

        fm_demod[k] = current_phase;

        prev_I = I[k];
        prev_Q = Q[k];
    }
}

