/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#ifndef DY4_PLL_H
#define DY4_PLL_H

// Add headers as needed
#include <iostream>
#include <vector>
#include <cmath>

typedef struct {
    float integrator;
    float phaseEst;
    float feedbackI;
    float feedbackQ;
    float ncoOut_0;
    float ncoQ;
    int trigOffset;
} PLL_state;

typedef struct {
    float integrator;
    float phaseEst;
    int trigOffset;
    float feedbackI;
    float feedbackQ;
    float ncoI;
    float ncoQ;
} PLL_stateIQ;


void fmPLL(std::vector<real> &ncoOut, std::vector<real> &pllIn, real freq, real Fs, PLL_state &state, real ncoScale, real phaseAdjust, real normBandwidth);
void fmPLLIQ(std::vector<real> &ncoOut, std::vector<real> &ncoQ, std::vector<real> &pllIn, real freq, real Fs, PLL_stateIQ &state, real ncoScale, real phaseAdjust, real normBandwidth);

#endif // DY4_PLL_H
