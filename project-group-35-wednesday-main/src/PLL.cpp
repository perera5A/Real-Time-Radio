#include "dy4.h"
#include "PLL.h"

void fmPLL(std::vector<real> &ncoOut, std::vector<real> &pllIn, real freq, real Fs, PLL_state &state, real ncoScale, real phaseAdjust, real normBandwidth) {

    const real Cp = 2.666;
    const real Ci = 3.555;

    real Kp = normBandwidth * Cp;
    real Ki = (normBandwidth * normBandwidth) * Ci;

    ncoOut.resize(pllIn.size() + 1);

    // Declare state variables
    real integrator, phaseEst, feedbackI, feedbackQ, trigOffset;

    // Initialize from the passed state
    integrator = state.integrator;
    phaseEst = state.phaseEst;
    feedbackI = state.feedbackI;
    feedbackQ = state.feedbackQ;
    trigOffset = state.trigOffset;
    ncoOut[0] = state.ncoOut_0;

    for (size_t k = 0; k < pllIn.size(); k++) {
        real errorI = pllIn[k] * feedbackI;
        real errorQ = pllIn[k] * -feedbackQ;

        real errorD = std::atan2(errorQ, errorI);

        integrator += Ki * errorD;
        phaseEst += Kp * errorD + integrator;

        trigOffset += 1;
        real trigArg = 2 * PI * (freq / Fs) * trigOffset + phaseEst;
        feedbackI = std::cos(trigArg);
        feedbackQ = std::sin(trigArg);

        ncoOut[k + 1] = std::cos(trigArg * ncoScale + phaseAdjust);
    }

    state.integrator = integrator;
    state.phaseEst = phaseEst;
    state.feedbackI = feedbackI;
    state.feedbackQ = feedbackQ;
    state.ncoOut_0 = ncoOut[ncoOut.size()-1];
    state.trigOffset = trigOffset;
}

void fmPLLIQ(std::vector<real> &ncoOut, std::vector<real> &ncoQ, std::vector<real> &pllIn, 
             real freq, real Fs, PLL_stateIQ &state, 
             real ncoScale = 1.0, real phaseAdjust = 0.0, real normBandwidth = 0.01) {

    const real Cp = 2.666;
    const real Ci = 3.555;

    real Kp = normBandwidth * Cp;
    real Ki = (normBandwidth * normBandwidth) * Ci;
    
    // Resize both output arrays to match input size + 1
    ncoOut.resize(pllIn.size() + 1);
    ncoQ.resize(pllIn.size() + 1);

    // Declare state variables with default initialization
    real integrator = state.integrator;
    real phaseEst = state.phaseEst;
    real feedbackI = state.feedbackI;
    real feedbackQ = state.feedbackQ;
    real trigOffset = state.trigOffset;

    // Initialize first output samples
    ncoOut[0] = state.ncoI;
    ncoQ[0] = state.ncoQ;

    
    for (size_t k = 0; k < pllIn.size(); k++) {
        real errorI = pllIn[k] * feedbackI;
        real errorQ = pllIn[k] * -feedbackQ;

        real errorD = std::atan2(errorQ, errorI);

        integrator += Ki * errorD;
        phaseEst += Kp * errorD + integrator;

        trigOffset += 1;
        real trigArg = 2 * PI * (freq / Fs) * trigOffset + phaseEst;
        feedbackI = std::cos(trigArg);
        feedbackQ = std::sin(trigArg);

        ncoOut[k + 1] = std::cos(trigArg * ncoScale + phaseAdjust);
        ncoQ[k + 1] = std::sin(trigArg * ncoScale + phaseAdjust);
    }

    // Update state
    state.integrator = integrator;
    state.phaseEst = phaseEst;
    state.feedbackI = feedbackI;
    state.feedbackQ = feedbackQ;
    state.ncoI = ncoOut[ncoOut.size() - 1];
    state.ncoQ = ncoQ[ncoQ.size() - 1];
    state.trigOffset = trigOffset;
}


