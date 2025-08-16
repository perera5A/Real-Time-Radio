import numpy as np
import math

def fmPll(pllIn, freq, Fs, ncoScale=1.0, phaseAdjust=0.0, normBandwidth=0.01, state=None):
	"""
	pllIn 	 		array of floats
					input signal to the PLL (assume known frequency)

	freq 			float
					reference frequency to which the PLL locks

	Fs  			float
					sampling rate for the input/output signals

	ncoScale		float
					frequency scale factor for the NCO output

	phaseAdjust		float
					phase adjust to be added to the NCO output only

	normBandwidth	float
					normalized bandwidth for the loop filter
					(relative to the sampling rate)

	state 			dict (optional)
					previous state to continue processing
	"""

	# scale factors for proportional/integrator terms
	Cp = 2.666
	Ci = 3.555
	Kp = normBandwidth * Cp
	Ki = (normBandwidth ** 2) * Ci

	# output array for the NCO
	ncoOut = np.empty(len(pllIn) + 1)

	# initialize internal state
	if state is None:
		integrator = 0.0
		phaseEst = 0.0
		trigOffset = 0
		feedbackI = 1.0
		feedbackQ = 0.0
		ncoOut[0] = 1.0
	else:
		integrator = state.get('integrator', 0.0)
		phaseEst = state.get('phaseEst', 0.0)
		trigOffset = state.get('trigOffset', 0)
		feedbackI = state.get('feedbackI', 1.0)
		feedbackQ = state.get('feedbackQ', 0.0)
		ncoOut[0] = state.get('ncoOut0', 1.0)
    

	for k in range(len(pllIn)):
		# Phase detector
		errorI = pllIn[k] * feedbackI
		errorQ = pllIn[k] * -feedbackQ

		# Arctangent discriminator
		errorD = math.atan2(errorQ, errorI)

		# Loop filter
		integrator += Ki * errorD

		# Update phase estimate
		phaseEst += Kp * errorD + integrator

		# Internal oscillator
		trigOffset += 1
		trigArg = 2 * math.pi * (freq / Fs) * trigOffset + phaseEst
		feedbackI = math.cos(trigArg)
		feedbackQ = math.sin(trigArg)
		ncoOut[k + 1] = math.cos(trigArg * ncoScale + phaseAdjust)

    # Save updated state
	updated_state = {
        'integrator': integrator,
        'phaseEst': phaseEst,
        'trigOffset': trigOffset,
        'feedbackI': feedbackI,
        'feedbackQ': feedbackQ,
		'ncoOut0': ncoOut[-1]
    }

	return ncoOut, updated_state

def fmPllwithIQ(pllIn, freq, Fs, ncoScale=1.0, phaseAdjust=0.0, normBandwidth=0.01, state=None):
	"""
	pllIn 	 		array of floats
					input signal to the PLL (assume known frequency)

	freq 			float
					reference frequency to which the PLL locks

	Fs  			float
					sampling rate for the input/output signals

	ncoScale		float
					frequency scale factor for the NCO output

	phaseAdjust		float
					phase adjust to be added to the NCO output only

	normBandwidth	float
					normalized bandwidth for the loop filter
					(relative to the sampling rate)

	state 			dict (optional)
					previous state to continue processing
	"""

	# scale factors for proportional/integrator terms
	Cp = 2.666
	Ci = 3.555
	Kp = normBandwidth * Cp
	Ki = (normBandwidth ** 2) * Ci

	# output array for the NCO
	ncoOutI = np.empty(len(pllIn) + 1)
	ncoOutQ = np.empty(len(pllIn) + 1)


	# initialize internal state
	if state is None:
		integrator = 0.0
		phaseEst = 0.0
		trigOffset = 0
		feedbackI = 1.0
		feedbackQ = 0.0
		ncoOutI[0] = math.cos(0)  # Initialize with cos(0) = 1.0
		ncoOutQ[0] = math.sin(0)  # Initialize with cos(0) = 1.0
	else:
		integrator = state.get('integrator', 0.0)
		phaseEst = state.get('phaseEst', 0.0)
		trigOffset = state.get('trigOffset', 0)
		feedbackI = state.get('feedbackI', 1.0)
		feedbackQ = state.get('feedbackQ', 0.0)
		ncoOutI[0] = state.get('ncoOutI', 1.0)
		ncoOutQ[0] = state.get('ncoOutQ', 0.0)
    

	for k in range(len(pllIn)):
		# Phase detector
		errorI = pllIn[k] * feedbackI
		errorQ = pllIn[k] * -feedbackQ

		# Arctangent discriminator
		errorD = math.atan2(errorQ, errorI)

		# Loop filter
		integrator += Ki * errorD

		# Update phase estimate
		phaseEst += Kp * errorD + integrator

		# Internal oscillator
		trigOffset += 1
		trigArg = 2 * math.pi * (freq / Fs) * trigOffset + phaseEst
		feedbackI = math.cos(trigArg)
		feedbackQ = math.sin(trigArg)
		ncoOutI[k + 1] = math.cos(trigArg * ncoScale + phaseAdjust)
		ncoOutQ[k + 1] = math.sin(trigArg * ncoScale + phaseAdjust)

    # Save updated state
	updated_state = {
        'integrator': integrator,
        'phaseEst': phaseEst,
        'trigOffset': trigOffset,
        'feedbackI': feedbackI,
        'feedbackQ': feedbackQ,
		'ncoOutI': ncoOutI[-1],
		'ncoOutQ': ncoOutQ[-1]
    }

	return ncoOutI, ncoOutQ, updated_state,

if __name__ == "__main__":
	pass
