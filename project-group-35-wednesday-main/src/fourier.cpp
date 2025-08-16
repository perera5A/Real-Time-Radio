/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

// Source code for Fourier-family of functions
#include "dy4.h"
#include "fourier.h"

// Just DFT function (no FFT)
void DFT(const std::vector<real> &x, std::vector<std::complex<real>> &Xf) {
	Xf.clear();
	Xf.resize(x.size(), std::complex<real>(0));
	for (int m = 0; m < (int)Xf.size(); m++) {
		for (int k = 0; k < (int)x.size(); k++) {
			std::complex<real> expval(0, -2 * PI * (k * m) / x.size());
			Xf[m] += x[k] * std::exp(expval);
		}
	}
}

// Function to compute the magnitude values in a complex vector
void computeVectorMagnitude(const std::vector<std::complex<real>> &Xf, std::vector<real> &Xmag)
{
	Xmag.clear();
	Xmag.resize(Xf.size(), real(0));
	for (int i = 0; i < (int)Xf.size(); i++) {
		Xmag[i] = std::abs(Xf[i]) / Xf.size();
	}
}

// Add your own code to estimate the PSD
// ...
void estimatePSD(const std::vector<real> &samples, int Fs, std::vector<real> &freq, std::vector<real> &psd_est){
	int freq_bins = NFFT;

	float df = (float) Fs / (float) freq_bins;

	freq.clear();
	freq.resize(freq_bins / 2);
	for (int i = 0; i < freq_bins / 2; i++){
		freq[i] = i * df;
	}

	std::vector<real> hann(freq_bins);
	for (int i = 0; i < freq_bins; i++){
		hann[i] = 0.5 * (1 - std::cos(2 * PI * i / (freq_bins - 1)));
	}

	int no_segments = (int)(std::floor((float)samples.size() / (float)freq_bins));

	std::vector<real> psd_list(no_segments * freq_bins / 2);
	int index = 0;

	for (int k = 0; k < no_segments; k++){
		
		std::vector<real> windowed_samples(freq_bins);

		for (int i = k * freq_bins, j = 0; i < (k + 1) * freq_bins; i++, j++){
			windowed_samples[j] = samples[i] * hann[j];
		}

		std::vector<std::complex<real>> Xf;
		DFT(windowed_samples, Xf);
		Xf.resize(Xf.size() / 2);

		std::vector<real> psd_seg(Xf.size());

		for (int i = 0; i < (int) Xf.size(); i++){
			psd_seg[i]	= 2 * (1.0 / (Fs * freq_bins / 2.0)) * std::pow(std::abs(Xf[i]), 2);
			psd_list[index] = psd_seg[i];
			index++;
		}
	}

	std::vector<real> psd_seg(freq_bins / 2, 0);

	for (int k = 0; k < freq_bins / 2; k++){
		for (int l = 0; l < no_segments; l++){
			psd_seg[k] += psd_list[k + l * (int)(freq_bins / 2)];
		}

		psd_seg[k] = psd_seg[k] / no_segments;
	}

	psd_est.clear();
	psd_est.resize(freq_bins / 2);
	for (int k = 0; k < freq_bins / 2; k++){
		psd_est[k] = 10 * std::log10(psd_seg[k]);
	}
}
//////////////////////////////////////////////////////////////
// New code as part of benchmarking/testing and the project
//////////////////////////////////////////////////////////////

void DFT_reference(const std::vector<real> &x, std::vector<std::complex<real>> &Xf) {

	Xf.clear();
	Xf.resize(x.size(), std::complex<real>(0));
	for (int m = 0; m < (int)Xf.size(); m++) {
		for (int k = 0; k < (int)x.size(); k++) {
			std::complex<real> expval(0, -2 * M_PI * (k * m) / x.size());
			Xf[m] +=  + x[k] * std::exp(expval);
		}
	}
}

void DFT_init_bins(const std::vector<real> &x, std::vector<std::complex<real>> &Xf) {

	int N = (int)x.size();
	std::fill(Xf.begin(), Xf.end(), std::complex<real>(0., 0.));
	for (int m = 0; m < N; m++) {
		for (int k = 0; k < N; k++) {
			std::complex<real> expval(0, -2 * M_PI * (k * m) / N);
			Xf[m] += x[k] * std::exp(expval);
		}
	}
}

void generate_DFT_twiddles(const int& N, std::vector<std::complex<real>> &Twiddle1D) {

	Twiddle1D.resize(N);
	for (int k = 0; k < N; k++) {
		std::complex<real> expval(0, -2 * M_PI * k / N);
		Twiddle1D[k] = std::exp(expval);
	}
}

void generate_DFT_matrix(const int& N, std::vector<std::vector<std::complex<real>>> &Twiddle2D) {

	Twiddle2D.resize(N, std::vector<std::complex<real>>(N));
    std::vector<std::complex<real>> Twiddle1D;
	generate_DFT_twiddles(N, Twiddle1D);

	for (int m = 0; m < N; m++) {
		for (int k = 0; k < N; k++) {
			Twiddle2D[m][k] = Twiddle1D[(k * m) % N];
		}
	}
}

