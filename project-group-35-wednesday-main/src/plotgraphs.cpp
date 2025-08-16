/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Copyright by Nicola Nicolici
Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#include "dy4.h"
#include "filter.h"
#include "fourier.h"
#include "genfunc.h"
#include "iofunc.h"
#include "logfunc.h"

int main()
{

	// By default, we use floats; if you wish to use double precision, compile as shown below:
	//
	// make DEFINES="-DDOUBLE"
	//
	// -DDOUBLE will define the DOUBLE macro for conditional compilation
	// if there are no changes to the source files make sure to do a "make clean" first
	std::cout << "Working with reals on " << sizeof(real) << " bytes" << std::endl;

	// Binary files can be generated through the
	// Python models from the "../model/" sub-folder
	const std::string in_fname = "../data/fm_demod_10.bin";
	std::vector<real> bin_data;
	readBinData(in_fname, bin_data);

	// Generate an index vector to be used by logVector on the X axis
	std::vector<real> vector_index;
	genIndexVector(vector_index, bin_data.size());
	// Log time data in the "../data/" subfolder in a file with the following name
	// note: .dat suffix will be added to the log file in the logVector function
	logVector("demod_time", vector_index, bin_data);

	// Take a slice of data with a limited number of samples for the Fourier transform
	// note: NFFT constant is actually just the number of points for the
	// Fourier transform - there is no FFT implementation ... yet
	// unless you wish to wait for a very long time, keep NFFT at 1024 or below
	std::vector<real> slice_data = \
		std::vector<real>(bin_data.begin(), bin_data.begin() + NFFT);
	// note: make sure that binary data vector is big enough to take the slice

	// Declare a vector of complex values for DFT
	std::vector<std::complex<real>> Xf;
	// ... In-lab ...
	// Compute the Fourier transform
	// the function is already provided in fourier.cpp

	DFT(slice_data, Xf);

	// Compute the magnitude of each frequency bin
	// note: we are concerned only with the magnitude of the frequency bin
	// (there is no logging of the phase response)
	std::vector<real> Xmag;
	// ... In-lab ...
	// Compute the magnitude of each frequency bin
	// the function is already provided in fourier.cpp

	computeVectorMagnitude(Xf, Xmag);

	// Log the frequency magnitude vector
	vector_index.clear();
	genIndexVector(vector_index, Xmag.size());
	logVector("demod_freq", vector_index, Xmag); // Log only positive freq

	// For your take-home exercise - repeat the above after implementing
	// your own function for PSD based on the Python code that has been provided
	// note the estimate PSD function should use the entire block of "bin_data"
	//
	// ... Complete as part of the take-home ...
	//

	std::vector<real> psd;
	std::vector<real> freq;
	int Fs = 240;
	estimatePSD(bin_data, Fs, freq, psd);

	logVector("demod_psd", freq, psd);

	// If you wish to write some binary files, see below example
	//
	// const std::string out_fname = "../data/outdata.bin";
	// writeBinData(out_fname, bin_data);
	//
	// output files can be imported, for example, in Python
	// for additional analysis or alternative forms of visualization

	// Naturally, you can comment the line below once you are comfortable to run GNU plot
	std::cout << "Run: gnuplot -e 'set terminal png size 1024,768' ../data/example.gnuplot > ../data/example.png\n";

	return 0;
}