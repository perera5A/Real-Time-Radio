/*
   Comp Eng 3DY4 (Computer Systems Integration Project)

   Department of Electrical and Computer Engineering
   McMaster University
   Ontario, Canada
*/

// This file shows how to write convolution unit tests, using Google C++ test framework.
// (it is based on https://github.com/google/googletest/blob/main/docs/index.md)

#include <limits.h>
#include "dy4.h"
#include "iofunc.h"
#include "filter.h"
#include "gtest/gtest.h"

namespace {

	class Convolution_Fixture: public ::testing::Test {

		public:

			const int N = 1024;	// signal size
			const int M = 101;	// kernel size
			const int lower_bound = -1;
			const int upper_bound = 1;
			const real EPSILON = 1e-4;

			std::vector<real> x, h, y_reference, y_test;

			Convolution_Fixture() {
				x.resize(N);
				h.resize(M);
				y_reference.resize(N + M - 1);
				y_test.resize(N + M - 1);
			}

			void SetUp() {
				generate_random_values(x, lower_bound, upper_bound);
				generate_random_values(h, lower_bound, upper_bound);
				convolveFIR_reference(y_reference, x, h);
			}

			void TearDown() {
			}

			~Convolution_Fixture() {
			}
	};

	TEST_F(Convolution_Fixture, convolveFIR_inefficient_NEAR) {

		convolveFIR_inefficient(y_test, x, h);

		ASSERT_EQ(y_reference.size(), y_test.size()) << "Output vector sizes for convolveFIR_reference and convolveFIR_inefficient are unequal";

		for (int i = 0; i < (int)y_reference.size(); ++i) {
			EXPECT_NEAR(y_reference[i], y_test[i], EPSILON) << "Original/convolveFIR_inefficient vectors differ at index " << i;
		}
	}

	TEST_F(Convolution_Fixture, convolveFIR_with_downsampling_NEAR) {
		int step_size = 4;
		
		// Downsample manually
		std::vector<real> y_reference_downsampled;
		for (size_t i = 0; i < y_reference.size(); i += step_size) {
			y_reference_downsampled.push_back(y_reference[i]);
		}

		// Compute with the function under test
		convolveFIR_with_downsampling(y_test, x, h, step_size);

		ASSERT_EQ(y_reference_downsampled.size(), y_test.size()) << "Output vector sizes for downsampled reference and convolveFIR_with_downsampling are unequal";

		for (int i = 0; i < (int)y_reference_downsampled.size(); ++i) {
			EXPECT_NEAR(y_reference_downsampled[i], y_test[i], EPSILON) 
				<< "Downsampled reference/convolveFIR_with_downsampling vectors differ at index " << i;
		}

	}

	TEST_F(Convolution_Fixture, convolveFIR_block_with_downsampling_NEAR) {

		// convolve full sample nomrally, downsample the reference 
		// split sample into blocks, convolve each block, concancenate together and test aginst reference
		int step_size = 4;
		int num_blocks = 4;
		int block_size = x.size() / num_blocks;
		std::vector<real> prev_samples(100, 0.0);  // Assume zero initial state
		
		// Compute full convolution output
		convolveFIR_reference(y_reference, x, h);
		
		// Downsample manually
		std::vector<real> y_reference_downsampled;
		for (size_t i = 0; i < y_reference.size(); i += step_size) {
			y_reference_downsampled.push_back(y_reference[i]);
		}

		std::vector<real> block(block_size);
		std::vector<real> y_block;
		std::vector<real> y_test;

		for (size_t i = 0; i < x.size(); i += block_size){
			block.assign(x.begin() + i, x.begin() + i + block_size);

			convolveFIR_block_with_downsampling(y_block, block, h, step_size, prev_samples);
			y_test.insert(y_test.end(), y_block.begin(), y_block.end());
		}

		ASSERT_EQ(y_reference_downsampled.size(), y_test.size()) << "Output sizes differ";

		for (size_t i = 0; i < y_reference_downsampled.size(); ++i) {
			EXPECT_NEAR(y_reference_downsampled[i], y_test[i], EPSILON)
				<< "Mismatch at index " << i;
			}
	}

	/*  
	TEST_F(Convolution_Fixture, convolveFIR_block_with_downsampling_with_unrolling_NEAR) {

		// convolve full sample nomrally, downsample the reference 
		// split sample into blocks, convolve each block, concancenate together and test aginst reference
		int step_size = 4;
		int num_blocks = 4;
		int block_size = x.size() / num_blocks;
		std::vector<real> prev_samples(100, 0.0);  // Assume zero initial state
		
		// Compute full convolution output
		convolveFIR_reference(y_reference, x, h);
		
		// Downsample manually
		std::vector<real> y_reference_downsampled;
		for (size_t i = 0; i < y_reference.size(); i += step_size) {
			y_reference_downsampled.push_back(y_reference[i]);
		}

		std::vector<real> block(block_size);
		std::vector<real> y_block;
		std::vector<real> y_test;

		for (size_t i = 0; i < x.size(); i += block_size){
			block.assign(x.begin() + i, x.begin() + i + block_size);

			convolveFIR_block_with_downsampling_with_unrolling(y_block, block, h, step_size, prev_samples);
			y_test.insert(y_test.end(), y_block.begin(), y_block.end());
		}

		ASSERT_EQ(y_reference_downsampled.size(), y_test.size()) << "Output sizes differ";

		for (size_t i = 0; i < y_reference_downsampled.size(); ++i) {
			EXPECT_NEAR(y_reference_downsampled[i], y_test[i], EPSILON)
				<< "Mismatch at index " << i;
			}
	}

	*/
	
	// Modified Mode 2 tests
	TEST_F(Convolution_Fixture, convolveFIR_manual_upsample_then_downsample_vs_polyphase_NEAR) {
		// Mode 2 parameters
		int upsample_factor = 147;
		int downsample_factor = 800;
		
		// Resize y_reference to 10240 and fill with random values
		y_reference.resize(10240);
		generate_random_values(y_reference, lower_bound, upper_bound);
		
		// Manual upsample then downsample
		std::vector<real> upsampled_fm_demod;
		std::vector<real> downsampled_fm_demod(ceil(y_reference.size() * upsample_factor / downsample_factor));
		
		// Polyphase version
		std::vector<real> y_output2(ceil(y_reference.size() * upsample_factor / downsample_factor));
		
		// Filter parameters
		int Fs = 240e3;
		int Fc = 16e3;
		std::vector<real> h_Demod;
		impulseResponseLPF_upscale(Fs, Fc, 101, h_Demod, upsample_factor);
		
		// State vectors
		std::vector<real> prev_data_demod(101 * upsample_factor - 1, 0.0);
		
		// Manual method
		upsampler(upsample_factor, y_reference, upsampled_fm_demod);
		convolveFIR_block_with_downsampling(downsampled_fm_demod, upsampled_fm_demod, h_Demod, 
											downsample_factor, prev_data_demod);
		
		// Reset state
		std::fill(prev_data_demod.begin(), prev_data_demod.end(), 0.0);
		
		// Polyphase method
		convolveFIR_block_polyphase(y_output2, y_reference, h_Demod, downsample_factor, 
									upsample_factor, prev_data_demod);
		
		// Verify sizes match
		ASSERT_EQ(downsampled_fm_demod.size(), y_output2.size()) << "Output sizes differ";
		
		// Compare outputs
		for (size_t i = 0; i < downsampled_fm_demod.size(); ++i) {
			EXPECT_NEAR(y_output2[i], downsampled_fm_demod[i], EPSILON)
				<< "Manual resampling and polyphase convolution results differ at index " << i;
		}
	}

TEST_F(Convolution_Fixture, fast_resampler_block_processing_NEAR) {
		// Mode 2 parameters
		int upsample_factor = 147;
		int downsample_factor = 800;
		int num_blocks = 4;
		
		// Resize y_reference to 10240 and fill with random values
		y_reference.resize(10240);
		generate_random_values(y_reference, lower_bound, upper_bound);
		
		int block_size = y_reference.size() / num_blocks;  // 2560 samples per block
		int output_block_size = ceil(block_size * (upsample_factor / downsample_factor));  // ~470 samples
		
		// Filter setup
		int Fs = 240e3;
		int Fc = 16e3;
		std::vector<real> h_Demod;
		impulseResponseLPF_upscale(Fs, Fc, 101, h_Demod, upsample_factor);
		
		// State vectors
		std::vector<real> prev_samples_poly((101 * upsample_factor) - 1, 0.0);
		std::vector<real> prev_samples_slow((101 * upsample_factor) - 1, 0.0);
		
		// Fast method (polyphase) with blocks
		std::vector<real> y_test;
		std::vector<real> block(block_size);
		std::vector<real> y_block(output_block_size);
		prev_samples_poly.assign((101 * upsample_factor) - 1, 0.0);  // Reset state

		
		for (size_t i = 0; i < y_reference.size(); i += block_size) {
			block.assign(y_reference.begin() + i, y_reference.begin() + i + block_size);
			convolveFIR_block_polyphase(y_block, block, h_Demod, downsample_factor, 
										upsample_factor, prev_samples_poly);
			y_test.insert(y_test.end(), y_block.begin(), y_block.end());
		}
		
		// Slow method with blocks
		std::vector<real> y_test2;
		prev_samples_slow.assign((101 * upsample_factor) - 1, 0.0);  // Reset state
		
		for (size_t i = 0; i < y_reference.size(); i += block_size) {
			block.assign(y_reference.begin() + i, y_reference.begin() + i + block_size);
			std::vector<real> upsampled_signal;
			upsampler(upsample_factor, block, upsampled_signal);
			convolveFIR_block_with_downsampling(y_block, upsampled_signal, h_Demod, 
												downsample_factor, prev_samples_slow);
			y_test2.insert(y_test2.end(), y_block.begin(), y_block.end());
		}
		
		// Verify results
		ASSERT_EQ(y_test2.size(), y_test.size()) << "Fast method amd slow method size mismatch";
		
		for (size_t i = 0; i < y_test2.size(); ++i) {
			
			EXPECT_NEAR(y_test2[i], y_test[i], EPSILON)
				<< "Slow method and fast method mismatch" << i;
		}
	}







	

} // end of namespace

