#ifndef DY4_RDSSUPPORT_H
#define DY4_RDSSUPPORT_H

#include <vector>
#include <array>
#include <deque>
#include <tuple>
#include <string>
#include <iostream>
#include "math.h"

void normalize(std::vector<real>& data, std::vector<real> &data2, std::vector<real>& normalized_data);

void parity_check(std::vector<int>& bit_stream, std::vector<int>& window, bool &checksync, std::vector<int>& previous_samples);

void getSamples(std::vector<real>& input_signal, int SPS, std::vector<real>& samples, std::vector<int>& indices, std::vector<real>& previous_samples);

void manchesterDecoding(std::vector<real>& input_signal, std::vector<int>& decoded_bits, real &previous_value);

void differentialDecoding(std::vector<int>& input_bits, std::vector<int>& output_bits, int &state);

#endif
