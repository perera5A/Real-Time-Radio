#include "dy4.h"
#include "RDSsupport.h"
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <unordered_map>
#include <iomanip>
#include <sstream>
#include <bitset>
//normalized_I_RRC = RRC_dataI / max(np.abs(RRC_dataI))

void normalize(std::vector<real>& data, std::vector<real> &data2, std::vector<real>& normalized_data){
    if(data.empty()) return;
    if(data2.empty()) return;

    normalized_data.resize(data2.size());

    real max_abs = 0;
    for (real val : data2) {
        if (std::abs(val) > max_abs) {
            max_abs = std::abs(val);
        }
    }

    for (size_t i = 0; i < data.size(); ++i) {
        normalized_data[i] = data[i] / max_abs;
    }
}

void getSamples(std::vector<real>& input_signal, int SPS, std::vector<real>& samples, std::vector<int>& indices, std::vector<real>& previous_samples) {
    samples.clear();
    indices.clear();
    int offset = 0;
    int block_count = 0;
    
    // Add previous_samples to the start of the first block if it's not empty
    size_t start_index = 0;
    if (!previous_samples.empty()) {
        start_index = previous_samples.size();
        // Prepend previous_samples to the beginning of the input_signal
        input_signal.insert(input_signal.begin(), previous_samples.begin(), previous_samples.end());
    }

    // Process each block of SPS size
    for (size_t i = start_index + SPS / 2; i < input_signal.size(); i += SPS) {
        // Adjust offset based on block processing logic
        if (block_count % 1000 == 0) {
            if (std::abs(input_signal[i + offset + 1]) > std::abs(input_signal[i + offset])) {
                offset = std::min(SPS / 2, offset + 1);
            } else if (std::abs(input_signal[i + offset - 1]) > std::abs(input_signal[i + offset])) {
                offset = std::max(0, offset - 1);
            }
        }

        // Process the current block and add the middle sample to samples
        if (i + offset < input_signal.size()) {
            samples.push_back(input_signal[i + offset]);
            indices.push_back(i + offset);
        }

        block_count++;
    }

    // If there's less than SPS values left in the final block, update previous_samples
    if (input_signal.size() % SPS != 0) {
        size_t last_block_start = input_signal.size() - (input_signal.size() % SPS);
        previous_samples.clear();
        previous_samples.insert(previous_samples.begin(), input_signal.begin() + last_block_start, input_signal.end());
    }
}


void manchesterDecoding(std::vector<real>& input_signal, std::vector<int>& decoded_bits, real &previous_value) {
    decoded_bits.reserve(input_signal.size() / 2);

    for (size_t i = 0; i < input_signal.size() - 1; i += 2) {
        real first_sample;

        if (previous_value != 0) {
            // Use the passed previous value for the first sample if it is provided
            first_sample = previous_value;
            previous_value = 0;  // Reset previous_value after using it
        } else {
            // Otherwise, use the current value from the input signal
            first_sample = input_signal[i];
        }

        real second_sample = input_signal[i + 1];

        // Manchester decoding logic
        if (first_sample > 0 && second_sample < 0) {
            decoded_bits.push_back(1);
        } else if (first_sample < 0 && second_sample > 0) {
            decoded_bits.push_back(0);
        } else {
            decoded_bits.push_back(0);
        }
    }

    // If there's an odd number of samples, assign the last sample to previous_value
    if (input_signal.size() % 2 != 0) {
        previous_value = input_signal.back();
    }
}

void differentialDecoding(std::vector<int>& input_bits, std::vector<int>& output_bits, int &state) {
    output_bits.resize(input_bits.size()); // resize output
    int previous_bit = state;

    for (size_t i = 0; i < input_bits.size(); ++i) {
        output_bits[i] = previous_bit ^ input_bits[i];
        previous_bit = input_bits[i];
    }
    state = previous_bit;
}

std::vector<std::vector<int>> parityMatrix = {
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1,    1, 1, 0, 1, 1, 0, 0, 1, 1, 1},
    {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1,    1, 1, 1, 0, 1, 1, 0, 0, 1, 1},
    {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0,    0, 0, 1, 0, 1, 1, 1, 1, 1, 0},
    {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,    1, 1, 0, 0, 1, 1, 1, 0, 0, 0},
    {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,    0, 1, 1, 0, 0, 1, 1, 1, 0, 0},
    {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1,    1, 1, 1, 0, 1, 0, 1, 0, 0, 1},
    {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0,    0, 0, 1, 0, 1, 1, 0, 0, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0,    1, 1, 0, 0, 1, 1, 1, 1, 1, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1,    0, 1, 1, 0, 0, 1, 1, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1,    1, 0, 1, 1, 0, 0, 1, 1, 1, 1}
};

std::map<std::string, std::vector<int>> syndromes = {
    {"A",  {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}},
    {"B",  {1, 1, 1, 1, 0, 1, 0, 1, 0, 0}},
    {"C",  {1, 0, 0, 1, 0, 1, 1, 1, 0, 0}},
    {"C'", {1, 1, 1, 1, 0, 0, 1, 1, 0, 0}},
    {"D",  {1, 0, 0, 1, 0, 1, 1, 0, 0, 0}}
};
                                              
std::vector<int> compute_syndrome(const std::vector<int>& window) {
    std::vector<int> syndrome;
    for (const auto& column : parityMatrix) {
        int xor_result = 0;
        for (size_t i = 0; i < column.size(); i++) {
            xor_result ^= (window[i] & column[i]);
        }
        syndrome.push_back(xor_result);
    }
    return syndrome;
}

std::string get_syndrome_key(const std::vector<int>& syndrome) {
    for (const auto& pair : syndromes) {
        if (pair.second == syndrome) {
            return pair.first;
        }
    }
    return "";
}

std::string getPIcode(const std::vector<int>& message_bits) {
    int pi_code_int = 0;
    for (size_t i = 0; i < message_bits.size(); ++i) {
        pi_code_int = (pi_code_int << 1) | message_bits[i]; // Convert binary to integer
    }

    std::stringstream ss;
    ss << std::uppercase << std::hex << std::setw(4) << std::setfill('0') << pi_code_int; // Convert to 4-digit hex
    return ss.str();
}

std::tuple<int, char, int, int> getGroupTypePlusPTY(const std::vector<int>& message_bits) {
    int number = 0;
    for (int i = 0; i < 4; ++i) {
        number = (number << 1) | message_bits[i]; // Convert first 4 bits to integer
    }

    char letter = (message_bits[4] == 0) ? 'A' : 'B';
    int TP = message_bits[5];

    int PTY = 0;
    for (int i = 6; i < 11; ++i) {
        PTY = (PTY << 1) | message_bits[i]; // Convert bits 6-10 to integer
    }

    return std::make_tuple(number, letter, TP, PTY);
}

std::string getProgramInfo(const std::vector<int>& message_bits) {
    int first_char = 0, second_char = 0;
    
    for (int i = 0; i < 8; ++i) {
        first_char = (first_char << 1) | message_bits[i];
    }
    for (int i = 8; i < 16; ++i) {
        second_char = (second_char << 1) | message_bits[i];
    }

    return std::string(1, static_cast<char>(first_char)) + static_cast<char>(second_char);
}

std::string getRadioText(const std::vector<int>& message_bits) {
    return getProgramInfo(message_bits); // Since it performs the same operation
}

void parity_check(std::vector<int>& bit_stream, std::vector<int>& window, bool &checksync, std::vector<int>& previous_samples) {
    bool synchronized = checksync;
    size_t i = 0;

    // Add previous_samples to the start of bit_stream if they exist
    if (!previous_samples.empty()) {
        bit_stream.insert(bit_stream.begin(), previous_samples.begin(), previous_samples.end());
    }

    std::vector<int> message(4, 0);
    std::unordered_map<int, std::string> radio_text;
    std::string program_info;
    size_t count = 0;
    int address = 0;

    while (!synchronized && i <= bit_stream.size() - 104) {  // 4 blocks * 26 bits = 104
        std::vector<std::string> sequence_keys;
        for (int j = 0; j < 4; ++j) {  // Try to get 4 syndromes 26 bits apart
            std::vector<int> window(bit_stream.begin() + i + j * 26, bit_stream.begin() + i + (j + 1) * 26);
            std::vector<int> syndrome = compute_syndrome(window);
            std::string key = get_syndrome_key(syndrome);
            if (!key.empty()) {
                std::cerr << "Processing block " << j + 1 << " at index " << i + j * 26 << ": ";
                for (int bit : window) std::cout << bit;
                std::cerr << "\nSyndrome for block " << j + 1 << " at index " << i + j * 26 << ": ";
                for (int s : syndrome) std::cout << s;
                std::cerr << ", Key: " << key << "\n";
                sequence_keys.push_back(key);
            } else {
                break;
            }
        }

        if (sequence_keys.size() == 4) {
            if (sequence_keys[0] == "A" && sequence_keys[1] == "B" && 
                (sequence_keys[2] == "C" || sequence_keys[2] == "C'") && sequence_keys[3] == "D") {
                synchronized = true;
                checksync = true;
                std::cerr << "Synchronized at index " << i << ", sequence: ";
                for (const auto& key : sequence_keys) std::cout << key << " ";
                std::cerr << "\n";
                i += 104;  // Move past synchronized group
                break;
            }
        }
        i += 1;
    }

    if (synchronized) {
        while (i <= bit_stream.size() - 26) {
            std::vector<int> window(bit_stream.begin() + i, bit_stream.begin() + i + 26);
            std::vector<int> syndrome = compute_syndrome(window);
            std::string key = get_syndrome_key(syndrome);
            window = std::vector<int>(window.begin(), window.begin() + 16);

            if (key == "A") {
                message[0] = std::stoi(getPIcode(window), nullptr, 16);  // Assuming getPIcode returns hex string
            } else if (key == "B") {
                auto groupType = getGroupTypePlusPTY(window);
                message[1] = std::get<0>(groupType);  // Assuming you want to store the number
                if (std::get<0>(groupType) == 2) {
                    address = std::stoi(std::string(window.end() - 5, window.end()));
                }
            } else if (key == "C") {
                if (message[1] == 2) {
                    radio_text[address] = getRadioText(window);  // Assuming getRadioText returns a string
                }
            } else if (key == "D") {
                if (message[1] == 2) {
                    radio_text[address] += getRadioText(window);  // Assuming getRadioText returns a string
                } else if (message[1] == 0) {
                    program_info += getProgramInfo(window);  // Assuming getProgramInfo returns a string
                }
            }

            count += 1;
            if (count % 4 == 0) {
                std::string all_radio_texts;
                for (const auto& texts : radio_text) {
                    all_radio_texts += texts.second;
                }

                std::cerr << "Station PI: " << message[0] << ", Station Group Type: " << message[1] << "\n";
                std::cerr << "Program Info: " << program_info << "\n";
                std::cerr << all_radio_texts << "\n";
            }

            i += 26;  // Process next block
        }
    }

    // Update previous_samples with the remaining bits
    if (bit_stream.size() - i > 0) {
        previous_samples.clear();
        previous_samples.insert(previous_samples.begin(), bit_stream.begin() + i, bit_stream.end());
    }
}

/*
int main() {
    // Example bit stream to test the ParityCheck function
    std::vector<int> checkword = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0};
    std::vector<int> syndrome = compute_syndrome(checkword);
    for (int bit : checkword) std::cout << bit;
    std::cout << "\nSyndrome: ";
    for (int s : syndrome) std::cout << s;
    std::cout << "\n";

    std::string key = get_syndrome_key(syndrome);
    std::cout << "Syndrome Key: " << key << "\n";

    return 0;
}
*/

