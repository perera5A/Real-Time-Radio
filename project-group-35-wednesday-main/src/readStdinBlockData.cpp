#include "dy4.h"
#include "readStdinBlockData.h"

void readStdinBlockData(unsigned int num_samples, unsigned int block_id, std::vector<real> &block_data){

    std::vector<char> raw_data(num_samples);

    std::cin.read(reinterpret_cast<char*>(&raw_data[0]), num_samples*sizeof(char));

    for (int k = 0; k < int(num_samples); k++){
        block_data[k] = float(((unsigned char) raw_data[k] - 128) / 128.0);
    }
}