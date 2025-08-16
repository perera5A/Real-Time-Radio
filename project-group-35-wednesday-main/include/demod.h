#ifndef DY4_DEMOD_H
#define DY4_DEMOD_H


#include <iostream>
#include <vector>
#include <complex>
#include <cmath>


void ownfmDemod(std::vector<real> I, std::vector<real> Q, real &prev_I, real &prev_Q, std::vector<real> &fm_demod);


#endif