#ifndef ELECFORECAST_GENFT_H
#define ELECFORECAST_GENFT_H
#include <Rcpp.h>
using namespace Rcpp;

NumericMatrix GenFT(NumericVector time_counter, int K, double period);

#endif