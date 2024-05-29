#ifndef ELECFORECAST_RIDGEREG_H
#define ELECFORECAST_RIDGEREG_H
#include <RcppArmadillo.h>

using namespace Rcpp;

List RidgeReg(const arma::mat& X, const arma::vec& y, double lambda);

#endif