#ifndef ELECFORECAST_TEST_DELETE_H
#define ELECFORECAST_TEST_DELETE_H
#include <RcppArmadillo.h>

using namespace Rcpp;

IntegerVector which4(IntegerVector vec, int value, bool equality = true);

arma::mat convertAndProcess(Rcpp::NumericMatrix x);

arma::vec makePredictions(const arma::mat& x_test, const arma::colvec& coefficients);

#endif