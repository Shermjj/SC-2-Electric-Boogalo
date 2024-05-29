// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
arma::mat rbf_kernel(const arma::mat& X, double l, double sigma) {
  arma::mat K = arma::zeros(X.n_rows, X.n_rows);
  for (unsigned i = 0; i < X.n_rows; ++i) {
    for (unsigned j = 0; j < X.n_rows; ++j) {
      double dist = arma::norm(X.row(i) - X.row(j), 2);
      K(i, j) = std::pow(sigma, 2) * std::exp(-std::pow(dist, 2) / (2 * std::pow(l, 2)));
    }
  }
  return K;
}