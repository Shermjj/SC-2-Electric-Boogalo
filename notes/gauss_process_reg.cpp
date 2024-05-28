#include <RcppArmadillo.h>

// Rest of your code goes here
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]

List gauss_process_regCpp(arma::mat X, arma::vec y, arma::mat Xnew, double sigma2, double l) {
  int n = X.n_rows;
  int nnew = Xnew.n_rows;
  mat K = exp(-1/(2*l*l)*square(X.each_row() - X.t()));
  K.diag() += sigma2;
  mat L = chol(K);
  vec alpha = solve(trimatl(L), solve(trimatu(L), y));
  mat Knew = exp(-1/(2*l*l)*square(X.each_row() - Xnew.t()));
  vec mu = Knew.t() * alpha;
  mat v = solve(trimatu(L), Knew);
  mat cov = exp(-1/(2*l*l)*square(Xnew.each_row() - Xnew.t())) - v.t() * v;
  return List::create(Named("mu") = mu, Named("cov") = cov);
}

