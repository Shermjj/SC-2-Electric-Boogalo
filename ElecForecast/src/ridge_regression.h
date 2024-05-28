// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::plugins(openmp)]]

#include <RcppArmadillo.h>
#include <RcppParallel.h>
using namespace Rcpp;
using namespace RcppParallel;

// Function to perform ridge regression and compute CV error
double performRidgeRegression(const arma::mat& X, const arma::vec& y, double lambda) {
  int n = X.n_rows, k = X.n_cols;
  arma::mat XtX = arma::trans(X) * X;
  arma::mat ridgePenalty = lambda * arma::eye<arma::mat>(k, k);
  arma::mat XtX_ridge = XtX + ridgePenalty;
  
  arma::colvec coef = arma::solve(XtX_ridge, arma::trans(X) * y);
  arma::colvec resid = y - X * coef;
  double sig2 = arma::as_scalar(arma::trans(resid) * resid / (n - k));
  return sig2; // Using MSE as the error metric
}

// Worker for parallel computation of CV errors
struct RidgeCVWorker : public Worker {
  const arma::mat& X;
  const arma::vec& y;
  const std::vector<double>& lambda_values;
  RVector<double> errors;
  
  RidgeCVWorker(const arma::mat& X, const arma::vec& y, const std::vector<double>& lambda_values, NumericVector errors)
    : X(X), y(y), lambda_values(lambda_values), errors(errors) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      double lambda = lambda_values[i];
      double error = performRidgeRegression(X, y, lambda);
      errors[i] = error;
    }
  }
};

// Rcpp export to perform parallel CV
// [[Rcpp::export]]
NumericVector parallelRidgeCV(const arma::mat& X, const arma::vec& y, std::vector<double> lambda_values) {
  NumericVector errors(lambda_values.size());
  RidgeCVWorker worker(X, y, lambda_values, errors);
  
  parallelFor(0, lambda_values.size(), worker);
  
  return errors;
}