// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]}
// [[Rcpp::plugins(openmp)]]
#include <RcppArmadillo.h>
#include <RcppParallel.h>
using namespace Rcpp;
using namespace RcppParallel;
using namespace arma;

// Function to perform ridge regression
// [[Rcpp::export]]
List performRidgeRegression(const arma::mat& X, const arma::vec& y, double lambda) {
  int n = X.n_rows, k = X.n_cols;
  arma::mat XtX = arma::trans(X) * X;
  arma::mat ridgePenalty = lambda * arma::eye<arma::mat>(k, k);
  arma::mat XtX_ridge = XtX + ridgePenalty;
  
  arma::colvec coef = arma::solve(XtX_ridge, arma::trans(X) * y);
  arma::colvec resid = y - X * coef;
  double sig2 = arma::as_scalar(arma::trans(resid) * resid / n);
  
  return List::create(Named("error") = sig2, Named("coefficients") = coef);
}

// Worker for parallel computation of CV errors
struct RidgeCVWorker : public Worker {
  // Inputs
  const arma::mat& X;
  const arma::vec& y;
  const std::vector<double>& lambda_values;
  
  // Outputs
  std::vector<double>& errors;
  std::vector<arma::colvec>& coefficients;
  
  // Constructor
  RidgeCVWorker(const arma::mat& X, const arma::vec& y, const std::vector<double>& lambda_values, 
                std::vector<double>& errors, std::vector<arma::colvec>& coefficients)
    : X(X), y(y), lambda_values(lambda_values), errors(errors), coefficients(coefficients) {}
  
  // Operator overloading
  void operator()(std::size_t begin, std::size_t end) override {
    for (std::size_t i = begin; i < end; ++i) {
      List result = performRidgeRegression(X, y, lambda_values[i]);
      errors[i] = as<double>(result["error"]);
      coefficients[i] = as<arma::colvec>(result["coefficients"]);
    }
  }
};

// Rcpp export to perform parallel CV
// [[Rcpp::export]]
List parallelRidgeCV(const arma::mat& X, const arma::vec& y, const std::vector<double>& lambda_values) {
  int n_lambdas = lambda_values.size();
  std::vector<double> errors(n_lambdas);
  std::vector<arma::colvec> coefficients(n_lambdas);
  
  RidgeCVWorker worker(X, y, lambda_values, errors, coefficients);
  parallelFor(0, lambda_values.size(), worker, 100);  // Adjust grain size for better threading performance
  
  // Find the minimum error and corresponding lambda
  auto min_iter = std::min_element(errors.begin(), errors.end());
  int min_index = std::distance(errors.begin(), min_iter);
  
  return List::create(
    Named("optimal_lambda") = lambda_values[min_index],
                                           Named("coefficients") = coefficients[min_index]
  );
}

