// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::plugins(openmp)]]

#include <RcppArmadillo.h>
#include <RcppParallel.h>
using namespace Rcpp;
using namespace RcppParallel;

// Function to perform ridge regression and compute CV error
// [[Rcpp::export]]
List performRidgeRegression(const arma::mat& X, const arma::vec& y, double lambda) {
  int n = X.n_rows;
  int k = X.n_cols + 1; // adding one column for the intercept
  
  // Add a column of ones to X for the intercept
  arma::mat X_with_intercept = arma::join_horiz(arma::ones<arma::vec>(n), X);
  
  // Compute XtX and XtY
  arma::mat XtX = arma::trans(X_with_intercept) * X_with_intercept;
  arma::mat ridgePenalty = lambda * arma::eye<arma::mat>(k, k);
  ridgePenalty(0, 0) = 0; // Do not penalize the intercept term
  arma::mat XtX_ridge = XtX + ridgePenalty;
  
  arma::colvec coef = arma::solve(XtX_ridge, arma::trans(X_with_intercept) * y);
  arma::colvec resid = y - X_with_intercept * coef;
  double sig2 = arma::as_scalar(arma::trans(resid) * resid / n); 
  
  return List::create(_["error"] = sig2, _["coefficients"] = coef);
}

// Worker for parallel computation of CV errors
struct RidgeCVWorker : public Worker {
  const arma::mat& X;
  const arma::vec& y;
  const std::vector<double>& lambda_values;
  std::vector<List>& results;
  
  RidgeCVWorker(const arma::mat& X, const arma::vec& y, const std::vector<double>& lambda_values, std::vector<List>& results)
    : X(X), y(y), lambda_values(lambda_values), results(results) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      double lambda = lambda_values[i];
      List result = performRidgeRegression(X, y, lambda);
      results[i] = result;
    }
  }
};

// Rcpp export to perform parallel CV
// [[Rcpp::export]]
List parallelRidgeCV(const arma::mat& X, const arma::vec& y, std::vector<double> lambda_values) {
  std::vector<List> results(lambda_values.size());
  RidgeCVWorker worker(X, y, lambda_values, results);
  
  parallelFor(0, lambda_values.size(), worker);
  
  // Find the index of the minimum error
  int min_error_index = 0;
  double min_error = as<double>(results[0]["error"]);
  for (int i = 1; i < results.size(); ++i) {
    double current_error = as<double>(results[i]["error"]);
    if (current_error < min_error) {
      min_error = current_error;
      min_error_index = i;
    }
  }
  
  // Get the optimal lambda value
  double optimal_lambda = lambda_values[min_error_index];
  
  // Get coefficients for the optimal lambda
  arma::colvec optimal_coefs = as<arma::colvec>(results[min_error_index]["coefficients"]);
  
  return List::create(_["optimal_lambda"] = optimal_lambda, _["coefficients"] = optimal_coefs);
}
