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
  const int n_folds;
  std::vector<List>& results;
  
  RidgeCVWorker(const arma::mat& X, const arma::vec& y, const std::vector<double>& lambda_values, int n_folds, std::vector<List>& results)
    : X(X), y(y), lambda_values(lambda_values), n_folds(n_folds), results(results) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      double lambda = lambda_values[i];
      double cv_error = 0.0;
      
      // Perform k-fold cross-validation
      for (int fold = 0; fold < n_folds; fold++) {
        arma::uvec test_indices = arma::regspace<arma::uvec>(fold, n_folds, X.n_rows - 1);
        arma::uvec train_indices = arma::find(arma::linspace<arma::uvec>(0, X.n_rows - 1, X.n_rows) != test_indices);
        
        arma::mat X_train = X.rows(train_indices);
        arma::vec y_train = y.elem(train_indices);
        arma::mat X_test = X.rows(test_indices);
        arma::vec y_test = y.elem(test_indices);
        
        List result = performRidgeRegression(X_train, y_train, lambda);
        arma::colvec coef = as<arma::colvec>(result["coefficients"]);
        
        // Add intercept to the test set
        arma::mat X_test_with_intercept = arma::join_horiz(arma::ones<arma::vec>(X_test.n_rows), X_test);
        arma::vec y_pred = X_test_with_intercept * coef;
        cv_error += arma::mean(arma::square(y_test - y_pred));
      }
      
      // Store the average CV error
      cv_error /= n_folds;
      results[i] = List::create(Named("error") = cv_error, Named("lambda") = lambda);
    }
  }
};

// Rcpp export to perform parallel CV
// [[Rcpp::export]]
List parallelRidgeCV(const arma::mat& X, const arma::vec& y, std::vector<double> lambda_values, int n_folds) {
  std::vector<List> results(lambda_values.size());
  RidgeCVWorker worker(X, y, lambda_values, n_folds, results);
  
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
  List optimal_result = performRidgeRegression(X, y, optimal_lambda);
  arma::colvec optimal_coefs = as<arma::colvec>(optimal_result["coefficients"]);
  
  return List::create(Named("optimal_lambda") = optimal_lambda, Named("coefficients") = optimal_coefs);
}