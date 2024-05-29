// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

#include <RcppArmadillo.h>
#include <omp.h>

using namespace Rcpp;

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

// Function to perform k-fold cross-validation
// [[Rcpp::export]]
List parallelRidgeCV(const arma::mat& X, const arma::vec& y, std::vector<double> lambda_values, int n_folds) {
  int n = X.n_rows;
  int num_lambdas = lambda_values.size();
  std::vector<double> cv_errors(num_lambdas, 0.0);
  
#pragma omp parallel for
  for (int i = 0; i < num_lambdas; i++) {
    double lambda = lambda_values[i];
    double cv_error = 0.0;
    
    // Create folds
    IntegerVector folds = sample(n_folds, n, true);
    
    // Perform k-fold cross-validation
    for (int fold = 1; fold <= n_folds; ++fold) {
      IntegerVector test_indices = which(folds == fold);
      IntegerVector train_indices = which(folds != fold);
      
      NumericMatrix x_train(train_indices.size(), x.ncol());
      NumericVector y_train(train_indices.size());
      NumericMatrix x_test(test_indices.size(), x.ncol());
      NumericVector y_test(test_indices.size());
      
      List result = performRidgeRegression(X_train, y_train, lambda);
      arma::colvec coef = as<arma::colvec>(result["coefficients"]);
      
      // Add intercept to the test set
      arma::mat x_test_with_intercept = arma::join_horiz(arma::ones<arma::vec>(x_test.n), x_test);
      arma::vec y_pred = x_test_with_intercept * coef;
      cv_error += arma::mean(arma::square(y_test - y_pred));
    }
    
    // Store the average CV error
    cv_errors[i] = cv_error / n_folds;
  }
  
  // Find the index of the minimum error
  int min_error_index = std::distance(cv_errors.begin(), std::min_element(cv_errors.begin(), cv_errors.end()));
  double optimal_lambda = lambda_values[min_error_index];
  
  // Get coefficients for the optimal lambda
  List optimal_result = performRidgeRegression(X, y, optimal_lambda);
  arma::colvec optimal_coefs = as<arma::colvec>(optimal_result["coefficients"]);
  
  return List::create(Named("optimal_lambda") = optimal_lambda, Named("coefficients") = optimal_coefs);
}
