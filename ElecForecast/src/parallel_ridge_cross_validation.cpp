#include <Rcpp.h>
#ifdef _OPENMP
  #include <omp.h>
#endif
#include "generate_fourier_terms.h"
#include "ridge_regression.h"
using namespace Rcpp;

// Cross-validation function
// [[Rcpp::export]]
NumericVector parallel_ridge_cross_validation(NumericMatrix x_vars,
                                              NumericVector y_var,
                                              NumericVector time_counter,
                                              double daily_period,
                                              double annual_period,
                                              int max_K,
                                              std::vector<double> lambda_values,
                                              int n_folds, 
                                              double alpha = 0) {
  NumericVector mse(max_K);
  int n = time_counter.size();
  int n_vars = x_vars.ncol();
  int num_lambdas = lambda_values.size();
  
  // Generate Fourier terms once for the maximum K
  NumericMatrix daily_terms = generate_fourier_terms(time_counter, max_K, daily_period);
  NumericMatrix annual_terms = generate_fourier_terms(time_counter, max_K, annual_period);
  
  // Perform cross-validation for each K
  #pragma omp parallel for
  for (int K = 1; K <= max_K; ++K) {
    
    // Combine daily terms, annual terms, and additional variables into a single matrix
    NumericMatrix x(n, 4 * K + n_vars);
    // Add daily and annual Fourier terms
    for (int j = 0; j < 2 * K; ++j) {
      x(_, j) = daily_terms(_, j);
      x(_, j + 2 * K) = annual_terms(_, j);
    }
    // Add additional variables
    for (int j = 0; j < n_vars; ++j) {
      x(_, 4 * K + j) = x_vars(_, j);
    }
    
    // Create folds
    IntegerVector folds = sample(n_folds, n, true);
    double total_mse = 0.0;
    
    for (int fold = 1; fold <= n_folds; ++fold) {
      IntegerVector test_indices = which(folds == fold);
      IntegerVector train_indices = which(folds != fold);
      
      NumericMatrix x_train(train_indices.size(), x.ncol());
      NumericVector y_train(train_indices.size());
      
      for (int i = 0; i < train_indices.size(); ++i) {
        for (int j = 0; j < x.ncol(); ++j) {
          x_train(i, j) = x(train_indices[i], j);
        }
        y_train[i] = y_var(train_indices[i], 0);
      }
      
      NumericMatrix x_test(test_indices.size(), x.ncol());
      NumericVector y_test(test_indices.size());
      
      for (int i = 0; i < test_indices.size(); ++i) {
        for (int j = 0; j < x.ncol(); ++j) {
          x_test(i, j) = x(test_indices[i], j);
        }
        y_test[i] = y_var(test_indices[i], 0);
      }
      
      // Loop over lambda values
      for (int i = 0; i < num_lambdas; ++i) {
        double lambda_i = lambda_values[i];
        List model = performRidgeRegression(x_train, y_train, lambda_i);
        NumericVector predictions = x_test * model["coefficients"];
      }
      
      double fold_mse = 0.0;
      for (int i = 0; i < predictions.size(); ++i) {
        fold_mse += pow(predictions[i] - y_test[i], 2);
      }
      fold_mse /= predictions.size();
      
      total_mse += fold_mse;
    }
    
    mse[K - 1] = total_mse / n_folds;
  }
  
  return mse;
}

