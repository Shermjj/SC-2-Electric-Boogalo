#include "RidgeReg.h"
#include "FeatureTransform.h"
#include "Test_Delete.h"
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;

// [[Rcpp::export]]
List parallel_ridge_cross_validation2(NumericMatrix x_vars,
                                     NumericVector y_var,
                                     NumericVector time_counter,
                                     double daily_period,
                                     double annual_period,
                                     int max_K,
                                     std::vector<double> lambda_values,
                                     int n_folds) {
  int n = time_counter.size();
  int n_vars = x_vars.ncol();
  int num_lambdas = lambda_values.size();
  NumericMatrix mse_results(max_K, num_lambdas);
  
  // Generate Fourier terms once for the maximum K
  NumericMatrix daily_terms = GenFT(time_counter, max_K, daily_period);
  NumericMatrix annual_terms = GenFT(time_counter, max_K, annual_period);
  
  // Store the lowest MSE as a very large number
  double min_mse = std::numeric_limits<double>::max();
  // Store best lambda and K values
  int best_K = 0;
  double best_lambda = 0.0;
  
  // Loop over K values
  for (int K = 1; K <= max_K; ++K) {
    
    // Combine daily terms, annual terms, and additional variables into a single matrix
    arma::mat X(n, 4 * K + n_vars);
    for (int j = 0; j < 2 * K; ++j) {
      X.col(j) = convertAndProcess(daily_terms).col(j);
      X.col(j + 2 * K) = convertAndProcess(annual_terms).col(j);
    }
    for (int j = 0; j < n_vars; ++j) {
      X.col(4 * K + j) = convertAndProcess(x_vars).col(j);
    }
    
    // Create folds
    IntegerVector folds = sample(n_folds, n, true);
    
    // Loop over lambda values
    for (int lambda_index = 0; lambda_index < num_lambdas; ++lambda_index) {
      double lambda = lambda_values[lambda_index];
      double total_mse = 0.0;
      
      // Preallocate matrices for train and test data
      arma::mat X_train, X_test;
      arma::vec y_train, y_test;
      
#pragma omp parallel for reduction(+:total_mse) schedule(dynamic)
      for (int fold = 1; fold <= n_folds; ++fold) {
        IntegerVector test_indices = which4(folds, fold, true);
        IntegerVector train_indices = which4(folds, fold, false);
        
        X_train.set_size(train_indices.size(), X.n_cols);
        y_train.set_size(train_indices.size());
        
        X_test.set_size(test_indices.size(), X.n_cols);
        y_test.set_size(test_indices.size());
        
        // Subset training and testing data
        for (int i = 0; i < train_indices.size(); ++i) {
          X_train.row(i) = X.row(train_indices[i]);
          y_train[i] = y_var[train_indices[i]];
        }
        for (int i = 0; i < test_indices.size(); ++i) {
          X_test.row(i) = X.row(test_indices[i]);
          y_test[i] = y_var[test_indices[i]];
        }
        
        // Train model and predict
        List model = RidgeRegPar(X_train, y_train, lambda);
        arma::colvec coef = Rcpp::as<arma::colvec>(model["coefficients"]);
        arma::vec predictions = makePredictions(X_test, coef);
        // Calculate MSE for this fold
        double fold_mse = arma::mean(arma::square(predictions - y_test));
        total_mse += fold_mse;
      }
      
      // Average MSE over all folds for this lambda and K
      double avg_mse = total_mse / n_folds;
      mse_results(K - 1, lambda_index) = avg_mse;
      if (avg_mse < min_mse) {
        min_mse = avg_mse;
        best_K = K;
        best_lambda = lambda;
      }
    }
  }
  
  // Train the final model using the best K and lambda values
  arma::mat X_final(n, 4 * best_K + n_vars);
  for (int j = 0; j < 2 * best_K; ++j) {
    X_final.col(j) = convertAndProcess(daily_terms).col(j);
    X_final.col(j + 2 * best_K) = convertAndProcess(annual_terms).col(j);
  }
  for (int j = 0; j < n_vars; ++j) {
    X_final.col(4 * best_K + j) = convertAndProcess(x_vars).col(j);
  }
  
  List final_model = RidgeRegPar(X_final, y_var, best_lambda);
  
  return List::create(Named("best_K") = best_K,
                      Named("best_lambda") = best_lambda,
                      Named("mse_results") = mse_results,
                      Named("final_model") = final_model);
}


