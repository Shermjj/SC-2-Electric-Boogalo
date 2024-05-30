#include "RidgeReg.h"
#include "FeatureTransform.h"
using namespace Rcpp;
IntegerVector which4(IntegerVector vec, int value, bool equality = true) {
  std::vector<int> indices;
  for (int i = 0; i < vec.size(); i++) {
    if ((vec[i] == value) == equality) {
      indices.push_back(i);
    }
  }
  return wrap(indices); // Convert std::vector to IntegerVector
}

arma::mat convertAndProcess(Rcpp::NumericMatrix x) {
  arma::mat y = Rcpp::as<arma::mat>(x);  // Explicit conversion
  // Now y can be used as an arma::mat, and you can perform your operations
  return y;
}

arma::vec makePredictions(const arma::mat& x_test, const arma::colvec& coefficients) {
  // Assuming x_test does not already have an intercept column
  int n_test = x_test.n_rows;
  arma::mat x_test_with_intercept = arma::join_horiz(arma::ones<arma::vec>(n_test), x_test);
  
  // Calculate predictions
  arma::vec predictions = x_test_with_intercept * coefficients;
  return predictions;
}

// [[Rcpp::export]]
List parallel_ridge_cross_validation(NumericMatrix x_vars,
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
    
    // Loop over lambda values
    for (int lambda_index = 0; lambda_index < num_lambdas; ++lambda_index) {
      double lambda = lambda_values[lambda_index];
      double total_mse = 0.0;
      
      for (int fold = 1; fold <= n_folds; ++fold) {
        IntegerVector test_indices = which4(folds, fold, true);
        IntegerVector train_indices = which4(folds, fold, false);
        
        NumericMatrix x_train(train_indices.size(), x.ncol());
        NumericVector y_train(train_indices.size());
        
        NumericMatrix x_test(test_indices.size(), x.ncol());
        NumericVector y_test(test_indices.size());
        
        // Subset training and testing data
        for (int i = 0; i < train_indices.size(); ++i) {
          x_train(i, _) = x(train_indices[i], _);
          y_train[i] = y_var[train_indices[i]];
        }
        for (int i = 0; i < test_indices.size(); ++i) {
          x_test(i, _) = x(test_indices[i], _);
          y_test[i] = y_var[test_indices[i]];
        }
        
        // Train model and predict
        List model = RidgeRegPar(convertAndProcess(x_train), y_train, lambda);
        arma::colvec coeffs = Rcpp::as<arma::colvec>(model["coefficients"]);
        arma::mat x_test_arma = Rcpp::as<arma::mat>(x_test);
        arma::vec predictions = makePredictions(x_test_arma, coeffs);
        // Calculate MSE for this fold
        double fold_mse = 0.0;
        for (int i = 0; i < predictions.n_elem; ++i) {
          fold_mse += pow(predictions[i] - y_test[i], 2);
        }
        fold_mse /= predictions.n_elem;
        
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
  NumericMatrix x_final(n, 4 * best_K + n_vars);
  for (int j = 0; j < 2 * best_K; ++j) {
    x_final(_, j) = daily_terms(_, j);
    x_final(_, j + 2 * best_K) = annual_terms(_, j);
  }
  for (int j = 0; j < n_vars; ++j) {
    x_final(_, 4 * best_K + j) = x_vars(_, j);
  }
  
  List final_model = RidgeRegPar(convertAndProcess(x_final), y_var, best_lambda);
  
  return List::create(Named("best_K") = best_K,
                      Named("best_lambda") = best_lambda,
                      Named("mse_results") = mse_results,
                      Named("final_model") = final_model);
}

// [[Rcpp::export]]
List predict_parallel_ridge_cv(List model,
                                        NumericMatrix x_test,
                                        NumericVector y_test,
                                        NumericVector time_counter,
                                        double daily_period,
                                        double annual_period) {
  int best_K = model["best_K"];
  double best_lambda = model["best_lambda"];
  List final_model = model["final_model"];
  
  // Generate Fourier terms for the test data
  int n_test = time_counter.size();
  NumericMatrix daily_terms = GenFT(time_counter, best_K, daily_period);
  NumericMatrix annual_terms = GenFT(time_counter, best_K, annual_period);
  
  // Combine daily terms, annual terms, and additional variables into a single matrix
  NumericMatrix x_final(n_test, 4 * best_K + x_test.ncol());
  for (int j = 0; j < 2 * best_K; ++j) {
    x_final(_, j) = daily_terms(_, j);
    x_final(_, j + 2 * best_K) = annual_terms(_, j);
  }
  for (int j = 0; j < x_test.ncol(); ++j) {
    x_final(_, 4 * best_K + j) = x_test(_, j);
  }
  
  // Make predictions
  arma::colvec coeffs = Rcpp::as<arma::colvec>(final_model["coefficients"]);
  arma::vec predictions = makePredictions(convertAndProcess(x_final), coeffs);
  
  // Calculate MSE
  arma::colvec resid = Rcpp::as<arma::colvec>(y_test) - predictions;
  double sig2 = arma::as_scalar(arma::trans(resid) * resid / n_test);
  
  return List::create(Named("error") = sig2, Named("predictions") = predictions);
}

