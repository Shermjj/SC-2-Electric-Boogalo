// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::plugins(openmp)]]

#include <RcppArmadillo.h>
#include <RcppParallel.h>
using namespace Rcpp;
using namespace RcppParallel;

struct MatrixMultiplier : public Worker {
  // Inputs
  const arma::mat& X;
  arma::mat& XtX;
  
  // Constructor
  MatrixMultiplier(const arma::mat& X, arma::mat& XtX) : X(X), XtX(XtX) {}
  
  // Parallel operator
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      for (size_t j = 0; j < X.n_cols; ++j) {
        for (size_t k = 0; k < X.n_rows; ++k) {
          XtX(i, j) += X(k, i) * X(k, j);
        }
      }
    }
  }
};

// Function to perform ridge regression and compute CV error
// [[Rcpp::export]]
List RidgeReg(const arma::mat& X, const arma::vec& y, double lambda) {
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

// [[Rcpp::export]]
List RidgeRegPar(const arma::mat& X, const arma::vec& y, double lambda) {
  int n = X.n_rows;
  int k = X.n_cols + 1;
  
  // Add a column of ones to X for the intercept
  arma::mat X_with_intercept = arma::join_horiz(arma::ones<arma::vec>(n), X);
  
  // Prepare to calculate XtX in parallel
  arma::mat XtX = arma::zeros<arma::mat>(k, k);
  MatrixMultiplier multiplier(X_with_intercept, XtX);
  parallelFor(0, k, multiplier);
  
  // Add the ridge penalty
  arma::mat ridgePenalty = lambda * arma::eye<arma::mat>(k, k);
  ridgePenalty(0, 0) = 0; // Do not penalize the intercept term
  arma::mat XtX_ridge = XtX + ridgePenalty;
  
  // Solve for coefficients
  arma::colvec coef = arma::solve(XtX_ridge, arma::trans(X_with_intercept) * y);
  arma::colvec resid = y - X_with_intercept * coef;
  double sig2 = arma::as_scalar(arma::trans(resid) * resid / n);
  
  return List::create(Named("error") = sig2, Named("coefficients") = coef);
}