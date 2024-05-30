// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::plugins(openmp)]]

#include <RcppArmadillo.h>
#include <RcppParallel.h>
using namespace Rcpp;
using namespace RcppParallel;
//' Matrix Multiplier Worker for Parallel Matrix Multiplication
//'
//' This worker class is used for parallel computation of the matrix product `X'X`
//' where `X'` is the transpose of `X`. The class uses the RcppParallel package to
//' leverage multicore processing, improving performance for large matrices.
//'
//' @section Fields:
//' \itemize{
//'   \item \code{X}: Constant reference to an `arma::mat` representing the input matrix.
//'   \item \code{XtX}: Reference to an `arma::mat` where the result `X'X` is stored.
//' }
//'
//' @section Methods:
//' \describe{
//'   \item{Constructor}{Initializes the class with the input matrix and the output matrix references.}
//'   \item{operator()}{Performs the matrix multiplication operation over a specified range of rows, to be used by RcppParallel.}
//' }
//'
//' @details
//' The operator() method is designed to be called by RcppParallel, and it divides the task of computing
//' the matrix multiplication across multiple threads. Each thread computes a portion of the resulting
//' matrix, specifically, the contributions to `XtX` from rows `begin` to `end` of matrix `X`.
//'
//' @examples
//' \dontrun{
//' library(Rcpp)
//' library(RcppParallel)
//' sourceCpp("path/to/MatrixMultiplier.cpp") // Ensure this path points to the file containing the class
//'
//' # Create a random matrix
//' X <- matrix(rnorm(100 * 10), ncol = 10)
//' XtX <- matrix(0, ncol = 10, nrow = 10)
//'
//' # Create and use the worker
//' worker <- MatrixMultiplier(X, XtX)
//' parallelFor(0, nrow(X), worker)
//'
//' print(XtX) # This should print the result of X'X
//' }
//'
//' @importFrom Rcpp sourceCpp
//' @importFrom RcppParallel parallelFor
//' @export
//'
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

//' Ridge Regression with Intercept
//'
//' Performs ridge regression on a given set of predictors and a response variable. 
//' This function adds an intercept to the model by appending a column of ones to 
//' the predictor matrix `X`. It computes the ridge regression coefficients by 
//' penalizing the magnitude of the coefficients through a regularization term `lambda`.
//' The intercept term is not penalized.
//'
//' @param X A numeric matrix of predictor variables where each row is an observation 
//'          and each column is a predictor.
//' @param y A numeric vector of the response variable corresponding to each observation.
//' @param lambda A numeric value specifying the regularization strength (lambda >= 0).
//'
//' @return An R list containing:
//' \itemize{
//'   \item \code{error}: The mean squared error of the model residuals.
//'   \item \code{coefficients}: A numeric vector of the estimated coefficients, including the intercept.
//' }
//'
//' @details
//' The regularization term is added to the diagonal of the cross-product matrix of the
//' predictors (including the intercept), except for the first diagonal element which corresponds
//' to the intercept. This adjustment ensures that the intercept is not shrunk towards zero.
//' The solution to the ridge regression problem is computed using the matrix inversion lemma,
//' which ensures numerical stability and efficiency.
//'
//' @examples
//' \dontrun{
//' library(Rcpp)
//' sourceCpp("path/to/RidgeReg.cpp") // Ensure this path points to the file containing the function
//'
//' # Simulate some data
//' set.seed(123)
//' X <- matrix(rnorm(100 * 10), ncol = 10)
//' y <- X %*% rnorm(10) + rnorm(100)
//' lambda <- 0.5
//'
//' # Perform ridge regression
//' results <- RidgeReg(X, y, lambda)
//' print(results$error)
//' print(results$coefficients)
//' }
//'
//' @importFrom Rcpp sourceCpp
//' @export
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
//' Parallel Ridge Regression with Intercept
//'
//' Computes the coefficients of a ridge regression model using parallel computing
//' to handle the matrix multiplications. The function includes an intercept in the
//' model by appending a column of ones to the matrix `X`. It uses parallelization
//' to efficiently compute the cross-product matrix `X'X`, and it applies ridge
//' regularization by adding a lambda penalty to the diagonal elements of the matrix,
//' except for the intercept term.
//'
//' @param X A numeric matrix of predictor variables, where each row represents an 
//'          observation and each column a predictor.
//' @param y A numeric vector of the response variable corresponding to each observation.
//' @param lambda A double specifying the strength of the regularization (lambda >= 0).
//'
//' @return An R list containing:
//' \itemize{
//'   \item \code{error}: The mean squared error of the model residuals.
//'   \item \code{coefficients}: A numeric vector of the estimated coefficients, including the intercept.
//' }
//'
//' @details
//' The function utilizes the `RcppParallel` package for parallel processing. It employs
//' the `MatrixMultiplier` worker class to perform the multiplication of `X'X` in parallel,
//' speeding up the computation significantly, especially for large datasets. The intercept
//' is not regularized, consistent with standard ridge regression practices.
//'
//' @examples
//' \dontrun{
//' library(Rcpp)
//' library(RcppParallel)
//' sourceCpp("path/to/RidgeRegPar.cpp") // Ensure this path points to the file containing the function
//'
//' # Simulate some data
//' set.seed(123)
//' X <- matrix(rnorm(100 * 10), ncol = 10)
//' y <- X %*% rnorm(10) + rnorm(100)
//' lambda <- 0.5
//'
//' # Perform parallel ridge regression
//' results <- RidgeRegPar(X, y, lambda)
//' print(results$error)
//' print(results$coefficients)
//' }
//'
//' @importFrom Rcpp sourceCpp
//' @importFrom RcppParallel parallelFor
//' @export
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