library(forecast)
library(ElecForecast)
df_day <- readRDS("./data/df_day.RData")
df_halfhr <- readRDS("./data/df_halfhr.RData")
df_hr <- readRDS("./data/df_hr.RData")
autoplot(ts(df_day[,1]))

mod <- linReg(df_day)
         
summary(mod)

library(Rcpp)
library(RcppArmadillo)

Rcpp::sourceCpp(code='
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::export]]
List fastLmRidge(const arma::vec & y, const arma::mat & X, double lambda) {

    int n = X.n_rows, k = X.n_cols;
    arma::mat XtX = arma::trans(X) * X;
    arma::mat ridgePenalty = lambda * arma::eye<arma::mat>(k, k);
    arma::mat XtX_ridge = XtX + ridgePenalty;

    arma::colvec coef = arma::solve(XtX_ridge, arma::trans(X) * y);
    arma::colvec resid = y - X * coef;
    double sig2 = arma::as_scalar(arma::trans(resid) * resid / (n - k));
    arma::colvec stderrest = arma::sqrt(sig2 * arma::diagvec(arma::inv(XtX_ridge)));

    return List::create(Named("coefficients") = coef,
                        Named("stderr") = stderrest,
                        Named("lambda") = lambda);
}
')

X <- matrix(rnorm(100 * 10), ncol=10)
y <- rnorm(100)
lambda <- 0.5
op <- fastLmRidge(y,X,lambda)
op$coefficients
mod <- glmnet(X, y,alpha = 0, lambda  = lambda)
summary(mod)
sourceCpp(code = '
 // [[Rcpp::depends(RcppArmadillo)]]
 #include <RcppArmadillo.h>
 using namespace Rcpp;

 // [[Rcpp::export(name = "MMv_arma")]]
 arma::vec MMv_arma_I(arma::mat& A, arma::mat& B, arma::vec& y) {
   return A * B * y;
}', verbose = TRUE)

library(Rcpp)
sourceCpp(code = '
#include <unistd.h>
#include <Rcpp.h>

// [[Rcpp::export(wait_a_second)]]
bool wait_a_second(int sec)
{
 for(int ii = 0; ii < sec; ii++)
 { 
  sleep(1);
 }
 return 1;
}
')

system.time( wait_a_second(2) )[3]
sourceCpp(code = '
#include <unistd.h>
#include <Rcpp.h>

// [[Rcpp::plugins(openmp)]]

// [[Rcpp::export(wait_a_second_omp)]]
bool wait_a_second_omp(int sec, int ncores)
{

 #if defined(_OPENMP)
  #pragma omp parallel num_threads(ncores)
  #pragma omp for
 #endif
 for(int ii = 0; ii < sec; ii++)
 { 
  sleep(1);
 }
 
 return 1;

 }
')
library(Rcpp)
sourceCpp(code = '
#include <boost/math/special_functions/erf.hpp>
#include <Rcpp.h>
#include <RcppParallel.h>
using namespace Rcpp;

// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::plugins(openmp)]]

// [[Rcpp::export(erfOmp)]]
NumericVector erfOmp(NumericVector x, int ncores)
{

 size_t n = x.size();
 NumericVector out(n);
 RcppParallel::RVector<double> wo(out);
 RcppParallel::RVector<double> wx(x);
 
 #if defined(_OPENMP)
  #pragma omp parallel for num_threads(ncores)
 #endif
 for(size_t ii = 0; ii < n; ii++)
 {
  wo[ii] = boost::math::erf(wx[ii]);
 }
 
 return out;

 }
')

library(Rcpp)
library(RcppArmadillo)
library(RcppParallel)
Rcpp::sourceCpp(code='
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
  int n = X.n_rows, k = X.n_cols;
  arma::mat XtX = arma::trans(X) * X;
  arma::mat ridgePenalty = lambda * arma::eye<arma::mat>(k, k);
  arma::mat XtX_ridge = XtX + ridgePenalty;
  
  arma::colvec coef = arma::solve(XtX_ridge, arma::trans(X) * y);
  arma::colvec resid = y - X * coef;
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
}')

# Generate example data
set.seed(123)
X <- matrix(rnorm(100 * 10), ncol=10)
y <- rnorm(100)
lambda_values <- seq(0.01, 100, length.out = 100)

# Compute cross-validation errors for each lambda
cv_errors <- parallelRidgeCV(X, y, lambda_values)

# Identify the optimal lambda
optimal_lambda <- lambda_values[which.min(cv_errors)]
print(paste("Optimal Lambda:", optimal_lambda))


Rcpp::sourceCpp(code='
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
// [[Rcpp::export]]
List performRidgeRegressionParallel(const arma::mat& X, const arma::vec& y, double lambda) {
    int n = X.n_rows, k = X.n_cols;

    // Prepare to calculate XtX in parallel
    arma::mat XtX = arma::zeros<arma::mat>(k, k);
    MatrixMultiplier multiplier(X, XtX);
    parallelFor(0, k, multiplier);

    // Add the ridge penalty
    arma::mat ridgePenalty = lambda * arma::eye<arma::mat>(k, k);
    arma::mat XtX_ridge = XtX + ridgePenalty;

    // Solve for coefficients
    arma::colvec coef = arma::solve(XtX_ridge, arma::trans(X) * y);
    arma::colvec resid = y - X * coef;
    double sig2 = arma::as_scalar(arma::trans(resid) * resid / n);

    return List::create(Named("error") = sig2, Named("coefficients") = coef);
}')

X <- matrix(rnorm(100 * 10), ncol=10)
y <- rnorm(100)
lambda <- 0.5
performRidgeRegressionParallel(X,y, lambda)


# Generate example time-series data
set.seed(123)
X <- matrix(rnorm(200 * 10), ncol=10)  # 200 time periods
y <- rnorm(200)
lambda_values <- seq(0.1, 10, length.out = 50)

# Create training and testing indices for forward chaining
train_indices <- list()
test_indices <- list()
for (i in 1:10) {  # For example, 10 folds
  train_indices[[i]] <- 1:(10*i)
  test_indices[[i]] <- (10*i + 1):(10*i + 10)
}

# Compute cross-validation errors for each lambda
cv_errors <- parallelRidgeCVTimeSeries(X, y, lambda_values, train_indices, test_indices)

# Identify the optimal lambda
optimal_lambda <- lambda_values[which.min(cv_errors)]
print(paste("Optimal Lambda:", optimal_lambda))



Rcpp:sourceCpp(code = '
#include <RcppArmadillo.h>

// Rest of your code goes here
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]

List gauss_process_regCpp(arma::mat X, arma::vec y, arma::mat Xnew, double sigma2, double l) {
  int n = X.n_rows;
  int nnew = Xnew.n_rows;
  mat K = exp(-1/(2*l*l)*square(X.each_row() - X.t()));
  K.diag() += sigma2;
  mat L = chol(K);
  vec alpha = solve(trimatl(L), solve(trimatu(L), y));
  mat Knew = exp(-1/(2*l*l)*square(X.each_row() - Xnew.t()));
  vec mu = Knew.t() * alpha;
  mat v = solve(trimatu(L), Knew);
  mat cov = exp(-1/(2*l*l)*square(Xnew.each_row() - Xnew.t())) - v.t() * v;
  return List::create(Named("mu") = mu, Named("cov") = cov);
}
')

