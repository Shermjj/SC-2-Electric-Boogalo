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
double performRidgeRegression(const arma::mat& X, const arma::vec& y, double lambda) {
    int n = X.n_rows, k = X.n_cols;
    arma::mat XtX = arma::trans(X) * X;
    arma::mat ridgePenalty = lambda * arma::eye<arma::mat>(k, k);
    arma::mat XtX_ridge = XtX + ridgePenalty;

    arma::colvec coef = arma::solve(XtX_ridge, arma::trans(X) * y);
    arma::colvec resid = y - X * coef;
    double sig2 = arma::as_scalar(arma::trans(resid) * resid / (n - k));
    return sig2; // Using MSE as the error metric
}

// Worker for parallel computation of CV errors
struct RidgeCVWorker : public Worker {
    const arma::mat& X;
    const arma::vec& y;
    const std::vector<double>& lambda_values;
    RVector<double> errors;

    RidgeCVWorker(const arma::mat& X, const arma::vec& y, const std::vector<double>& lambda_values, NumericVector errors)
    : X(X), y(y), lambda_values(lambda_values), errors(errors) {}

    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; i++) {
            double lambda = lambda_values[i];
            double error = performRidgeRegression(X, y, lambda);
            errors[i] = error;
        }
    }
};

// Rcpp export to perform parallel CV
// [[Rcpp::export]]
NumericVector parallelRidgeCV(const arma::mat& X, const arma::vec& y, std::vector<double> lambda_values) {
    NumericVector errors(lambda_values.size());
    RidgeCVWorker worker(X, y, lambda_values, errors);
    
    parallelFor(0, lambda_values.size(), worker);
    
    return errors;
}
')

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

// Function to perform ridge regression on a subset of data
double performRidgeRegression(const arma::mat& X, const arma::vec& y, const arma::uvec& train_idx, const arma::uvec& test_idx, double lambda) {
    arma::mat X_train = X.rows(train_idx);
    arma::vec y_train = y.elem(train_idx);
    arma::mat X_test = X.rows(test_idx);
    arma::vec y_test = y.elem(test_idx);

    int n = X_train.n_rows, k = X_train.n_cols;
    arma::mat XtX = arma::trans(X_train) * X_train;
    arma::mat ridgePenalty = lambda * arma::eye<arma::mat>(k, k);
    arma::mat XtX_ridge = XtX + ridgePenalty;

    arma::colvec coef = arma::solve(XtX_ridge, arma::trans(X_train) * y_train);
    arma::vec preds = X_test * coef;
    double mse = mean(square(preds - y_test));
    return mse;
}

// Worker for parallel computation of CV errors using forward chaining
struct RidgeCVWorker : public Worker {
    const arma::mat& X;
    const arma::vec& y;
    const std::vector<double>& lambda_values;
    const std::vector<arma::uvec>& train_indices;
    const std::vector<arma::uvec>& test_indices;
    RVector<double> errors;

    RidgeCVWorker(const arma::mat& X, const arma::vec& y, const std::vector<double>& lambda_values,
                  const std::vector<arma::uvec>& train_indices, const std::vector<arma::uvec>& test_indices, NumericVector errors)
    : X(X), y(y), lambda_values(lambda_values), train_indices(train_indices), test_indices(test_indices), errors(errors) {}

    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; i++) {
            double lambda = lambda_values[i];
            double total_mse = 0.0;
            for (size_t j = 0; j < train_indices.size(); ++j) {
                double mse = performRidgeRegression(X, y, train_indices[j], test_indices[j], lambda);
                total_mse += mse;
            }
            errors[i] = total_mse / train_indices.size(); // Average MSE over all folds
        }
    }
};

// Rcpp export to perform parallel CV with forward chaining
// [[Rcpp::export]]
NumericVector parallelRidgeCVTimeSeries(const arma::mat& X, const arma::vec& y, std::vector<double> lambda_values,
                                        std::vector<arma::uvec> train_indices, std::vector<arma::uvec> test_indices) {
    NumericVector errors(lambda_values.size());
    RidgeCVWorker worker(X, y, lambda_values, train_indices, test_indices, errors);
    
    parallelFor(0, lambda_values.size(), worker);
    
    return errors;
}
')


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
