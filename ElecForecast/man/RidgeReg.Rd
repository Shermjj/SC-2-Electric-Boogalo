% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{RidgeReg}
\alias{RidgeReg}
\title{Ridge Regression with Intercept}
\usage{
RidgeReg(X, y, lambda)
}
\arguments{
\item{X}{A numeric matrix of predictor variables where each row is an observation 
and each column is a predictor.}

\item{y}{A numeric vector of the response variable corresponding to each observation.}

\item{lambda}{A numeric value specifying the regularization strength (lambda >= 0).}
}
\value{
An R list containing:
\itemize{
  \item \code{error}: The mean squared error of the model residuals.
  \item \code{coefficients}: A numeric vector of the estimated coefficients, including the intercept.
}
}
\description{
Performs ridge regression on a given set of predictors and a response variable. 
This function adds an intercept to the model by appending a column of ones to 
the predictor matrix `X`. It computes the ridge regression coefficients by 
penalizing the magnitude of the coefficients through a regularization term `lambda`.
The intercept term is not penalized.
}
\details{
The regularization term is added to the diagonal of the cross-product matrix of the
predictors (including the intercept), except for the first diagonal element which corresponds
to the intercept. This adjustment ensures that the intercept is not shrunk towards zero.
The solution to the ridge regression problem is computed using the matrix inversion lemma,
which ensures numerical stability and efficiency.
}
\examples{
\dontrun{
library(Rcpp)
sourceCpp("path/to/RidgeReg.cpp") // Ensure this path points to the file containing the function

# Simulate some data
set.seed(123)
X <- matrix(rnorm(100 * 10), ncol = 10)
y <- X \%*\% rnorm(10) + rnorm(100)
lambda <- 0.5

# Perform ridge regression
results <- RidgeReg(X, y, lambda)
print(results$error)
print(results$coefficients)
}

}
