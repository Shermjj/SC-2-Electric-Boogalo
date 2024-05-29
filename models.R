library(kernlab)
library(dplyr)
library(caret)
library(ggplot2)
library(ElecForecast)

set.seed(123)
n <- 100
n_vars <- 5
max_K <- 10
daily_period <- 24
annual_period <- 365.25
n_folds <- 5
lambda_values <- seq(0.01, 0.1, by = 0.01)
# Simulate independent variables
x_vars <- matrix(rnorm(n * n_vars), ncol = n_vars)
# Create a time vector (e.g., hourly data over n hours)
time_counter <- seq(1, n, by = 1)
# Simulate a dependent variable
y_var <- sin(2 * pi * time_counter / daily_period) +
  sin(2 * pi * time_counter / annual_period) +
  rnorm(n)
# Run the cross-validation function
result <- parallel_ridge_cross_validation(x_vars, y_var, time_counter, daily_period, annual_period,max_K, lambda_values, n_folds)
result
RidgeReg(x_vars, y_var, 0.1)
