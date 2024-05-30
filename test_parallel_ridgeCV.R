# Load our package
library(ElecForecast)
library(dplyr)

df_halfhr_scaled <- readRDS(here::here("data/df_halfhr_scaled.RData"))
df_hr_scaled <- readRDS(here::here("data/df_hr_scaled.RData"))
df_day_scaled <- readRDS(here::here("data/df_day_scaled.RData"))

# Split the data into training and test sets
# 90% training set
n_train_halfhr <- round(0.9*nrow(df_halfhr_scaled))
# Training and testing counter
train_counter_halfhr <- df_halfhr_scaled$counter[1:n_train_halfhr]
test_counter_halfhr <- df_halfhr_scaled$counter[-(1:n_train_halfhr)]
# Training and testing features
halfhr_features_train <- df_halfhr_scaled[1:n_train_halfhr,] %>%
  select(-counter)
halfhr_features_test <- df_halfhr_scaled[-(1:n_train_halfhr),] %>%
  select(-counter)

# Range of lambda values
lambda_values_halfhr <- seq(0.01, 1, length.out = 10) * nrow(halfhr_features_train)

# DE with original function
tictoc::tic()
de_orig <- parallel_ridge_cross_validation(
  x_vars = as.matrix(halfhr_features_train[,7:ncol(halfhr_features_train)]),
  y_var = halfhr_features_train$DE,
  time_counter = train_counter_halfhr,
  daily_period = 48,
  annual_period = 48 * 365,
  max_K = 5,
  lambda_values = lambda_values_halfhr,
  n_folds = 10
)
tictoc::toc() # 2.085 sec

# DE with new function
tictoc::tic()
de_orig_new <- parallel_ridge_cross_validation2(
  x_vars = as.matrix(halfhr_features_train[,7:ncol(halfhr_features_train)]),
  y_var = halfhr_features_train$DE,
  time_counter = train_counter_halfhr,
  daily_period = 48,
  annual_period = 48 * 365,
  max_K = 5,
  lambda_values = lambda_values_halfhr,
  n_folds = 10
)
tictoc::toc()