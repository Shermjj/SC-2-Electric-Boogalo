library(kernlab)
library(dplyr)
library(caret)
library(ggplot2)
library(ElecForecast)

#Import dataframes for daily, hourly, and half-hourly data
df_half_hour <- readRDS("data/df_halfhr.RData")
df_hour <- readRDS("data/df_hr.RData")
df_day <- readRDS("data/df_day.RData")

#Assign Column 1 a name
colnames(df_day)[0] <- "dateTime"
colnames(df_hour)[0] <- "dateTime"
colnames(df_half_hour)[0] <- "dateTime"

all_plots <- list()

class_names <- list("DE", "C1", "C2", "AB", "F")
class_colours <- c("DE" = "#9900ff",
                   "C1" = "#d95f02",
                   "C2" = "#0059ff",
                   "AB" = "#e7298a",
                   "F" = "#45b828")

for (class_name in class_names) {
  # Perform Gaussian Process Regression on the class
  gpr_result <- gaussian_process_reg(df_day, class = class_name, plot = TRUE)
  
  # Add the plot to all_plots
  all_plots[[class_name]] <- gpr_result$plot
}


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
