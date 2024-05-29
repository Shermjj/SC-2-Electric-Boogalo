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
