library(kernlab)
library(dplyr)
library(caret)
library(ggplot2)
library(Rcpp)
library(RcppArmadillo)

#Import dataframes for daily, hourly, and half-hourly data
df_half_hour <- readRDS("data/df_halfhr.RData")
df_hour <- readRDS("data/df_hr.RData")
df_day <- readRDS("data/df_day.RData")

#Assign Column 1 a name
colnames(df_day)[0] <- "dateTime"
colnames(df_hour)[0] <- "dateTime"
colnames(df_half_hour)[0] <- "dateTime"


# Define a function which performs Gaussian process regression
gaussian_process_reg <- function(data,
                                 class = "DE",
                                 kernel = "rbfdot",
                                 plot = FALSE,
                                 sigma = 100) {

  #Training and test set, first 90% of data is training, last 10% is test
  train_index <- round(nrow(data) * 0.9)
  train_set <- data[1:train_index, ]
  test_set <- data[(train_index + 1):nrow(data), ]

  # Define the Gaussian process model
  gpr_model <- kernlab::gausspr(as.vector(train_set$counter),
                                as.vector(train_set[[class]]),
                                kernel = kernel,
                                kpar = list(sigma = sigma))

  # Predict the mean function for plotting
  mean_func <- predict(gpr_model, as.vector(data$counter))
  prediction <- data.frame(
                           counter = data$counter,
                           mean = mean_func)

  data_with_pred <- left_join(data, prediction, by = "counter")

  # Evaluate performance
  test_mean_func <- predict(gpr_model, as.vector(test_set$counter))
  performance <- postResample(pred = test_mean_func, obs = test_set$DE)[["RMSE"]]^2

  # Include Plots ?
  if (plot){
    pl <- ggplot(data_with_pred, aes(x = counter)) +
      geom_point(aes(y = get(class)), color = "#5a5a5a", size = 0.2) +
      geom_line(aes(y = mean), color = class_colours[class]) +
      labs(title = paste("GPR model predictions for class", class),
           x = "Date", y = "Demand")
  } else {
    pl <- NULL
  }
  return(list(model = gpr_model,
              data = data_with_pred,
              performance = performance,
              plot = pl))
}


find_optimal_params <- function(X, y) {
  # Source the C++ file
  sourceCpp("gauss_process_reg.cpp")
  
  # Call the C++ function
  theta <- optimise_gaussian_process(X, y)
  
  # Return the optimal parameters
  return(theta)
}




#Perform Gaussian Process Regression on daily data for each class

all_plots <- list()

class_names <- list("DE", "C1", "C2", "AB", "F")
class_colours <- c("DE" = "#ff0000",
                   "C1" = "#d95f02",
                   "C2" = "#0059ff",
                   "AB" = "#e7298a",
                   "F" = "#45b828")



save_gpr_data <- function(df,n){
  all_plots <- list()
  for (class_name in class_names) {
  df_hour <- df_hour[1:n,]
   #Perform Gaussian Process Regression on the class
  gpr_result <- gaussian_process_reg(df_hour, class = class_name, plot = TRUE)
  df <- gpr_result$data
  saveRDS(df, file = paste0("data/data_gpr/df_hourWEEK_gpr_",class_name,".RData"))
  jpeg(filename = paste0("data/plots_gpr/hourWEEK_",class_name,".jpeg"), width = 1500, height = 1500, res = 300)
  print(gpr_result$plot)
  dev.off()
  
  all_plots[[class_name]] <- gpr_result$plot
}


}

class_names <- list("DE", "C1", "C2", "AB", "F")

display_plots <- function(class_name) {
  # Load the data
  df <- readRDS(file = paste0("data/data_gpr/df_hour_gpr_", class_name, ".RData"))
  df <- df[1:168,]
  # Create a plot
  p <- ggplot(df, aes(x = counter)) +
    geom_point(aes(y = get(class_name)), color = "#5a5a5a", size = 0.5) +
    geom_line(aes(y = mean), color = class_colours[class_name]) +
    labs(title = paste("GPR model predictions for class", class_name),
         x = "Date", y = "Demand")
  
  # Return the plot
  return(p)
}


get_rmse <- function(class) {
  perf <- gaussian_process_reg(df_half_hour[1:336,], class = class, plot = TRUE)$performance
  return(data.frame(class = class, mse = perf))
}

class_names <- list("DE", "C1", "C2", "AB", "F")

# Use lapply to get the RMSE for each class and combine the results into a data frame
mse_results <- do.call(rbind, lapply(class_names, get_rmse))

# Print the results
print(mse_results)

saveRDS(mse_results, file = "data/data_gpr/mse_results_half_hour.RData")