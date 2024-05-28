library(kernlab)
library(dplyr)
library(caret)
library(ggplot2)

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
  gpr_model <- kernlab::gausspr(as.vector(train_set$toy),
                                as.vector(train_set[[class]]),
                                kernel = kernel, kpar = list(sigma = sigma))

  # Predict the mean function for plotting
  mean_func <- predict(gpr_model, as.vector(data$toy))
  prediction <- data.frame(
                           toy = data$toy,
                           mean = mean_func)

  data_with_pred <- left_join(data, prediction, by = "toy")

  # Evaluate performance
  test_mean_func <- predict(gpr_model, as.vector(test_set$toy))
  performance <- postResample(pred = test_mean_func, obs = test_set$DE)

  # Include Plots ?
  if (plot){
    pl <- ggplot(data_with_pred, aes(x = toy)) +
      geom_point(aes(y = get(class)), color = "#414141") +
      geom_line(aes(y = mean), color = class_colours[class]) +
      labs(title = "GPR model predictions",
           x = "Date", y = "Demand")
    pl
  } else {
    pl <- NULL
  }
  return(list(model = gpr_model,
              data <- data_with_pred,
              performance,
              plot = pl))
}


#print(head(df_day))
#print(head(df_hour))
#print(head(df_half_hour))

#Perform Gaussian Process Regression on daily data for each class

all_plots <- list()

class_names <- list("DE", "C1", "C2", "AB", "F")
class_colours <- c("DE" = "#9900ff",
                   "C1" = "#d95f02",
                   "C2" = "#0059ff",
                   "AB" = "#e7298a",
                   "F" = "#45b828")

for (class_name in class_names) {
  # Perform Gaussian Process Regression on the class
  gpr_result <- gaussian_process_reg(df_hour, class = class_name, plot = TRUE)
  
  # Add the plot to all_plots
  all_plots[[class_name]] <- gpr_result$plot
}

# Combine all the plots into a single plot
combined_plot <- patchwork::wrap_plots(all_plots, ncol = 1)
png(filename = "data/plots_gpr/hourlycombined.png")
print(combined_plot)
dev.off()


# Save daily plots
for (class_name in class_names) {
  # Create a new .png file
  png(filename = paste0("data/plots_gpr/hourly_", class_name, ".png"))
  
  # Print the plot to the .png file
  print(all_plots[[class_name]])
  
  # Close the .png file
  dev.off()
}

