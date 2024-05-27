

irish_data <- readRDS("data/daily_agg_data.RDS")

print(head(irish_data))


# Define a function which performs Gaussian process regression
gaussian_process_reg <- function(data, kernel = "rbfdot",plot = FALSE) {
  train_indices <- createDataPartition(data$demand, p = 0.7, list = FALSE)
  train_set <- data[train_indices, ]

# Use the remaining 30% for the test set
  test_set <- data[-train_indices, ]

  x = as.vector(train_set$dateTime)
  y = as.vector(train_set$demand)
  # Define the Gaussian process model
  gpr_model <- kernlab::gausspr(x,y, kernel = kernel,kpar = list(sigma = 100))
  prediction <- data.frame(
    dateTime = data$dateTime,
    mean = predict(gpr_model, as.vector(data$dateTime)))
  test_predictions <- predict(gpr_model,as.vector(test_set$dateTime))
  performance <- postResample(pred = test_predictions, obs = test_set$demand)

  data_predict <- left_join(data, prediction, by = "dateTime")
  if (plot){
    pl <- ggplot(data_predict, aes(x = dateTime)) +
      geom_point(aes(y = demand), color = "#000000") +
      geom_line(aes(y = mean), color = "red") +
      labs(title = "GPR model predictions",
           x = "Date", y = "Predicted demand")
    
  } else {
    pl <- NULL
  }
  return(list(model = gpr_model, data = data_predict, pl = pl,performance= performance))
}

#Define a function which performs gaussian process regression via Rcpp
