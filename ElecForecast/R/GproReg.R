#' Gaussian Process Regression for Demand Estimation
#'
#' This function applies Gaussian Process Regression (GPR) using the \code{kernlab::gausspr} function
#' to model and predict the specified class variable based on a subset of predictor variables from the dataset.
#' It splits the data into training and test sets, trains the GPR model, evaluates its performance,
#' and optionally generates a plot of the predictions.
#'
#' @param data A data frame containing the variables of interest.
#' @param class A character string specifying the column name of the dependent variable.
#' @param kernel A character string specifying the kernel function to use in the GPR model.
#'               Default is "rbfdot", which indicates a radial basis function kernel.
#' @param plot Logical; if TRUE, a plot of the GPR predictions versus actual data will be displayed.
#' @param sigma Numeric; the sigma parameter for the kernel function, controlling the width of the kernel.
#'              Default value is 100.
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{model}: The trained GPR model object.
#'   \item \code{data}: The input data frame with an additional column for the mean predictions.
#'   \item \code{performance}: A list with performance metrics including RMSE, R squared, etc.
#'   \item \code{plot}: An optional ggplot object visualizing the actual data and the GPR predictions.
#' }
#'
#' @examples
#' \dontrun{
#' data <- data.frame(toy = seq(as.Date("2020-01-01"), by = "day", length.out = 100),
#'                    temp = rnorm(100, mean = 15, sd = 5),
#'                    DE = rnorm(100, mean = 200, sd = 20))
#' gpr_results <- gaussian_process_reg(data,
#'                                     class = "DE",
#'                                     kernel = "rbfdot",
#'                                     plot = TRUE,
#'                                     sigma = 50)
#' }
#'
#' @importFrom kernlab gausspr
#' @importFrom caret postResample
#' @importFrom dplyr left_join
#' @importFrom ggplot2 ggplot aes geom_point geom_line labs
#' @export
gaussian_process_reg <- function(data,
                                 class = "DE",
                                 kernel = "rbfdot",
                                 plot = FALSE,
                                 sigma = 1000) {
  
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
  performance <- postResample(pred = test_mean_func, obs = test_set$DE)
  
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
              performance,
              plot = pl))
  }
