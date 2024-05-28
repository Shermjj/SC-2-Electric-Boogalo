#' Add two numbers together
#' 
#' @param x,y A pair of numbers.
#' @export
linReg <- function(df, y="DE", x="temp+toy+dow"){
  return(lm(as.formula(paste(y, "~", x)), data=df))
}