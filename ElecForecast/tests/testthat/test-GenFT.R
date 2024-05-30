test_that("Function returns matrix of correct dimensions", {
  time_vec <- seq(0, 10, length.out = 100)
  K_value <- 5
  period_value <- 10
  result <- GenFT(time_vec, K_value, period_value)
  expect_equal(dim(result), c(100, 10))
})

test_that("Function handles K = 0 correctly", {
  time_vec <- seq(0, 10, length.out = 100)
  K_value <- 0
  period_value <- 10
  result <- GenFT(time_vec, K_value, period_value)
  expect_equal(dim(result), c(100, 0))
})

test_that("Function handles non-integer period values", {
  time_vec <- seq(0, 10, length.out = 10)
  K_value <- 1
  period_value <- 10.5
  result <- GenFT(time_vec, K_value, period_value)
  expect_equal(dim(result), c(10, 2))
})

test_that("Function performs correctly with large K values", {
  time_vec <- seq(0, 1, length.out = 10)
  K_value <- 100  # Large K
  period_value <- 1
  result <- GenFT(time_vec, K_value, period_value)
  expect_equal(dim(result), c(10, 200))
})