
test_that("Output matrix has correct dimensions", {
  X <- matrix(rnorm(50), nrow=10)
  K <- rbf_kernel(X, 1, 1)
  expect_equal(dim(K), c(10, 10))
})

test_that("Kernel matrix is symmetric", {
  X <- matrix(rnorm(50), nrow=10)
  K <- rbf_kernel(X, 1, 1)
  expect_true(isTRUE(all(K == t(K))))
})

test_that("All entries are sigma^2 when all rows of X are the same", {
  X <- matrix(rep(1, 50), nrow=10)  # All rows are the same
  sigma <- 2
  K <- rbf_kernel(X, 1, sigma)
  expect_true(all(K == matrix(sigma^2, 10, 10)))
})

test_that("Kernel matrix is positive definite for positive l and sigma", {
  X <- matrix(rnorm(50), nrow=10)
  l <- 1
  sigma <- 1
  K <- rbf_kernel(X, l, sigma)
  eigenvalues <- eigen(K)$values
  expect_true(all(eigenvalues > 0))
})

test_that("Function reacts correctly to different sigma and l values", {
  X <- matrix(rnorm(20), nrow=5)
  l1 <- 0.5
  sigma1 <- 0.5
  K1 <- rbf_kernel(X, l1, sigma1)
  
  l2 <- 2
  sigma2 <- 2
  K2 <- rbf_kernel(X, l2, sigma2)
  
  # Expect different outputs for different parameters
  expect_false(all(K1 == K2))
})