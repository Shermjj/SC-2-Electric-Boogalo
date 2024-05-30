// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
//' Radial Basis Function (RBF) Kernel Matrix Computation
//'
//' This function computes the RBF kernel matrix for a given matrix of input features `X`.
//' The RBF kernel is also known as the Gaussian kernel. It is a measure of similarity
//' between rows in `X` that decays exponentially with the squared Euclidean distance
//' between rows. Each element `K(i, j)` of the output matrix `K` is computed as:
//' \deqn{K(i, j) = \sigma^2 * exp(-||X[i, ] - X[j, ]||^2 / (2 * l^2))}
//' where `||X[i, ] - X[j, ]||` is the Euclidean distance between row `i` and `j` of `X`,
//' `l` is the length scale of the kernel, and `sigma` is the scale parameter.
//'
//' @param X A numeric matrix where each row represents an observation and each column represents a feature.
//' @param l A double specifying the length scale of the kernel.
//' @param sigma A double specifying the scale parameter of the kernel.
//'
//' @return A symmetric numeric matrix of size equal to the number of rows in `X`.
//'         This matrix is the RBF kernel matrix, where each element represents the
//'         similarity between the corresponding pair of observations in `X`.
//'
//' @examples
//' # Generate some data
//' X <- matrix(rnorm(50), nrow=10)
//' # Compute the RBF kernel with length scale 1 and sigma 1
//' K <- rbf_kernel(X, 1, 1)
//' print(K)
//'
//' @export
// [[Rcpp::export]]
arma::mat rbf_kernel(const arma::mat& X, double l, double sigma) {
  arma::mat K = arma::zeros(X.n_rows, X.n_rows);
  for (unsigned i = 0; i < X.n_rows; ++i) {
    for (unsigned j = 0; j < X.n_rows; ++j) {
      double dist = arma::norm(X.row(i) - X.row(j), 2);
      K(i, j) = std::pow(sigma, 2) * std::exp(-std::pow(dist, 2) / (2 * std::pow(l, 2)));
    }
  }
  return K;
}