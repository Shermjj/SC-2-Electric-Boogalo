#include <Rcpp.h>
using namespace Rcpp;
//' Generate Fourier Transform Matrix
//'
//' This function computes the Fourier transform terms for a given vector of time points.
//' It constructs a matrix with sine and cosine terms up to the specified harmonic order `K`.
//' Each row corresponds to a time point, with interleaved sine and cosine values for
//' frequencies from `1` to `K`.
//'
//' @param time_counter A numeric vector of time points at which the Fourier terms are evaluated.
//' @param K An integer specifying the number of harmonics to include in the transformation.
//' @param period A double indicating the period with respect to which the frequencies are calculated.
//'
//' @return A numeric matrix where each row corresponds to a time point in `time_counter`.
//'         Each row contains `2 * K` entries: for each k from 1 to K, there is a sine term
//'         followed by a cosine term corresponding to the k-th harmonic.
//'
//' @examples
//' time_vec <- seq(0, 10, length.out = 100)
//' K_value <- 5
//' period_value <- 10
//' fourier_matrix <- GenFT(time_vec, K_value, period_value)
//' dim(fourier_matrix) # should be 100 x 10
//'
//' @export
// [[Rcpp::export]]
 NumericMatrix GenFT(NumericVector time_counter, int K, double period) {
   int n = time_counter.size();
   NumericMatrix terms(n, 2 * K);
   
   for (int i = 0; i < n; ++i) {
     for (int k = 1; k <= K; ++k) {
       terms(i, 2 * (k - 1)) = sin(2 * M_PI * k * time_counter[i] / period);
       terms(i, 2 * (k - 1) + 1) = cos(2 * M_PI * k * time_counter[i] / period);
     }
   }
   
   return terms;
 }