#include <Rcpp.h>
using namespace Rcpp;


//' Fourier terms
 //'
 //' @description Generate Fourier terms
 //'
 //' @param time_counter Vector of time counter
 //' @param K Maximum order of Fourier terms
 //' @param period Number of time increments per period
 //'
 //' @return Matrix of the Fourier terms
 //' @export
 //'
 // [[Rcpp::export(name = "GenFT")]]
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