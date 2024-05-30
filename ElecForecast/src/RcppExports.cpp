// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// GenFT
NumericMatrix GenFT(NumericVector time_counter, int K, double period);
RcppExport SEXP _ElecForecast_GenFT(SEXP time_counterSEXP, SEXP KSEXP, SEXP periodSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type time_counter(time_counterSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< double >::type period(periodSEXP);
    rcpp_result_gen = Rcpp::wrap(GenFT(time_counter, K, period));
    return rcpp_result_gen;
END_RCPP
}
// rbf_kernel
arma::mat rbf_kernel(const arma::mat& X, double l, double sigma);
RcppExport SEXP _ElecForecast_rbf_kernel(SEXP XSEXP, SEXP lSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type l(lSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(rbf_kernel(X, l, sigma));
    return rcpp_result_gen;
END_RCPP
}
// RidgeReg
List RidgeReg(const arma::mat& X, const arma::vec& y, double lambda);
RcppExport SEXP _ElecForecast_RidgeReg(SEXP XSEXP, SEXP ySEXP, SEXP lambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    rcpp_result_gen = Rcpp::wrap(RidgeReg(X, y, lambda));
    return rcpp_result_gen;
END_RCPP
}
// RidgeRegPar
List RidgeRegPar(const arma::mat& X, const arma::vec& y, double lambda);
RcppExport SEXP _ElecForecast_RidgeRegPar(SEXP XSEXP, SEXP ySEXP, SEXP lambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    rcpp_result_gen = Rcpp::wrap(RidgeRegPar(X, y, lambda));
    return rcpp_result_gen;
END_RCPP
}
// parallel_ridge_cross_validation
List parallel_ridge_cross_validation(NumericMatrix x_vars, NumericVector y_var, NumericVector time_counter, double daily_period, double annual_period, int max_K, std::vector<double> lambda_values, int n_folds);
RcppExport SEXP _ElecForecast_parallel_ridge_cross_validation(SEXP x_varsSEXP, SEXP y_varSEXP, SEXP time_counterSEXP, SEXP daily_periodSEXP, SEXP annual_periodSEXP, SEXP max_KSEXP, SEXP lambda_valuesSEXP, SEXP n_foldsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type x_vars(x_varsSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type y_var(y_varSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type time_counter(time_counterSEXP);
    Rcpp::traits::input_parameter< double >::type daily_period(daily_periodSEXP);
    Rcpp::traits::input_parameter< double >::type annual_period(annual_periodSEXP);
    Rcpp::traits::input_parameter< int >::type max_K(max_KSEXP);
    Rcpp::traits::input_parameter< std::vector<double> >::type lambda_values(lambda_valuesSEXP);
    Rcpp::traits::input_parameter< int >::type n_folds(n_foldsSEXP);
    rcpp_result_gen = Rcpp::wrap(parallel_ridge_cross_validation(x_vars, y_var, time_counter, daily_period, annual_period, max_K, lambda_values, n_folds));
    return rcpp_result_gen;
END_RCPP
}
// predict_parallel_ridge_cv
NumericVector predict_parallel_ridge_cv(List model, NumericMatrix x_test, NumericVector time_counter, double daily_period, double annual_period);
RcppExport SEXP _ElecForecast_predict_parallel_ridge_cv(SEXP modelSEXP, SEXP x_testSEXP, SEXP time_counterSEXP, SEXP daily_periodSEXP, SEXP annual_periodSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type model(modelSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type x_test(x_testSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type time_counter(time_counterSEXP);
    Rcpp::traits::input_parameter< double >::type daily_period(daily_periodSEXP);
    Rcpp::traits::input_parameter< double >::type annual_period(annual_periodSEXP);
    rcpp_result_gen = Rcpp::wrap(predict_parallel_ridge_cv(model, x_test, time_counter, daily_period, annual_period));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_ElecForecast_GenFT", (DL_FUNC) &_ElecForecast_GenFT, 3},
    {"_ElecForecast_rbf_kernel", (DL_FUNC) &_ElecForecast_rbf_kernel, 3},
    {"_ElecForecast_RidgeReg", (DL_FUNC) &_ElecForecast_RidgeReg, 3},
    {"_ElecForecast_RidgeRegPar", (DL_FUNC) &_ElecForecast_RidgeRegPar, 3},
    {"_ElecForecast_parallel_ridge_cross_validation", (DL_FUNC) &_ElecForecast_parallel_ridge_cross_validation, 8},
    {"_ElecForecast_predict_parallel_ridge_cv", (DL_FUNC) &_ElecForecast_predict_parallel_ridge_cv, 5},
    {NULL, NULL, 0}
};

RcppExport void R_init_ElecForecast(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
