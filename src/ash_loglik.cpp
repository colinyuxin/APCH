// [[Rcpp::depends(Rcpp)]]
// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>
#include <cmath>

#include "calc_density.h"

using namespace Rcpp;

// Compute log N(x | 0, s^2 + sd^2) for each (i,k).
// Returns n x K matrix.
// [[Rcpp::export]]
NumericMatrix loglik_normalmix_cpp(const NumericVector& x,
                                   const NumericVector& s,
                                   const NumericVector& sd) {
  const int n = x.size();
  const int K = sd.size();
  NumericMatrix out(n, K);
  const double log2pi = std::log(2.0 * M_PI);
  for (int k = 0; k < K; ++k) {
    const double v_add = sd[k] * sd[k];
    for (int i = 0; i < n; ++i) {
      const double v = s[i] * s[i] + v_add;
      const double logv = std::log(v);
      const double z = x[i] * x[i] / v;
      out(i, k) = -0.5 * (log2pi + logv + z);
    }
  }
  return out;
}

// Same as loglik_normalmix_cpp but only for a subset of grid indices.
// idx is 1-based, as in R.
// [[Rcpp::export]]
NumericMatrix loglik_normalmix_subset_cpp(const NumericVector& x,
                                          const NumericVector& s,
                                          const NumericVector& sd,
                                          const IntegerVector& idx) {
  const int n = x.size();
  const int K = idx.size();
  NumericMatrix out(n, K);
  const double log2pi = std::log(2.0 * M_PI);
  for (int kk = 0; kk < K; ++kk) {
    const int k = idx[kk] - 1;
    const double v_add = sd[k] * sd[k];
    for (int i = 0; i < n; ++i) {
      const double v = s[i] * s[i] + v_add;
      const double logv = std::log(v);
      const double z = x[i] * x[i] / v;
      out(i, kk) = -0.5 * (log2pi + logv + z);
    }
  }
  return out;
}

// [[Rcpp::export]]
NumericVector lfdr_null_from_logL_cpp(const NumericMatrix& logL,
                                      const NumericVector& pi,
                                      const int null_index) {
  const int n = logL.nrow();
  const int k = logL.ncol();
  const int idx0 = null_index - 1;
  NumericVector out(n);
  for (int i = 0; i < n; ++i) {
    double m = row_max_at(logL, i);
    double denom = 0.0;
    for (int j = 0; j < k; ++j) {
      denom += std::exp(logL(i, j) - m) * pi[j];
    }
    double numer = std::exp(logL(i, idx0) - m) * pi[idx0];
    out[i] = (denom > 0.0) ? (numer / denom) : 0.0;
  }
  return out;
}


// Compute row-wise mixture CDF for normal-mixture components.
// [[Rcpp::export]]
NumericVector mixcdf_normal_cpp(const NumericVector& x,
                                const NumericVector& s,
                                const NumericVector& sd,
                                const NumericVector& pi) {
  const int n = x.size();
  const int K = sd.size();

  NumericVector out(n);
  for (int i = 0; i < n; ++i) {
    double ui = 0.0;
    const double si2 = s[i] * s[i];
    for (int k = 0; k < K; ++k) {
      const double v = si2 + sd[k] * sd[k];
      const double z = x[i] / std::sqrt(v);
      ui += pi[k] * R::pnorm5(z, 0.0, 1.0, 1, 0);
    }
    out[i] = ui;
  }
  return out;
}

// Compute row-wise mixture CDF with fixed-null cdf vector.
// [[Rcpp::export]]
NumericVector mixcdf_fixednull_cpp(const NumericVector& x,
                                   const NumericVector& s,
                                   const NumericVector& sd_pos,
                                   const NumericVector& pi_pos,
                                   const NumericVector& cdf_null,
                                   const double pi0) {
  const int n = x.size();
  const int K = sd_pos.size();

  NumericVector out(n);
  for (int i = 0; i < n; ++i) {
    double ui = pi0 * cdf_null[i];
    const double si2 = s[i] * s[i];
    for (int k = 0; k < K; ++k) {
      const double v = si2 + sd_pos[k] * sd_pos[k];
      const double z = x[i] / std::sqrt(v);
      ui += pi_pos[k] * R::pnorm5(z, 0.0, 1.0, 1, 0);
    }
    out[i] = ui;
  }
  return out;
}
