// calc_density.h
#pragma once

// [[Rcpp::depends(Rcpp)]]
// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace Rcpp;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::LLT;

using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

static inline double row_max_at(const NumericMatrix& x, int i) {
  const int p = x.ncol();
  double m = x(i, 0);
  for (int j = 1; j < p; ++j) {
    const double v = x(i, j);
    if (v > m) m = v;
  }
  return m;
}

static inline NumericVector row_max_vec(const NumericMatrix& x) {
  const int n = x.nrow();
  NumericVector out(n);
  for (int i = 0; i < n; ++i) {
    out[i] = row_max_at(x, i);
  }
  return out;
}

// log N(x | 0, S)
// Core Gaussian log-density used by Stage-A.
static inline double ldmvnorm0(const VectorXd& x, const MatrixXd& S) {
  const double LOG2PI = std::log(2.0 * M_PI);
  LLT<MatrixXd> chol(S);
  if (chol.info() != Eigen::Success) return R_NegInf;
  MatrixXd L = chol.matrixL();
  double logdet = 2.0 * L.diagonal().array().log().sum();
  VectorXd y = chol.solve(x);
  return -0.5 * (x.size() * LOG2PI + logdet + x.dot(y));
}

static inline double ldmvnorm_mean(const VectorXd& x,
                                   const VectorXd& mu,
                                   const MatrixXd& S) {
  const double LOG2PI = std::log(2.0 * M_PI);
  LLT<MatrixXd> chol(S);
  if (chol.info() != Eigen::Success) return R_NegInf;
  VectorXd xc = x - mu;
  MatrixXd L = chol.matrixL();
  double logdet = 2.0 * L.diagonal().array().log().sum();
  VectorXd y = chol.solve(xc);
  return -0.5 * (x.size() * LOG2PI + logdet + xc.dot(y));
}

// stably compute log(sum(exp(v)))
static inline double lse_row(const std::vector<double>& v) {
  double m = *std::max_element(v.begin(), v.end());
  if (!std::isfinite(m)) return m;
  double s = 0.0;
  for (double x : v) if (std::isfinite(x)) s += std::exp(x - m);
  return m + std::log(s);
}

// log-sum-exp over config terms using Gaussian loglik.
// If idx_map is provided, it maps local dimension -> original index.
template <typename TermList>
static inline double logsum_terms_dense(const VectorXd& x_O,
                                        const MatrixXd& E_OO,
                                        const TermList& terms,
                                        const std::vector<int>* idx_map) {
  if (terms.empty()) return R_NegInf;
  const int o = x_O.size();
  MatrixXd Sigma = E_OO;
  VectorXd base_diag = E_OO.diagonal();
  for (int a = 0; a < o; ++a) base_diag(a) += 1e-10;

  double maxv = R_NegInf;
  double sumexp = 0.0;
  for (const auto& term : terms) {
    for (int a = 0; a < o; ++a) {
      const int j = idx_map ? (*idx_map)[a] : a;
      Sigma(a, a) = base_diag(a) + term.prior_var[j];
    }
    const double ll = ldmvnorm0(x_O, Sigma);
    const double val = term.log_weight + ll;
    if (!std::isfinite(val)) continue;
    if (val > maxv) {
      sumexp = std::isfinite(maxv) ? (sumexp * std::exp(maxv - val) + 1.0) : 1.0;
      maxv = val;
    } else {
      sumexp += std::exp(val - maxv);
    }
  }
  return std::isfinite(maxv) ? (maxv + std::log(sumexp)) : R_NegInf;
}

// -----------------------------------------------------------------------------
// log-likelihood matrices (shared C and per-effect C_i) — normal only
// -----------------------------------------------------------------------------

// shared correlation matrix
RowMat logmat_normal(const NumericMatrix &Bhat,
                     const NumericMatrix &S,
                     const NumericMatrix &C_cor,
                     const IntegerMatrix &alloc_mat,
                     const std::vector< std::vector<double> > &grids,
                     bool use_parallel = true);

// per-effect noise matrices supplied as a list (length n)
RowMat logmat_normal_list(const NumericMatrix &Bhat,
                          const NumericMatrix &S,
                          const List &C_list,
                          const IntegerMatrix &alloc_mat,
                          const std::vector< std::vector<double> > &grids,
                          bool use_parallel = true);
