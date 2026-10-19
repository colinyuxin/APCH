// [[Rcpp::depends(RcppParallel)]]
#include <Rcpp.h>
#include <RcppParallel.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using namespace Rcpp;
using namespace RcppParallel;

static inline std::vector<int> active_idx_from_code(int code, int p) {
  std::vector<int> out;
  out.reserve(p);
  for (int j = 0; j < p; ++j) {
    if (((code >> (p - 1 - j)) & 1) != 0) out.push_back(j);
  }
  return out;
}

static inline double logsumexp_vec(const std::vector<double>& x) {
  double m = R_NegInf;
  for (double v : x) if (std::isfinite(v) && v > m) m = v;
  if (!std::isfinite(m)) return R_NegInf;
  double s = 0.0;
  for (double v : x) if (std::isfinite(v)) s += std::exp(v - m);
  return m + std::log(s);
}

static inline std::vector<double> normalize_logweights(const std::vector<double>& x) {
  std::vector<double> out(x.size(), R_NegInf);
  const double lse = logsumexp_vec(x);
  if (!std::isfinite(lse)) return out;
  for (std::size_t i = 0; i < x.size(); ++i) out[i] = x[i] - lse;
  return out;
}

static inline double row_max_loglik(const NumericMatrix& x, int i) {
  const int k = x.ncol();
  double m = R_NegInf;
  for (int j = 0; j < k; ++j) {
    const double v = x(i, j);
    if (v > m) m = v;
  }
  return m;
}

static inline std::vector< std::vector<double> > list_to_numvecs(const List& x) {
  const int n = x.size();
  std::vector< std::vector<double> > out(n);
  for (int i = 0; i < n; ++i) {
    NumericVector xi = x[i];
    out[i].assign(xi.begin(), xi.end());
  }
  return out;
}

// Minimal APCH-local copy of mixsqp core adapted from the upstream mixsqp
// package. We keep only the direct dense path we use in APCH.
// Build mixsqp inputs from log-likelihood matrix.
// Returns list(A, w, row_max).
// [[Rcpp::export]]
List build_mixsqp_inputs_cpp(const NumericMatrix& logL,
                             const NumericVector& prior) {
  const int n = logL.nrow();
  const int k = logL.ncol();

  std::vector<int> diag_idx;
  diag_idx.reserve(k);
  for (int j = 0; j < k; ++j) {
    if (prior[j] - 1.0 != 0.0) diag_idx.push_back(j);
  }
  const int n_diag = static_cast<int>(diag_idx.size());

  NumericVector row_max(n);
  NumericMatrix A(n + n_diag, k);
  NumericVector w(n + n_diag);

  for (int r = 0; r < n_diag; ++r) {
    const int j = diag_idx[r];
    A(r, j) = 1.0;
    w[r] = prior[j] - 1.0;
  }

  for (int i = 0; i < n; ++i) {
    const double m = row_max_loglik(logL, i);
    row_max[i] = m;
    const int row = n_diag + i;
    for (int j = 0; j < k; ++j) {
      A(row, j) = std::exp(logL(i, j) - m);
    }
    w[row] = 1.0;
  }

  return List::create(
    _["A"] = A,
    _["w"] = w,
    _["row_max"] = row_max
  );
}

// Compute posterior probabilities and mean log-likelihood directly from
// log-likelihood matrix and mixture weights, without materializing log-post.
// [[Rcpp::export]]
List posterior_from_logL_cpp(const NumericMatrix& logL,
                             const NumericVector& pi) {
  const int n = logL.nrow();
  const int k = logL.ncol();
  NumericMatrix post_prob(n, k);

  std::vector<double> log_pi(k);
  for (int j = 0; j < k; ++j) {
    log_pi[j] = (pi[j] > 0.0) ? std::log(pi[j]) : R_NegInf;
  }

  double acc = 0.0;
  for (int i = 0; i < n; ++i) {
    double m = R_NegInf;
    for (int j = 0; j < k; ++j) {
      const double v = logL(i, j) + log_pi[j];
      if (v > m) m = v;
      post_prob(i, j) = v;
    }

    double rs = 0.0;
    for (int j = 0; j < k; ++j) {
      const double v = post_prob(i, j);
      if (R_finite(v)) {
        post_prob(i, j) = std::exp(v - m);
        rs += post_prob(i, j);
      } else {
        post_prob(i, j) = 0.0;
      }
    }

    const double inv_rs = 1.0 / rs;
    for (int j = 0; j < k; ++j) {
      post_prob(i, j) *= inv_rs;
    }
    acc += m + std::log(rs);
  }

  return List::create(
    _["post_prob"] = post_prob,
    _["ll"] = acc / static_cast<double>(n)
  );
}

// Compute posterior column sums and mean log-likelihood directly from
// log-likelihood matrix and mixture weights, without materializing post_prob.
// [[Rcpp::export]]
List posterior_colsum_from_logL_cpp(const NumericMatrix& logL,
                                    const NumericVector& pi) {
  const int n = logL.nrow();
  const int k = logL.ncol();
  NumericVector post_colsum(k);

  std::vector<double> log_pi(k);
  for (int j = 0; j < k; ++j) {
    log_pi[j] = (pi[j] > 0.0) ? std::log(pi[j]) : R_NegInf;
  }

  double acc = 0.0;
  std::vector<double> row_post(k);
  for (int i = 0; i < n; ++i) {
    double m = R_NegInf;
    for (int j = 0; j < k; ++j) {
      const double v = logL(i, j) + log_pi[j];
      if (v > m) m = v;
      row_post[j] = v;
    }

    double rs = 0.0;
    for (int j = 0; j < k; ++j) {
      const double v = row_post[j];
      if (R_finite(v)) {
        row_post[j] = std::exp(v - m);
        rs += row_post[j];
      } else {
        row_post[j] = 0.0;
      }
    }

    const double inv_rs = 1.0 / rs;
    for (int j = 0; j < k; ++j) {
      const double pij = row_post[j] * inv_rs;
      post_colsum[j] += pij;
    }
    acc += m + std::log(rs);
  }

  return List::create(
    _["post_colsum"] = post_colsum,
    _["ll"] = acc / static_cast<double>(n)
  );
}

// Compute mean log-likelihood given logL, row_max, and pi.
// [[Rcpp::export]]
double mixloglik_from_logL_cpp(const NumericMatrix& logL,
                               const NumericVector& row_max,
                               const NumericVector& pi) {
  const int n = logL.nrow();
  const int k = logL.ncol();

  double acc = 0.0;
  for (int i = 0; i < n; ++i) {
    const double m = row_max[i];
    double dens = 0.0;
    for (int j = 0; j < k; ++j) {
      dens += std::exp(logL(i, j) - m) * pi[j];
    }
    acc += m + std::log(dens);
  }
  return acc / static_cast<double>(n);
}

// [[Rcpp::export]]
List apch_alloc_core_cpp(const IntegerVector& Kvec) {
  const int p = Kvec.size();
  double Ld = 1.0;
  for (int j = 0; j < p; ++j) Ld *= static_cast<double>(Kvec[j]);
  const int L = static_cast<int>(Ld);

  IntegerMatrix alloc_mat(L, p);
  IntegerVector alloc_code(L);

  for (int l = 0; l < L; ++l) {
    int x = l;
    int code = 0;
    for (int j = 0; j < p; ++j) {
      const int Kj = Kvec[j];
      const int a = x % Kj;
      x /= Kj;
      alloc_mat(l, j) = a;
      code = (code << 1) | (a > 0 ? 1 : 0);
    }
    alloc_code[l] = code;
  }

  return List::create(
    Named("alloc_mat") = alloc_mat,
    Named("alloc_code") = alloc_code
  );
}

// [[Rcpp::export]]
List apch_local_prune_eval_cpp(
    const IntegerMatrix& alloc_mat,
    const IntegerVector& alloc_code,
    const NumericMatrix& alloc_logw_cond,
    const NumericVector& alloc_post_colsum,
    const NumericVector& prior_q,
    const bool use_prior,
    const double prior_frac,
    const int n_effect,
    const double prior_alpha_min,
    const double prior_alpha_cap,
    const double prior_alpha_cap_null,
    const double eff_n_thresh,
    const double eff_over_alpha_thresh,
    const double cond_weight_thresh,
    const IntegerVector& prune_codes
) {
  const int L = alloc_mat.nrow();
  const int p = alloc_mat.ncol();
  const int Rm = alloc_logw_cond.ncol();
  if (alloc_code.size() != L || alloc_post_colsum.size() != L) {
    stop("alloc_code/alloc_post_colsum length mismatch with alloc_mat rows.");
  }
  if (alloc_logw_cond.nrow() != L) {
    stop("alloc_logw_cond nrow mismatch with alloc_mat rows.");
  }
  if (use_prior && prior_q.size() != L) {
    stop("prior_q length mismatch with alloc_mat rows.");
  }

  std::vector<unsigned char> prune_code_flag(std::max(Rm, 1), 0);
  for (int i = 0; i < prune_codes.size(); ++i) {
    const int c = prune_codes[i];
    if (c >= 0 && c < Rm) prune_code_flag[c] = 1;
  }
  const double log_cond_weight_thresh = std::log(cond_weight_thresh);

  LogicalVector allowed(L, true);
  LogicalVector ratio_active(L, false);
  LogicalVector pruned(L, false);
  NumericVector logw(L, R_NegInf);
  NumericVector cond_weight(L, 0.0);
  NumericVector alpha(L, 0.0);
  NumericVector eff_n(L);
  NumericVector eff_over_alpha(L, R_PosInf);

  for (int l = 0; l < L; ++l) {
    const int code = alloc_code[l];
    if (code < 0 || code >= Rm) {
      stop("alloc_code value out of range for alloc_logw_cond columns.");
    }
    const double eff_n_l = alloc_post_colsum[l];
    eff_n[l] = eff_n_l;

    const double logw_l = alloc_logw_cond(l, code);
    logw[l] = logw_l;
    cond_weight[l] = std::isfinite(logw_l) ? std::exp(logw_l) : 0.0;

    if (use_prior) {
      double alpha_l = prior_frac * static_cast<double>(n_effect) * prior_q[l];
      if (prior_alpha_min > 0.0) alpha_l = std::max(alpha_l, prior_alpha_min);

      bool is_null = true;
      for (int j = 0; j < p; ++j) {
        if (alloc_mat(l, j) > 0) {
          is_null = false;
          break;
        }
      }
      if (is_null) {
        alpha_l = std::min(alpha_l, prior_alpha_cap_null);
      } else {
        alpha_l = std::min(alpha_l, prior_alpha_cap);
      }
      alpha[l] = alpha_l;

      const bool ratio_ok = (alpha_l > 0.0 && std::isfinite(alpha_l));
      ratio_active[l] = ratio_ok;
      eff_over_alpha[l] = ratio_ok ? (eff_n_l / alpha_l) : R_PosInf;
    }

    if (!prune_code_flag[code]) {
      allowed[l] = true;
      continue;
    }

    const bool prune_tinyw = (logw_l < log_cond_weight_thresh);
    const bool prune_eff = ratio_active[l] &&
      (eff_n_l < eff_n_thresh) &&
      (eff_over_alpha[l] < eff_over_alpha_thresh);
    const bool prune_l = prune_tinyw || prune_eff;
    pruned[l] = prune_l;
    allowed[l] = !prune_l;
  }

  return List::create(
    _["allowed"] = allowed,
    _["pruned"] = pruned,
    _["logw"] = logw,
    _["cond_weight"] = cond_weight,
    _["alpha"] = alpha,
    _["eff_n"] = eff_n,
    _["ratio_active"] = ratio_active,
    _["eff_over_alpha"] = eff_over_alpha
  );
}

// [[Rcpp::export]]
NumericMatrix apch_build_apchb_alloc_logw_cond_cpp(
    const IntegerMatrix& alloc_mat,
    const IntegerVector& alloc_code,
    const List& pi_block
) {
  const int L = alloc_mat.nrow();
  const int p = alloc_mat.ncol();
  const int Rm = 1 << p;
  const std::vector< std::vector<double> > pi = list_to_numvecs(pi_block);
  NumericMatrix out(L, Rm);
  std::fill(out.begin(), out.end(), R_NegInf);

  std::vector< std::vector<int> > ids_by_code(Rm);
  for (int l = 0; l < L; ++l) ids_by_code[alloc_code[l]].push_back(l);

  for (int c = 0; c < Rm; ++c) {
    const std::vector<int>& ids = ids_by_code[c];
    if (ids.empty()) continue;
    if (c == 0) {
      for (int id : ids) out(id, 0) = 0.0;
      continue;
    }
    const std::vector<int> active_idx = active_idx_from_code(c, p);
    std::vector<double> logw(ids.size(), 0.0);
    for (std::size_t ii = 0; ii < ids.size(); ++ii) {
      const int l = ids[ii];
      double lp = 0.0;
      for (int j : active_idx) {
        const int a = alloc_mat(l, j);
        const std::vector<double>& pi_j = pi[j];
        double q = (a >= 1 && a <= static_cast<int>(pi_j.size())) ? pi_j[a - 1] : 0.0;
        q = std::max(q, 1e-15);
        lp += std::log(q);
      }
      logw[ii] = lp;
    }
    const std::vector<double> logw_norm = normalize_logweights(logw);
    for (std::size_t ii = 0; ii < ids.size(); ++ii) out(ids[ii], c) = logw_norm[ii];
  }

  return out;
}

// [[Rcpp::export]]
List apch_build_apchs_alloc_logw_cond_cpp(
    const IntegerVector& alloc_code,
    const NumericVector& alloc_prior,
    const int p
) {
  const int L = alloc_code.size();
  const int Rm = 1 << p;
  NumericVector config_prior_m(Rm);
  for (int l = 0; l < L; ++l) config_prior_m[alloc_code[l]] += alloc_prior[l];

  NumericMatrix alloc_logw_cond(L, Rm);
  std::fill(alloc_logw_cond.begin(), alloc_logw_cond.end(), R_NegInf);

  for (int l = 0; l < L; ++l) {
    const int c = alloc_code[l];
    const double den = config_prior_m[c];
    if (den > 0) alloc_logw_cond(l, c) = std::log(alloc_prior[l] / den);
  }

  return List::create(
    Named("alloc_logw_cond") = alloc_logw_cond,
    Named("config_prior_m") = config_prior_m
  );
}

// [[Rcpp::export]]
NumericMatrix apch_alloc_posterior_from_config_prior_cpp(
    const NumericMatrix& log_alloc_mat,
    const IntegerVector& alloc_code,
    const NumericMatrix& alloc_logw_cond,
    const IntegerVector& config_code,
    const NumericVector& config_prior
) {
  const int n = log_alloc_mat.nrow();
  const int L = log_alloc_mat.ncol();
  int Rfull = 0;
  for (int r = 0; r < config_code.size(); ++r) Rfull = std::max(Rfull, config_code[r] + 1);

  std::vector<double> log_config_prior(Rfull, R_NegInf);
  for (int r = 0; r < config_code.size(); ++r) {
    if (config_prior[r] > 0) log_config_prior[config_code[r]] = std::log(config_prior[r]);
  }

  std::vector<double> log_prior_alloc(L, R_NegInf);
  for (int l = 0; l < L; ++l) {
    const int c = alloc_code[l];
    const double lp = (c < Rfull ? log_config_prior[c] : R_NegInf);
    const double lw = alloc_logw_cond(l, c);
    log_prior_alloc[l] = (std::isfinite(lp) && std::isfinite(lw)) ? lp + lw : R_NegInf;
  }

  NumericMatrix post_prob(n, L);
  struct AllocPosteriorWorker : public Worker {
    const RMatrix<double> log_alloc_mat;
    const std::vector<double>& log_prior_alloc;
    RMatrix<double> post_prob;

    AllocPosteriorWorker(
        const NumericMatrix& log_alloc_mat_,
        const std::vector<double>& log_prior_alloc_,
        NumericMatrix& post_prob_
    ) : log_alloc_mat(log_alloc_mat_),
        log_prior_alloc(log_prior_alloc_),
        post_prob(post_prob_) {}

    void operator()(std::size_t begin, std::size_t end) {
      const int L = log_alloc_mat.ncol();
      for (std::size_t i = begin; i < end; ++i) {
        double m = R_NegInf;
        for (int l = 0; l < L; ++l) {
          const double v = log_alloc_mat(i, l) + log_prior_alloc[l];
          post_prob(i, l) = v;
          if (std::isfinite(v) && v > m) m = v;
        }
        double s = 0.0;
        for (int l = 0; l < L; ++l) {
          const double v = post_prob(i, l);
          if (std::isfinite(v)) {
            post_prob(i, l) = std::exp(v - m);
            s += post_prob(i, l);
          } else {
            post_prob(i, l) = 0.0;
          }
        }
        const double inv_s = 1.0 / s;
        for (int l = 0; l < L; ++l) post_prob(i, l) *= inv_s;
      }
    }
  };

  AllocPosteriorWorker worker(log_alloc_mat, log_prior_alloc, post_prob);
  if (n > 1) {
    parallelFor(0, static_cast<std::size_t>(n), worker, static_cast<std::size_t>(64));
  } else {
    worker(0, static_cast<std::size_t>(n));
  }

  return post_prob;
}
