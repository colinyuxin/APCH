// calc_density.cpp
#include "calc_density.h"
#include "apch_blocks.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <RcppParallel.h>

using namespace RcppParallel;

// -----------------------------------------------------------------------------
// log-likelihood for normal priors (row-parallel), shared noise correlation
// -----------------------------------------------------------------------------

// Evaluate only on the observed sub-dimensions O_i
struct LogMatNormalWorker : public RcppParallel::Worker {
  const NumericMatrix &Bhat;
  const NumericMatrix &S;
  const NumericMatrix &C_cor;
  const RMatrix<int> alloc_mat;
  const std::vector< std::vector<double> > &grids;
  RowMat &logd;
  int p, L;

  LogMatNormalWorker(const NumericMatrix &Bhat_,
                     const NumericMatrix &S_,
                     const NumericMatrix &C_cor_,
                     const IntegerMatrix &alloc_mat_,
                     const std::vector< std::vector<double> > &grids_,
                     RowMat &logd_)
    : Bhat(Bhat_), S(S_), C_cor(C_cor_),
      alloc_mat(alloc_mat_), grids(grids_),
      logd(logd_), p(Bhat_.ncol()),
      L(alloc_mat_.nrow()) {}

  void operator()(std::size_t begin, std::size_t end) {
    const int p_local = p;
    const int L_local = L;

    for (std::size_t ii = begin; ii < end; ++ii) {
      int i = static_cast<int>(ii);

      std::vector<int> O;
      O.reserve(p_local);
      for (int j = 0; j < p_local; ++j) {
        double x = Bhat(i,j), s = S(i,j);
        if (R_finite(x) && R_finite(s)) O.push_back(j);
      }
      if (O.empty()) {
        for (int l = 0; l < L_local; ++l) logd(i,l) = 0.0;
        continue;
      }
      const int o = (int)O.size();

      Eigen::MatrixXd Sigma_OO(o, o);
      for (int a = 0; a < o; ++a) {
        for (int b = 0; b <= a; ++b) {
          int ja = O[a], jb = O[b];
          double val = S(i,ja) * S(i,jb) * C_cor(ja,jb);
          Sigma_OO(a,b) = val;
          Sigma_OO(b,a) = val;
        }
      }
      Eigen::VectorXd x_O(o);
      for (int a = 0; a < o; ++a) x_O(a) = Bhat(i, O[a]);

      for (int l = 0; l < L_local; ++l) {
        Eigen::MatrixXd Sigma = Sigma_OO;
        Eigen::VectorXd base_diag = Sigma_OO.diagonal();
        for (int a = 0; a < o; ++a) {
          int j = O[a];
          int g = alloc_mat(l, j);
          double s2 = (g == 0 ? 0.0 : grids[j][g] * grids[j][g]);
          Sigma(a,a) = base_diag(a) + s2;
        }
        logd(i,l) = ldmvnorm0(x_O, Sigma);
      }
    }
  }
};

// Shared-C allocation log-likelihood.
// Fast path triggers when all observations are finite and S is constant across
// effects. In that case we precompute the Cholesky factor for each allocation,
// then reuse it across all effects.
RowMat logmat_normal(const NumericMatrix &Bhat,
                     const NumericMatrix &S,
                     const NumericMatrix &C_cor,
                     const IntegerMatrix &alloc_mat,
                     const std::vector< std::vector<double> > &grids,
                     bool use_parallel) {
  const int n = Bhat.nrow();
  const int p = Bhat.ncol();
  const int L = alloc_mat.nrow();

  bool full_obs = true;
  bool same_S = true;
  for (int i = 0; i < n && (full_obs || same_S); ++i) {
    for (int j = 0; j < p; ++j) {
      const double x = Bhat(i, j);
      const double s = S(i, j);
      if (!R_finite(x) || !R_finite(s)) {
        full_obs = false;
        break;
      }
      if (i > 0 && std::abs(s - S(0, j)) > 1e-12) {
        same_S = false;
      }
    }
  }

  if (full_obs && same_S) {
    Eigen::MatrixXd Sigma_base(p, p);
    for (int a = 0; a < p; ++a) {
      for (int b = 0; b <= a; ++b) {
        const double val = S(0, a) * S(0, b) * C_cor(a, b);
        Sigma_base(a, b) = val;
        Sigma_base(b, a) = val;
      }
    }

    std::vector<Eigen::MatrixXd> L_list(L);
    std::vector<double> logdet_list(L, R_NegInf);
    std::vector<char> ok(L, 0);
    for (int l = 0; l < L; ++l) {
      Eigen::MatrixXd Sigma = Sigma_base;
      for (int j = 0; j < p; ++j) {
        int g = alloc_mat(l, j);
        double s2 = (g == 0 ? 0.0 : grids[j][g] * grids[j][g]);
        Sigma(j, j) += s2;
      }
      Eigen::LLT<Eigen::MatrixXd> chol(Sigma);
      if (chol.info() != Eigen::Success) {
        ok[l] = 0;
        continue;
      }
      Eigen::MatrixXd Lm = chol.matrixL();
      L_list[l] = Lm;
      logdet_list[l] = 2.0 * Lm.diagonal().array().log().sum();
      ok[l] = 1;
    }

    struct SharedCachedWorker : public RcppParallel::Worker {
      const NumericMatrix &Bhat;
      const std::vector<Eigen::MatrixXd> &L_list;
      const std::vector<double> &logdet_list;
      const std::vector<char> &ok;
      RowMat &logd;
      int p, L;

      SharedCachedWorker(const NumericMatrix &Bhat_,
                         const std::vector<Eigen::MatrixXd> &L_list_,
                         const std::vector<double> &logdet_list_,
                         const std::vector<char> &ok_,
                         RowMat &logd_,
                         int L_)
        : Bhat(Bhat_), L_list(L_list_),
          logdet_list(logdet_list_), ok(ok_),
          logd(logd_), p(Bhat_.ncol()), L(L_) {}

      void operator()(std::size_t begin, std::size_t end) {
        const double LOG2PI = std::log(2.0 * M_PI);
        Eigen::VectorXd x(p);
        for (std::size_t ii = begin; ii < end; ++ii) {
          int i = static_cast<int>(ii);
          for (int j = 0; j < p; ++j) x(j) = Bhat(i, j);
          for (int l = 0; l < L; ++l) {
            if (!ok[l]) {
              logd(i, l) = R_NegInf;
              continue;
            }
            const Eigen::MatrixXd &Lm = L_list[l];
            Eigen::VectorXd y = Lm.triangularView<Eigen::Lower>().solve(x);
            double quad = y.squaredNorm();
            logd(i, l) = -0.5 * (p * LOG2PI + logdet_list[l] + quad);
          }
        }
      }
    };

    RowMat logd(n, L);
    SharedCachedWorker worker(Bhat, L_list, logdet_list, ok, logd, L);
    if (n > 1 && use_parallel)
      RcppParallel::parallelFor(0, (size_t)n, worker, (size_t)512);
    else
      worker(0, (size_t)n);
    return logd;
  }

  RowMat logd(n, L);
  LogMatNormalWorker worker(Bhat, S, C_cor, alloc_mat, grids, logd);
  if (n > 1 && use_parallel)
    RcppParallel::parallelFor(0, (size_t)n, worker, (size_t)512);
  else
    worker(0, (size_t)n);

  return logd;
}

// -----------------------------------------------------------------------------
// Combine blockwise log-allocation matrices into a full log-allocation matrix.
// This is exact when the noise covariance is block-diagonal: logliks add.
// Columns follow the allocation order given by alloc_mat (0-based slab indices).
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
NumericMatrix apch_combine_log_alloc_blocks_cpp(
    const List& log_alloc_blocks,
    const List& blocks,
    const IntegerVector& Kvec,
    const IntegerMatrix& alloc_mat
) {
  const int B = log_alloc_blocks.size();
  if (B == 0) return NumericMatrix(0, 0);
  const NumericMatrix first = log_alloc_blocks[0];
  const int n = first.nrow();
  const int L = alloc_mat.nrow();
  const int p = alloc_mat.ncol();
  if (p != Kvec.size()) {
    stop("Kvec length must match alloc_mat columns.");
  }

  // precompute block indices and multipliers
  std::vector< std::vector<int> > block_idx(B);
  std::vector< std::vector<int> > block_mult(B);
  for (int b = 0; b < B; ++b) {
    IntegerVector idx = blocks[b];
    const int m = idx.size();
    block_idx[b].resize(m);
    block_mult[b].resize(m);
    int mult = 1;
    for (int t = 0; t < m; ++t) {
      int j = idx[t] - 1; // 1-based to 0-based
      if (j < 0 || j >= p) stop("block index out of range.");
      block_idx[b][t] = j;
      block_mult[b][t] = mult;
      mult *= Kvec[j];
    }
  }

  // precompute code per block for each allocation
  std::vector< std::vector<int> > codes(B, std::vector<int>(L));
  for (int b = 0; b < B; ++b) {
    const int m = block_idx[b].size();
    for (int l = 0; l < L; ++l) {
      int code = 0;
      for (int t = 0; t < m; ++t) {
        const int j = block_idx[b][t];
        code += alloc_mat(l, j) * block_mult[b][t];
      }
      codes[b][l] = code;
    }
  }

  RowMat out(n, L);
  out.setZero();
  for (int b = 0; b < B; ++b) {
    NumericMatrix mat = log_alloc_blocks[b];
    const int Lb = mat.ncol();
    for (int i = 0; i < n; ++i) {
      double* out_row = out.row(i).data();
      for (int l = 0; l < L; ++l) {
        const int cb = codes[b][l];
        if (cb >= 0 && cb < Lb) {
          out_row[l] += mat(i, cb);
        } else {
          out_row[l] += R_NegInf;
        }
      }
    }
  }

  return wrap(out);
}

// ============================================================================
// per-effect C_i supplied as List — row-parallel
// ============================================================================

// Normal prior with per-effect noise correlation C_i. When C_i is block
// diagonal on the observed coordinates, the log-likelihood is the sum over
// blocks.
struct LogMatNormalXwasWorker : public RcppParallel::Worker {
  const NumericMatrix &Bhat;
  const NumericMatrix &S;
  const std::vector<NumericMatrix> &C_list;
  const RMatrix<int> alloc_mat;
  const std::vector< std::vector<double> > &grids;
  RowMat &logd;
  int p, L;
  static constexpr double block_tol = 1e-4;

  LogMatNormalXwasWorker(const NumericMatrix &Bhat_,
                         const NumericMatrix &S_,
                         const std::vector<NumericMatrix> &C_list_,
                         const IntegerMatrix &alloc_mat_,
                         const std::vector< std::vector<double> > &grids_,
                         RowMat &logd_)
    : Bhat(Bhat_), S(S_), C_list(C_list_),
      alloc_mat(alloc_mat_), grids(grids_),
      logd(logd_), p(Bhat_.ncol()),
      L(alloc_mat_.nrow()) {}

  double eval_block_loglik(const Eigen::VectorXd& x_b,
                           const Eigen::MatrixXd& Sigma_b,
                           const std::vector<int>& orig_idx,
                           const int l) const {
    const int k = x_b.size();
    Eigen::MatrixXd Sigma = Sigma_b;
    Eigen::VectorXd base_diag = Sigma_b.diagonal();
    for (int a = 0; a < k; ++a) {
      const int j = orig_idx[a];
      const int g = alloc_mat(l, j);
      const double s2 = (g == 0 ? 0.0 : grids[j][g] * grids[j][g]);
      Sigma(a, a) = base_diag(a) + s2;
    }
    return ldmvnorm0(x_b, Sigma);
  }

  double eval_block_partition_loglik(
      const Eigen::VectorXd& x_O,
      const Eigen::MatrixXd& Sigma_OO,
      const std::vector<int>& O,
      const std::vector< std::vector<int> >& blocks,
      const int l
  ) const {
    double out = 0.0;
    for (const std::vector<int>& block : blocks) {
      const int k = static_cast<int>(block.size());
      Eigen::VectorXd x_b(k);
      Eigen::MatrixXd Sigma_b(k, k);
      std::vector<int> orig_idx(k);
      for (int a = 0; a < k; ++a) {
        const int la = block[a];
        x_b(a) = x_O(la);
        orig_idx[a] = O[la];
        for (int b = 0; b <= a; ++b) {
          const int lb = block[b];
          Sigma_b(a, b) = Sigma_OO(la, lb);
          Sigma_b(b, a) = Sigma_b(a, b);
        }
      }
      const double ll = eval_block_loglik(x_b, Sigma_b, orig_idx, l);
      if (!std::isfinite(ll)) return R_NegInf;
      out += ll;
    }
    return out;
  }

  void operator()(std::size_t begin, std::size_t end) {
    const int p_local = p;
    const int L_local = L;

    for (std::size_t ii = begin; ii < end; ++ii) {
      int i = static_cast<int>(ii);
      const NumericMatrix &C_i = C_list[i];

      std::vector<int> O;
      O.reserve(p_local);
      for (int j = 0; j < p_local; ++j) {
        double x = Bhat(i,j), s = S(i,j);
        if (R_finite(x) && R_finite(s)) O.push_back(j);
      }
      if (O.empty()) {
        for (int l = 0; l < L_local; ++l) logd(i,l) = 0.0;
        continue;
      }
      const int o = (int)O.size();

      Eigen::MatrixXd Sigma_OO(o, o);
      for (int a = 0; a < o; ++a) {
        for (int b = 0; b <= a; ++b) {
          int ja = O[a], jb = O[b];
          double val = S(i,ja) * S(i,jb) * C_i(ja,jb);
          Sigma_OO(a,b) = val;
          Sigma_OO(b,a) = val;
        }
      }
      Eigen::VectorXd x_O(o);
      for (int a = 0; a < o; ++a) x_O(a) = Bhat(i, O[a]);

      const std::vector< std::vector<int> > blocks =
        apch_blocks_from_numeric_matrix_subset(C_i, O, block_tol);
      if (blocks.size() > 1) {
        for (int l = 0; l < L_local; ++l) {
          logd(i, l) = eval_block_partition_loglik(x_O, Sigma_OO, O, blocks, l);
        }
        continue;
      }

      for (int l = 0; l < L_local; ++l) {
        Eigen::MatrixXd Sigma = Sigma_OO;
        Eigen::VectorXd base_diag = Sigma_OO.diagonal();
        for (int a = 0; a < o; ++a) {
          int j = O[a];
          int g = alloc_mat(l, j);
          double s2 = (g == 0 ? 0.0 : grids[j][g] * grids[j][g]);
          Sigma(a,a) = base_diag(a) + s2;
        }
        logd(i,l) = ldmvnorm0(x_O, Sigma);
      }
    }
  }
};

RowMat logmat_normal_list(const NumericMatrix &Bhat,
                          const NumericMatrix &S,
                          const List &C_list,
                          const IntegerMatrix &alloc_mat,
                          const std::vector< std::vector<double> > &grids,
                          bool use_parallel) {
  const int n = Bhat.nrow();
  const int L = alloc_mat.nrow();

  RowMat logd(n, L);

  std::vector<NumericMatrix> C_list_mat(n);
  for (int i = 0; i < n; ++i) {
    C_list_mat[i] = Rcpp::as<NumericMatrix>(C_list[i]);
  }

  LogMatNormalXwasWorker worker(Bhat, S, C_list_mat, alloc_mat, grids, logd);
  if (n > 1 && use_parallel)
    RcppParallel::parallelFor(0, (size_t)n, worker, (size_t)512);
  else
    worker(0, (size_t)n);

  return logd;
}

// [[Rcpp::export]]
NumericVector row_max_cpp(const NumericMatrix& x) {
  return row_max_vec(x);
}

// [[Rcpp::export]]
Rcpp::List normalize_logpost_cpp(const NumericMatrix& log_post) {
  const int n = log_post.nrow();
  const int p = log_post.ncol();
  NumericMatrix post_prob(n, p);
  NumericVector row_max(n);
  double ll_sum = 0.0;

  for (int i = 0; i < n; ++i) {
    double m = row_max_at(log_post, i);
    row_max[i] = m;
    double rs = 0.0;
    for (int j = 0; j < p; ++j) {
      const double v = std::exp(log_post(i, j) - m);
      post_prob(i, j) = v;
      rs += v;
    }
    const double inv_rs = 1.0 / rs;
    for (int j = 0; j < p; ++j) {
      post_prob(i, j) *= inv_rs;
    }
    ll_sum += m + std::log(rs);
  }

  return Rcpp::List::create(
    Rcpp::Named("post_prob") = post_prob,
    Rcpp::Named("ll") = ll_sum / n
  );
}

// [[Rcpp::export]]
NumericVector row_logsumexp_cpp(const NumericMatrix& log_mat) {
  const int n = log_mat.nrow();
  const int p = log_mat.ncol();
  NumericVector out(n);
  for (int i = 0; i < n; ++i) {
    double m = row_max_at(log_mat, i);
    double rs = 0.0;
    for (int j = 0; j < p; ++j) {
      rs += std::exp(log_mat(i, j) - m);
    }
    out[i] = m + std::log(rs);
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::List loglik_to_lik_cpp(const NumericMatrix& log_mat) {
  const int n = log_mat.nrow();
  const int p = log_mat.ncol();
  NumericMatrix L(n, p);
  NumericVector row_max(n);
  for (int i = 0; i < n; ++i) {
    double m = row_max_at(log_mat, i);
    row_max[i] = m;
    for (int j = 0; j < p; ++j) {
      L(i, j) = std::exp(log_mat(i, j) - m);
    }
  }
  return Rcpp::List::create(
    Rcpp::Named("L") = L,
    Rcpp::Named("row_max") = row_max
  );
}
