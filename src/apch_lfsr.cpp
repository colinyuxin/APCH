// apch_lfsr.cpp
// [[Rcpp::depends(Rcpp)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppParallel)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include "apch_blocks.h"

using namespace Rcpp;
using namespace RcppParallel;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::LLT;

static std::vector< std::vector<int> > subset_alloc_idx_from_alloc(
    const IntegerMatrix& subset_vec_mat,
    const IntegerMatrix& alloc_mat
) {
  const int M = subset_vec_mat.nrow();
  const int p = subset_vec_mat.ncol();
  const int L = alloc_mat.nrow();

  std::vector< std::vector<int> > subset_alloc_idx(M);
  for (int m = 0; m < M; ++m) {
    std::vector<int>& keep = subset_alloc_idx[m];
    keep.reserve(L);
    for (int l = 0; l < L; ++l) {
      bool has_null = false;
      for (int j = 0; j < p; ++j) {
        if (subset_vec_mat(m, j) > 0 && alloc_mat(l, j) == 0) {
          has_null = true;
          break;
        }
      }
      if (!has_null) keep.push_back(l);
    }
  }
  return subset_alloc_idx;
}

static std::vector< std::vector<int> > subset_idx_from_subset_vec(
    const IntegerMatrix& subset_vec_mat
) {
  const int M = subset_vec_mat.nrow();
  const int p = subset_vec_mat.ncol();
  std::vector< std::vector<int> > subset_idx(M);
  for (int m = 0; m < M; ++m) {
    for (int j = 0; j < p; ++j) {
      if (subset_vec_mat(m, j) > 0) subset_idx[m].push_back(j);
    }
  }
  return subset_idx;
}

static bool subset_row_has_nonfinite(
    const NumericMatrix& z,
    const int i,
    const std::vector<int>& U
) {
  for (int idx = 0; idx < static_cast<int>(U.size()); ++idx) {
    if (!R_finite(z(i, U[idx]))) return true;
  }
  return false;
}

static std::vector< std::vector< std::vector<int> > > subset_block_pos_from_subset_and_blocks(
    const IntegerMatrix& subset_vec_mat,
    const std::vector< std::vector<int> >& blocks
) {
  const int M = subset_vec_mat.nrow();
  const int B = static_cast<int>(blocks.size());
  std::vector< std::vector< std::vector<int> > > subset_block_pos(
    M,
    std::vector< std::vector<int> >(B)
  );
  for (int m = 0; m < M; ++m) {
    for (int b = 0; b < B; ++b) {
      const std::vector<int>& idx = blocks[b];
      std::vector<int>& pos = subset_block_pos[m][b];
      pos.reserve(idx.size());
      for (int t = 0; t < static_cast<int>(idx.size()); ++t) {
        if (subset_vec_mat(m, idx[t]) > 0) pos.push_back(t);
      }
    }
  }
  return subset_block_pos;
}

struct SharedCacheParsed {
  int B = 0;
  std::vector< std::vector<int> > blocks;
  std::vector<int> k_block;
  std::vector<LogicalVector> ok_block;
  std::vector<NumericVector> A_block;
  std::vector<NumericVector> V_block;
};

static SharedCacheParsed parse_shared_cache(const List& shared_cache) {
  List blocks_r = shared_cache["blocks"];
  List ok_r = shared_cache["ok"];
  List A_r = shared_cache["A"];
  List V_r = shared_cache["V"];

  SharedCacheParsed parsed;
  parsed.B = blocks_r.size();
  parsed.blocks.resize(parsed.B);
  parsed.k_block.resize(parsed.B, 0);
  parsed.ok_block.resize(parsed.B);
  parsed.A_block.resize(parsed.B);
  parsed.V_block.resize(parsed.B);

  for (int b = 0; b < parsed.B; ++b) {
    IntegerVector idx1 = blocks_r[b];
    const int k = idx1.size();
    parsed.blocks[b].resize(k);
    for (int t = 0; t < k; ++t) parsed.blocks[b][t] = idx1[t] - 1;
    parsed.k_block[b] = k;
    parsed.ok_block[b] = ok_r[b];
    parsed.A_block[b] = A_r[b];
    parsed.V_block[b] = V_r[b];
  }
  return parsed;
}

struct SharedSubsetMetaParsed {
  std::vector< std::vector<int> > subset_idx;
  std::vector< std::vector<int> > subset_alloc_idx;
  std::vector< std::vector< std::vector<int> > > subset_block_pos;
  std::vector< std::vector<int> > sid_active_blocks;
};

static List build_shared_subset_meta_list(
    const IntegerMatrix& subset_vec_mat,
    const IntegerMatrix& alloc_mat,
    const std::vector< std::vector<int> >& blocks
) {
  const int M = subset_vec_mat.nrow();
  const int B = static_cast<int>(blocks.size());

  std::vector< std::vector<int> > subset_idx = subset_idx_from_subset_vec(subset_vec_mat);
  std::vector< std::vector<int> > subset_alloc_idx =
    subset_alloc_idx_from_alloc(subset_vec_mat, alloc_mat);
  std::vector< std::vector< std::vector<int> > > subset_block_pos =
    subset_block_pos_from_subset_and_blocks(subset_vec_mat, blocks);

  std::vector< std::vector<int> > sid_active_blocks(M);
  for (int m = 0; m < M; ++m) {
    sid_active_blocks[m].reserve(B);
    for (int b = 0; b < B; ++b) {
      if (!subset_block_pos[m][b].empty()) sid_active_blocks[m].push_back(b);
    }
  }

  List subset_idx_r(M);
  List subset_alloc_idx_r(M);
  List subset_block_pos_r(M);
  List sid_active_blocks_r(M);

  for (int m = 0; m < M; ++m) {
    const std::vector<int>& idx_m = subset_idx[m];
    IntegerVector idx_r(idx_m.size());
    for (int i = 0; i < static_cast<int>(idx_m.size()); ++i) idx_r[i] = idx_m[i] + 1;
    subset_idx_r[m] = idx_r;

    const std::vector<int>& alloc_m = subset_alloc_idx[m];
    IntegerVector alloc_r(alloc_m.size());
    for (int i = 0; i < static_cast<int>(alloc_m.size()); ++i) alloc_r[i] = alloc_m[i] + 1;
    subset_alloc_idx_r[m] = alloc_r;

    List block_pos_m(B);
    for (int b = 0; b < B; ++b) {
      const std::vector<int>& pos_mb = subset_block_pos[m][b];
      IntegerVector pos_r(pos_mb.size());
      for (int i = 0; i < static_cast<int>(pos_mb.size()); ++i) pos_r[i] = pos_mb[i] + 1;
      block_pos_m[b] = pos_r;
    }
    subset_block_pos_r[m] = block_pos_m;

    const std::vector<int>& active_b = sid_active_blocks[m];
    IntegerVector active_r(active_b.size());
    for (int i = 0; i < static_cast<int>(active_b.size()); ++i) active_r[i] = active_b[i] + 1;
    sid_active_blocks_r[m] = active_r;
  }

  return List::create(
    _["subset_idx"] = subset_idx_r,
    _["subset_alloc_idx"] = subset_alloc_idx_r,
    _["subset_block_pos"] = subset_block_pos_r,
    _["sid_active_blocks"] = sid_active_blocks_r
  );
}

static SharedSubsetMetaParsed parse_shared_subset_meta(
    const List& subset_meta,
    const int B
) {
  List subset_idx_r = subset_meta["subset_idx"];
  List subset_alloc_idx_r = subset_meta["subset_alloc_idx"];
  List subset_block_pos_r = subset_meta["subset_block_pos"];
  List sid_active_blocks_r = subset_meta["sid_active_blocks"];

  const int M = subset_idx_r.size();
  SharedSubsetMetaParsed parsed;
  parsed.subset_idx.resize(M);
  parsed.subset_alloc_idx.resize(M);
  parsed.subset_block_pos.resize(M, std::vector< std::vector<int> >(B));
  parsed.sid_active_blocks.resize(M);

  for (int m = 0; m < M; ++m) {
    IntegerVector idx_r = subset_idx_r[m];
    parsed.subset_idx[m].resize(idx_r.size());
    for (int i = 0; i < idx_r.size(); ++i) parsed.subset_idx[m][i] = idx_r[i] - 1;

    IntegerVector alloc_r = subset_alloc_idx_r[m];
    parsed.subset_alloc_idx[m].resize(alloc_r.size());
    for (int i = 0; i < alloc_r.size(); ++i) parsed.subset_alloc_idx[m][i] = alloc_r[i] - 1;

    List block_pos_m = subset_block_pos_r[m];
    for (int b = 0; b < B; ++b) {
      IntegerVector pos_r = block_pos_m[b];
      parsed.subset_block_pos[m][b].resize(pos_r.size());
      for (int i = 0; i < pos_r.size(); ++i) parsed.subset_block_pos[m][b][i] = pos_r[i] - 1;
    }

    IntegerVector active_r = sid_active_blocks_r[m];
    parsed.sid_active_blocks[m].resize(active_r.size());
    for (int i = 0; i < active_r.size(); ++i) parsed.sid_active_blocks[m][i] = active_r[i] - 1;
  }

  return parsed;
}

static bool compute_block_prob_common(
    const VectorXd& mu_u,
    const MatrixXd& V_u,
    const VectorXd& t_u,
    Function& pmvnorm,
    Function& pbivnorm,
    const RObject& tvpack_alg,
    const int maxpts,
    const double abseps,
    const double releps,
    const double jitter,
    double& p_block
) {
  const int ku = mu_u.size();
  VectorXd mu_t = mu_u.array() * t_u.array();

  if (ku == 1) {
    const double v = V_u(0, 0);
    if (!R_finite(v) || v <= 0.0) return false;
    p_block = R::pnorm5(mu_t(0) / std::sqrt(v), 0.0, 1.0, 1, 0);
    return R_finite(p_block);
  }

  if (ku == 2) {
    const double v1 = V_u(0, 0);
    const double v2 = V_u(1, 1);
    if (!R_finite(v1) || !R_finite(v2) || v1 <= 0.0 || v2 <= 0.0) return false;
    const double sd1 = std::sqrt(v1);
    const double sd2 = std::sqrt(v2);
    double rho = V_u(0, 1) / (sd1 * sd2);
    rho = std::max(-1.0, std::min(1.0, rho));
    NumericVector res = pbivnorm(
      _["x"] = NumericVector::create(mu_t(0) / sd1),
      _["y"] = NumericVector::create(mu_t(1) / sd2),
      _["rho"] = NumericVector::create(rho),
      _["recycle"] = false
    );
    p_block = res[0];
    return R_finite(p_block);
  }

  MatrixXd sign_outer = t_u * t_u.transpose();
  MatrixXd V_t = V_u.cwiseProduct(sign_outer);
  V_t.diagonal().array() += jitter;

  NumericVector lower(ku, 0.0);
  NumericVector upper(ku, R_PosInf);
  NumericVector mean(ku);
  NumericMatrix sigma(ku, ku);
  for (int a = 0; a < ku; ++a) {
    mean[a] = mu_t(a);
    for (int c = 0; c < ku; ++c) sigma(a, c) = V_t(a, c);
  }

  if (ku == 3) {
    NumericVector res = pmvnorm(
      _["lower"] = lower,
      _["upper"] = upper,
      _["mean"] = mean,
      _["sigma"] = sigma,
      _["algorithm"] = tvpack_alg,
      _["keepAttr"] = false
    );
    p_block = res[0];
    return R_finite(p_block);
  }

  NumericVector res = pmvnorm(
    _["lower"] = lower,
    _["upper"] = upper,
    _["mean"] = mean,
    _["sigma"] = sigma,
    _["maxpts"] = maxpts,
    _["abseps"] = abseps,
    _["releps"] = releps,
    _["keepAttr"] = false
  );
  p_block = res[0];
  return R_finite(p_block);
}

struct LfsrDiagWorker : public Worker {
  const RMatrix<double> z;
  const RMatrix<double> e_diag;
  const RMatrix<int> alloc_mat;
  const RMatrix<double> alpha_mat;
  const std::vector< std::vector<double> >& grids;
  const std::vector< std::vector<int> >& subset_idx;
  const std::vector< std::vector<int> >& subset_alloc_idx;
  RMatrix<double> out;

  LfsrDiagWorker(
      const NumericMatrix& z_,
      const NumericMatrix& e_diag_,
      const IntegerMatrix& alloc_mat_,
      const NumericMatrix& alpha_mat_,
      const std::vector< std::vector<double> >& grids_,
      const std::vector< std::vector<int> >& subset_idx_,
      const std::vector< std::vector<int> >& subset_alloc_idx_,
      NumericMatrix& out_
  ) : z(z_),
      e_diag(e_diag_),
      alloc_mat(alloc_mat_),
      alpha_mat(alpha_mat_),
      grids(grids_),
      subset_idx(subset_idx_),
      subset_alloc_idx(subset_alloc_idx_),
      out(out_) {}

  void operator()(std::size_t begin, std::size_t end) {
    const int p = z.ncol();
    const int M = subset_idx.size();

    for (std::size_t i = begin; i < end; ++i) {
      std::vector< std::vector<double> > sign_prob(p);
      std::vector<int> bad_feature(p, 0);

      for (int j = 0; j < p; ++j) {
        const int Gj = grids[j].size();
        sign_prob[j].assign(Gj, NA_REAL);
        const double zval = z(i, j);
        if (!R_finite(zval)) {
          bad_feature[j] = 1;
          continue;
        }

        sign_prob[j][0] = 1.0;
        const double e = e_diag(i, j);
        if (!R_finite(e) || e <= 0.0) {
          continue;
        }
        const double t = (zval >= 0.0) ? 1.0 : -1.0;
        for (int g = 1; g < Gj; ++g) {
          const double sd = grids[j][g];
          const double d = sd * sd;
          if (!R_finite(d) || d <= 0.0) continue;
          const double ratio = t * zval * std::sqrt(d / (e * (e + d)));
          sign_prob[j][g] = R::pnorm5(ratio, 0.0, 1.0, 1, 0);
        }
      }

      for (int m = 0; m < M; ++m) {
        const std::vector<int>& U = subset_idx[m];
        bool bad = false;
        for (int idx = 0; idx < (int)U.size(); ++idx) {
          if (bad_feature[U[idx]]) {
            bad = true;
            break;
          }
        }
        if (bad) {
          out(i, m) = NA_REAL;
          continue;
        }

        double sum_prob = 0.0;
        const std::vector<int>& alloc_keep = subset_alloc_idx[m];
        for (int q = 0; q < (int)alloc_keep.size(); ++q) {
          const int l = alloc_keep[q];
          double prob_l = 1.0;
          for (int idx = 0; idx < (int)U.size(); ++idx) {
            const int j = U[idx];
            const int g = alloc_mat(l, j);
            const double pval = sign_prob[j][g];
            if (!R_finite(pval)) {
              prob_l = NA_REAL;
              break;
            }
            prob_l *= pval;
          }
          if (!R_finite(prob_l)) continue;
          sum_prob += alpha_mat(i, l) * prob_l;
        }

        out(i, m) = 1.0 - sum_prob;
      }
    }
  }
};

// [[Rcpp::export]]
NumericMatrix lfsr_diag_cpp(
    const NumericMatrix& z,
    const NumericMatrix& e_diag,
    const IntegerMatrix& subset_vec_mat,
    const IntegerMatrix& alloc_mat,
    const List& grid_list,
    const NumericMatrix& alpha_mat,
    const bool use_parallel = true,
    const int grain_size = 64
) {
  const int p = z.ncol();
  const int M = subset_vec_mat.nrow();

  std::vector< std::vector<double> > grids(p);
  for (int j = 0; j < p; ++j) {
    NumericVector g = grid_list[j];
    grids[j].assign(g.begin(), g.end());
  }

  std::vector< std::vector<int> > subset_idx = subset_idx_from_subset_vec(subset_vec_mat);
  std::vector< std::vector<int> > subset_alloc_idx =
    subset_alloc_idx_from_alloc(subset_vec_mat, alloc_mat);

  NumericMatrix out(z.nrow(), M);
  LfsrDiagWorker worker(
    z, e_diag, alloc_mat, alpha_mat,
    grids, subset_idx, subset_alloc_idx, out
  );
  if (use_parallel && z.nrow() > 1) {
    parallelFor(0, (std::size_t)z.nrow(), worker, (std::size_t)std::max(grain_size, 1));
  } else {
    worker(0, (std::size_t)z.nrow());
  }

  return out;
}

struct LfsrDiagIndexedWorker : public Worker {
  const RMatrix<double> z;
  const RMatrix<double> e_diag;
  const RMatrix<int> alloc_mat;
  const RMatrix<double> alpha_mat;
  const RVector<int> effect_idx;
  const std::vector< std::vector<double> >& grids;
  const std::vector< std::vector<int> >& subset_idx;
  const std::vector< std::vector<int> >& subset_alloc_idx;
  RMatrix<double> out;

  LfsrDiagIndexedWorker(
      const NumericMatrix& z_,
      const NumericMatrix& e_diag_,
      const IntegerMatrix& alloc_mat_,
      const NumericMatrix& alpha_mat_,
      const IntegerVector& effect_idx_,
      const std::vector< std::vector<double> >& grids_,
      const std::vector< std::vector<int> >& subset_idx_,
      const std::vector< std::vector<int> >& subset_alloc_idx_,
      NumericMatrix& out_
  ) : z(z_),
      e_diag(e_diag_),
      alloc_mat(alloc_mat_),
      alpha_mat(alpha_mat_),
      effect_idx(effect_idx_),
      grids(grids_),
      subset_idx(subset_idx_),
      subset_alloc_idx(subset_alloc_idx_),
      out(out_) {}

  void operator()(std::size_t begin, std::size_t end) {
    const int p = z.ncol();
    const int M = subset_idx.size();

    for (std::size_t r = begin; r < end; ++r) {
      const int i = effect_idx[r] - 1;
      std::vector< std::vector<double> > sign_prob(p);
      std::vector<int> bad_feature(p, 0);

      for (int j = 0; j < p; ++j) {
        const int Gj = grids[j].size();
        sign_prob[j].assign(Gj, NA_REAL);
        const double zval = z(i, j);
        if (!R_finite(zval)) {
          bad_feature[j] = 1;
          continue;
        }

        sign_prob[j][0] = 1.0;
        const double e = e_diag(i, j);
        if (!R_finite(e) || e <= 0.0) {
          continue;
        }
        const double t = (zval >= 0.0) ? 1.0 : -1.0;
        for (int g = 1; g < Gj; ++g) {
          const double sd = grids[j][g];
          const double d = sd * sd;
          if (!R_finite(d) || d <= 0.0) continue;
          const double ratio = t * zval * std::sqrt(d / (e * (e + d)));
          sign_prob[j][g] = R::pnorm5(ratio, 0.0, 1.0, 1, 0);
        }
      }

      for (int m = 0; m < M; ++m) {
        const std::vector<int>& U = subset_idx[m];
        bool bad = false;
        for (int idx = 0; idx < static_cast<int>(U.size()); ++idx) {
          if (bad_feature[U[idx]]) {
            bad = true;
            break;
          }
        }
        if (bad) {
          out(r, m) = NA_REAL;
          continue;
        }

        double sum_prob = 0.0;
        const std::vector<int>& alloc_keep = subset_alloc_idx[m];
        for (int q = 0; q < static_cast<int>(alloc_keep.size()); ++q) {
          const int l = alloc_keep[q];
          double prob_l = 1.0;
          for (int idx = 0; idx < static_cast<int>(U.size()); ++idx) {
            const int j = U[idx];
            const int g = alloc_mat(l, j);
            const double pval = sign_prob[j][g];
            if (!R_finite(pval)) {
              prob_l = NA_REAL;
              break;
            }
            prob_l *= pval;
          }
          if (!R_finite(prob_l)) continue;
          sum_prob += alpha_mat(i, l) * prob_l;
        }

        out(r, m) = 1.0 - sum_prob;
      }
    }
  }
};

// [[Rcpp::export]]
NumericMatrix lfsr_diag_indexed_cpp(
    const NumericMatrix& z,
    const NumericMatrix& e_diag,
    const IntegerVector& effect_idx,
    const IntegerMatrix& subset_vec_mat,
    const IntegerMatrix& alloc_mat,
    const List& grid_list,
    const NumericMatrix& alpha_mat,
    const bool use_parallel = true,
    const int grain_size = 64
) {
  const int p = z.ncol();
  const int M = subset_vec_mat.nrow();

  std::vector< std::vector<double> > grids(p);
  for (int j = 0; j < p; ++j) {
    NumericVector g = grid_list[j];
    grids[j].assign(g.begin(), g.end());
  }

  std::vector< std::vector<int> > subset_idx = subset_idx_from_subset_vec(subset_vec_mat);
  std::vector< std::vector<int> > subset_alloc_idx =
    subset_alloc_idx_from_alloc(subset_vec_mat, alloc_mat);

  NumericMatrix out(effect_idx.size(), M);
  LfsrDiagIndexedWorker worker(
    z, e_diag, alloc_mat, alpha_mat, effect_idx,
    grids, subset_idx, subset_alloc_idx, out
  );
  if (use_parallel && effect_idx.size() > 1) {
    parallelFor(0, static_cast<std::size_t>(effect_idx.size()), worker, static_cast<std::size_t>(std::max(grain_size, 1)));
  } else {
    worker(0, static_cast<std::size_t>(effect_idx.size()));
  }

  return out;
}

// [[Rcpp::export]]
NumericMatrix lfsr_nondiag_cpp(
    const NumericMatrix& z,
    const NumericMatrix& C_shared,
    const List& C_list,
    const bool use_shared,
    const IntegerMatrix& subset_vec_mat,
    const IntegerMatrix& alloc_mat,
    const List& grid_list,
    const NumericMatrix& alpha_mat,
    const int maxpts = 20000,
    const double abseps = 1e-6,
    const double releps = 1e-6,
    const double jitter = 1e-10,
    const double block_tol = 1e-12
) {
  const int n = z.nrow();
  const int p = z.ncol();
  const int M = subset_vec_mat.nrow();
  const int L = alloc_mat.nrow();

  std::vector< std::vector<double> > grids(p);
  for (int j = 0; j < p; ++j) {
    NumericVector g = grid_list[j];
    grids[j].assign(g.begin(), g.end());
  }

  std::vector< std::vector<int> > subset_idx = subset_idx_from_subset_vec(subset_vec_mat);
  std::vector< std::vector<int> > subset_alloc_idx =
    subset_alloc_idx_from_alloc(subset_vec_mat, alloc_mat);

  std::vector< std::vector<int> > blocks_shared;
  if (use_shared) {
    blocks_shared = apch_blocks_from_numeric_matrix(C_shared, block_tol);
  }
  std::vector<NumericMatrix> C_list_mat;
  if (!use_shared) {
    C_list_mat.resize(n);
    for (int i = 0; i < n; ++i) {
      C_list_mat[i] = as<NumericMatrix>(C_list[i]);
    }
  }

  Environment mvtnorm = Environment::namespace_env("mvtnorm");
  Function pmvnorm = mvtnorm["pmvnorm"];
  Function TVPACK = mvtnorm["TVPACK"];
  RObject tvpack_alg = TVPACK();
  Environment pbiv = Environment::namespace_env("pbivnorm");
  Function pbivnorm = pbiv["pbivnorm"];

  NumericMatrix out(n, M);

  if (use_shared) {
    const int B = static_cast<int>(blocks_shared.size());

    struct SharedBlockAllocCache {
      bool ok;
      MatrixXd A;
      MatrixXd V;
    };

    std::vector< std::vector< std::vector<int> > > subset_block_pos =
      subset_block_pos_from_subset_and_blocks(subset_vec_mat, blocks_shared);

    std::vector<MatrixXd> E_blocks(B);
    for (int b = 0; b < B; ++b) {
      const std::vector<int>& idx = blocks_shared[b];
      const int k = static_cast<int>(idx.size());
      MatrixXd E_b(k, k);
      for (int a = 0; a < k; ++a) {
        for (int c = 0; c < k; ++c) {
          E_b(a, c) = C_shared(idx[a], idx[c]);
        }
      }
      E_blocks[b] = E_b;
    }

    std::vector<SharedBlockAllocCache> cache(B * L);
    for (int b = 0; b < B; ++b) {
      const std::vector<int>& idx = blocks_shared[b];
      const int k = static_cast<int>(idx.size());
      const MatrixXd& E_b = E_blocks[b];

      for (int l = 0; l < L; ++l) {
        SharedBlockAllocCache& item = cache[b * L + l];

        VectorXd d_vec(k);
        for (int t = 0; t < k; ++t) {
          const int j = idx[t];
          const int g = alloc_mat(l, j);
          double d = 0.0;
          if (g > 0 && g < static_cast<int>(grids[j].size())) {
            const double sd = grids[j][g];
            d = sd * sd;
          }
          d_vec(t) = d;
        }

        MatrixXd K = E_b;
        K.diagonal() += d_vec;

        LLT<MatrixXd> chol(K);
        if (chol.info() != Eigen::Success) {
          item.ok = false;
          continue;
        }

        MatrixXd D = d_vec.asDiagonal();
        MatrixXd Kinv_D = chol.solve(D);
        item.ok = true;
        item.A = Kinv_D.transpose();
        item.V = D - D * Kinv_D;
      }
    }

    for (int i = 0; i < n; ++i) {
      for (int m = 0; m < M; ++m) {
        const std::vector<int>& U = subset_idx[m];
        if (U.empty()) {
          out(i, m) = NA_REAL;
          continue;
        }
        if (subset_row_has_nonfinite(z, i, U)) {
          out(i, m) = NA_REAL;
          continue;
        }

        double sum_prob = 0.0;

        const std::vector<int>& alloc_keep = subset_alloc_idx[m];
        for (int q = 0; q < static_cast<int>(alloc_keep.size()); ++q) {
          const int l = alloc_keep[q];
          double prob_l = 1.0;

          for (int b = 0; b < B; ++b) {
            const std::vector<int>& pos = subset_block_pos[m][b];
            if (pos.empty()) continue;
            const std::vector<int>& idx = blocks_shared[b];
            const SharedBlockAllocCache& item = cache[b * L + l];
            if (!item.ok) {
              prob_l = NA_REAL;
              break;
            }

            const int k = static_cast<int>(idx.size());
            VectorXd z_b(k);
            for (int t = 0; t < k; ++t) {
              z_b(t) = z(i, idx[t]);
            }
            VectorXd mu_b = item.A * z_b;

            const int ku = static_cast<int>(pos.size());
            VectorXd mu_u(ku);
            MatrixXd V_u(ku, ku);
            VectorXd t_u(ku);
            for (int a = 0; a < ku; ++a) {
              const int pa = pos[a];
              const int ja = idx[pa];
              mu_u(a) = mu_b(pa);
              t_u(a) = (z(i, ja) >= 0.0) ? 1.0 : -1.0;
              for (int c = 0; c < ku; ++c) {
                V_u(a, c) = item.V(pa, pos[c]);
              }
            }

            double p_block = NA_REAL;
            if (!compute_block_prob_common(
                  mu_u, V_u, t_u,
                  pmvnorm, pbivnorm, tvpack_alg,
                  maxpts, abseps, releps, jitter,
                  p_block
                )) {
              prob_l = NA_REAL;
              break;
            }
            prob_l *= p_block;
          }

          if (!R_finite(prob_l)) continue;
          sum_prob += alpha_mat(i, l) * prob_l;
        }

        out(i, m) = 1.0 - sum_prob;
      }
    }
    return out;
  }

  for (int i = 0; i < n; ++i) {
    const NumericMatrix& E_i = C_list_mat[i];
    std::vector<int> obs_idx;
    obs_idx.reserve(p);
    for (int j = 0; j < p; ++j) {
      if (R_finite(z(i, j))) obs_idx.push_back(j);
    }

    NumericMatrix E_obs(obs_idx.size(), obs_idx.size());
    for (int a = 0; a < static_cast<int>(obs_idx.size()); ++a) {
      for (int c = 0; c < static_cast<int>(obs_idx.size()); ++c) {
        E_obs(a, c) = E_i(obs_idx[a], obs_idx[c]);
      }
    }
    const std::vector< std::vector<int> > blocks_obs = apch_blocks_from_numeric_matrix(E_obs, block_tol);

    for (int m = 0; m < M; ++m) {
      const std::vector<int>& U = subset_idx[m];
      if (U.empty()) {
        out(i, m) = NA_REAL;
        continue;
      }
      if (subset_row_has_nonfinite(z, i, U)) {
        out(i, m) = NA_REAL;
        continue;
      }

      double sum_prob = 0.0;
      const std::vector<int>& alloc_keep = subset_alloc_idx[m];
      for (int q = 0; q < static_cast<int>(alloc_keep.size()); ++q) {
        const int l = alloc_keep[q];
        double prob_l = 1.0;

        for (size_t b = 0; b < blocks_obs.size(); ++b) {
          const std::vector<int>& idx_obs = blocks_obs[b];
          std::vector<int> idx;
          idx.reserve(idx_obs.size());
          for (int t = 0; t < static_cast<int>(idx_obs.size()); ++t) {
            idx.push_back(obs_idx[idx_obs[t]]);
          }
          std::vector<int> U_b;
          U_b.reserve(idx.size());
          std::vector<int> pos_b;
          pos_b.reserve(idx.size());
          for (int t = 0; t < static_cast<int>(idx.size()); ++t) {
            const int j = idx[t];
            if (subset_vec_mat(m, j) > 0) {
              U_b.push_back(j);
              pos_b.push_back(t);
            }
          }
          if (U_b.empty()) continue;

          const int k = static_cast<int>(idx.size());
          VectorXd d_vec(k);
          for (int t = 0; t < k; ++t) {
            const int j = idx[t];
            const int g = alloc_mat(l, j);
            double d = 0.0;
            if (g > 0 && g < static_cast<int>(grids[j].size())) {
              const double sd = grids[j][g];
              d = sd * sd;
            }
            d_vec(t) = d;
          }

          MatrixXd E_b(k, k);
          for (int a = 0; a < k; ++a) {
            for (int c = 0; c < k; ++c) {
              E_b(a, c) = E_i(idx[a], idx[c]);
            }
          }
          MatrixXd K = E_b;
          K.diagonal() += d_vec;

          LLT<MatrixXd> chol(K);
          if (chol.info() != Eigen::Success) {
            prob_l = NA_REAL;
            break;
          }

          VectorXd z_b(k);
          for (int a = 0; a < k; ++a) z_b(a) = z(i, idx[a]);
          VectorXd Kinv_z = chol.solve(z_b);
          VectorXd mu_b = d_vec.array() * Kinv_z.array();

          MatrixXd D = d_vec.asDiagonal();
          MatrixXd Kinv_D = chol.solve(D);
          MatrixXd V_b = D - D * Kinv_D;

          const int ku = static_cast<int>(U_b.size());
          VectorXd mu_u(ku);
          MatrixXd V_u(ku, ku);
          VectorXd t_u(ku);
          for (int a = 0; a < ku; ++a) {
            const int pa = pos_b[a];
            const int ja = U_b[a];
            mu_u(a) = mu_b(pa);
            t_u(a) = (z(i, ja) >= 0.0) ? 1.0 : -1.0;
            for (int c = 0; c < ku; ++c) {
              V_u(a, c) = V_b(pa, pos_b[c]);
            }
          }

          double p_block = NA_REAL;
          if (!compute_block_prob_common(
                mu_u, V_u, t_u,
                pmvnorm, pbivnorm, tvpack_alg,
                maxpts, abseps, releps, jitter,
                p_block
              )) {
            prob_l = NA_REAL;
            break;
          }
          prob_l *= p_block;
        }

        if (!R_finite(prob_l)) continue;
        sum_prob += alpha_mat(i, l) * prob_l;
      }

      out(i, m) = 1.0 - sum_prob;
    }
  }

  return out;
}

// [[Rcpp::export]]
List lfsr_nondiag_list_build_cache_cpp(
    const NumericMatrix& z,
    const List& C_list,
    const double block_tol = 1e-12
) {
  const int n = z.nrow();
  const int p = z.ncol();

  List obs_r(n);
  List blocks_r(n);

  for (int i = 0; i < n; ++i) {
    NumericMatrix E_i = as<NumericMatrix>(C_list[i]);

    std::vector<int> obs_idx;
    obs_idx.reserve(p);
    for (int j = 0; j < p; ++j) {
      if (R_finite(z(i, j))) obs_idx.push_back(j);
    }

    IntegerVector obs_i(obs_idx.size());
    for (int t = 0; t < static_cast<int>(obs_idx.size()); ++t) {
      obs_i[t] = obs_idx[t] + 1;
    }
    obs_r[i] = obs_i;

    std::vector< std::vector<int> > blocks =
      apch_blocks_from_numeric_matrix_subset(E_i, obs_idx, block_tol);
    List blocks_i(blocks.size());
    for (int b = 0; b < static_cast<int>(blocks.size()); ++b) {
      IntegerVector idx_b(blocks[b].size());
      for (int t = 0; t < static_cast<int>(blocks[b].size()); ++t) {
        idx_b[t] = blocks[b][t] + 1;
      }
      blocks_i[b] = idx_b;
    }
    blocks_r[i] = blocks_i;
  }

  return List::create(
    _["obs"] = obs_r,
    _["blocks"] = blocks_r
  );
}

// [[Rcpp::export]]
NumericMatrix lfsr_nondiag_list_cached_cpp(
    const NumericMatrix& z,
    const List& C_list,
    const List& list_cache,
    const IntegerMatrix& subset_vec_mat,
    const IntegerMatrix& alloc_mat,
    const List& grid_list,
    const NumericMatrix& alpha_mat,
    const int maxpts = 20000,
    const double abseps = 1e-6,
    const double releps = 1e-6,
    const double jitter = 1e-10
) {
  const int n = z.nrow();
  const int p = z.ncol();
  const int M = subset_vec_mat.nrow();
  const int L = alloc_mat.nrow();

  std::vector< std::vector<double> > grids(p);
  for (int j = 0; j < p; ++j) {
    NumericVector g = grid_list[j];
    grids[j].assign(g.begin(), g.end());
  }

  std::vector< std::vector<int> > subset_idx = subset_idx_from_subset_vec(subset_vec_mat);
  std::vector< std::vector<int> > subset_alloc_idx =
    subset_alloc_idx_from_alloc(subset_vec_mat, alloc_mat);

  std::vector<NumericMatrix> C_list_mat(n);
  for (int i = 0; i < n; ++i) {
    C_list_mat[i] = as<NumericMatrix>(C_list[i]);
  }

  List obs_cache = list_cache["obs"];
  List blocks_cache = list_cache["blocks"];

  Environment mvtnorm = Environment::namespace_env("mvtnorm");
  Function pmvnorm = mvtnorm["pmvnorm"];
  Function TVPACK = mvtnorm["TVPACK"];
  RObject tvpack_alg = TVPACK();
  Environment pbiv = Environment::namespace_env("pbivnorm");
  Function pbivnorm = pbiv["pbivnorm"];

  NumericMatrix out(n, M);

  for (int i = 0; i < n; ++i) {
    const NumericMatrix& E_i = C_list_mat[i];

    IntegerVector obs_i = obs_cache[i];
    std::vector<int> obs_idx(obs_i.size());
    for (int t = 0; t < obs_i.size(); ++t) obs_idx[t] = obs_i[t] - 1;

    List blocks_i_r = blocks_cache[i];
    const int B = blocks_i_r.size();
    std::vector< std::vector<int> > blocks_obs(B);
    for (int b = 0; b < B; ++b) {
      IntegerVector block_r = blocks_i_r[b];
      blocks_obs[b].resize(block_r.size());
      for (int t = 0; t < block_r.size(); ++t) blocks_obs[b][t] = block_r[t] - 1;
    }

    std::vector< std::vector<int> > idx_blocks(B);
    for (int b = 0; b < B; ++b) {
      const std::vector<int>& idx_obs = blocks_obs[b];
      std::vector<int>& idx = idx_blocks[b];
      idx.reserve(idx_obs.size());
      for (int t = 0; t < static_cast<int>(idx_obs.size()); ++t) {
        idx.push_back(obs_idx[idx_obs[t]]);
      }
    }

    std::vector< std::vector< std::vector<int> > > subset_block_pos(
      M,
      std::vector< std::vector<int> >(B)
    );
    for (int m = 0; m < M; ++m) {
      for (int b = 0; b < B; ++b) {
        const std::vector<int>& idx_obs = blocks_obs[b];
        std::vector<int>& pos = subset_block_pos[m][b];
        pos.reserve(idx_obs.size());
        for (int t = 0; t < static_cast<int>(idx_obs.size()); ++t) {
          const int j = obs_idx[idx_obs[t]];
          if (subset_vec_mat(m, j) > 0) pos.push_back(t);
        }
      }
    }

    for (int m = 0; m < M; ++m) {
      const std::vector<int>& U = subset_idx[m];
      if (U.empty()) {
        out(i, m) = NA_REAL;
        continue;
      }
      if (subset_row_has_nonfinite(z, i, U)) {
        out(i, m) = NA_REAL;
        continue;
      }

      double sum_prob = 0.0;
      const std::vector<int>& alloc_keep = subset_alloc_idx[m];
      for (int q = 0; q < static_cast<int>(alloc_keep.size()); ++q) {
        const int l = alloc_keep[q];
        double prob_l = 1.0;

        for (int b = 0; b < B; ++b) {
          const std::vector<int>& pos_b = subset_block_pos[m][b];
          if (pos_b.empty()) continue;

          const std::vector<int>& idx = idx_blocks[b];
          const int k = static_cast<int>(idx.size());

          VectorXd d_vec(k);
          for (int t = 0; t < k; ++t) {
            const int j = idx[t];
            const int g = alloc_mat(l, j);
            double d = 0.0;
            if (g > 0 && g < static_cast<int>(grids[j].size())) {
              const double sd = grids[j][g];
              d = sd * sd;
            }
            d_vec(t) = d;
          }

          MatrixXd E_b(k, k);
          for (int a = 0; a < k; ++a) {
            for (int c = 0; c < k; ++c) {
              E_b(a, c) = E_i(idx[a], idx[c]);
            }
          }
          MatrixXd K = E_b;
          K.diagonal() += d_vec;

          LLT<MatrixXd> chol(K);
          if (chol.info() != Eigen::Success) {
            prob_l = NA_REAL;
            break;
          }

          VectorXd z_b(k);
          for (int a = 0; a < k; ++a) z_b(a) = z(i, idx[a]);
          VectorXd Kinv_z = chol.solve(z_b);
          VectorXd mu_b = d_vec.array() * Kinv_z.array();

          MatrixXd D = d_vec.asDiagonal();
          MatrixXd Kinv_D = chol.solve(D);
          MatrixXd V_b = D - D * Kinv_D;

          const int ku = static_cast<int>(pos_b.size());
          VectorXd mu_u(ku);
          MatrixXd V_u(ku, ku);
          VectorXd t_u(ku);
          for (int a = 0; a < ku; ++a) {
            const int pa = pos_b[a];
            const int ja = idx[pa];
            mu_u(a) = mu_b(pa);
            t_u(a) = (z(i, ja) >= 0.0) ? 1.0 : -1.0;
            for (int c = 0; c < ku; ++c) {
              V_u(a, c) = V_b(pa, pos_b[c]);
            }
          }

          double p_block = NA_REAL;
          if (!compute_block_prob_common(
                mu_u, V_u, t_u,
                pmvnorm, pbivnorm, tvpack_alg,
                maxpts, abseps, releps, jitter,
                p_block
              )) {
            prob_l = NA_REAL;
            break;
          }
          prob_l *= p_block;
        }

        if (!R_finite(prob_l)) continue;
        sum_prob += alpha_mat(i, l) * prob_l;
      }

      out(i, m) = 1.0 - sum_prob;
    }
  }

  return out;
}

// [[Rcpp::export]]
List lfsr_nondiag_shared_build_cache_cpp(
    const NumericMatrix& C_shared,
    const IntegerMatrix& alloc_mat,
    const List& grid_list,
    const double block_tol = 1e-12
) {
  const int p = C_shared.nrow();
  const int L = alloc_mat.nrow();

  std::vector< std::vector<double> > grids(p);
  for (int j = 0; j < p; ++j) {
    NumericVector g = grid_list[j];
    grids[j].assign(g.begin(), g.end());
  }

  std::vector< std::vector<int> > blocks = apch_blocks_from_numeric_matrix(C_shared, block_tol);
  const int B = static_cast<int>(blocks.size());

  List blocks_r(B);
  List ok_r(B);
  List A_r(B);
  List V_r(B);

  for (int b = 0; b < B; ++b) {
    const std::vector<int>& idx = blocks[b];
    const int k = static_cast<int>(idx.size());

    IntegerVector idx_r(k);
    for (int t = 0; t < k; ++t) idx_r[t] = idx[t] + 1;
    blocks_r[b] = idx_r;

    MatrixXd E_b(k, k);
    for (int a = 0; a < k; ++a) {
      for (int c = 0; c < k; ++c) {
        E_b(a, c) = C_shared(idx[a], idx[c]);
      }
    }

    LogicalVector ok_b(L, false);
    NumericVector A_arr(L * k * k);
    NumericVector V_arr(L * k * k);
    A_arr.attr("dim") = IntegerVector::create(L, k, k);
    V_arr.attr("dim") = IntegerVector::create(L, k, k);

    for (int l = 0; l < L; ++l) {
      VectorXd d_vec(k);
      for (int t = 0; t < k; ++t) {
        const int j = idx[t];
        const int g = alloc_mat(l, j);
        double d = 0.0;
        if (g > 0 && g < static_cast<int>(grids[j].size())) {
          const double sd = grids[j][g];
          d = sd * sd;
        }
        d_vec(t) = d;
      }

      MatrixXd K = E_b;
      K.diagonal() += d_vec;
      LLT<MatrixXd> chol(K);
      if (chol.info() != Eigen::Success) {
        ok_b[l] = false;
        continue;
      }

      MatrixXd D = d_vec.asDiagonal();
      MatrixXd Kinv_D = chol.solve(D);
      MatrixXd A = Kinv_D.transpose();
      MatrixXd V = D - D * Kinv_D;
      ok_b[l] = true;

      for (int a = 0; a < k; ++a) {
        for (int c = 0; c < k; ++c) {
          const int off = l + L * (a + k * c);
          A_arr[off] = A(a, c);
          V_arr[off] = V(a, c);
        }
      }
    }

    ok_r[b] = ok_b;
    A_r[b] = A_arr;
    V_r[b] = V_arr;
  }

  return List::create(
    _["blocks"] = blocks_r,
    _["ok"] = ok_r,
    _["A"] = A_r,
    _["V"] = V_r
  );
}

// [[Rcpp::export]]
List lfsr_nondiag_shared_build_subset_meta_cpp(
    const IntegerMatrix& subset_vec_mat,
    const IntegerMatrix& alloc_mat,
    const List& shared_cache
) {
  SharedCacheParsed parsed = parse_shared_cache(shared_cache);
  return build_shared_subset_meta_list(subset_vec_mat, alloc_mat, parsed.blocks);
}

// [[Rcpp::export]]
NumericMatrix lfsr_nondiag_shared_cached_cpp(
    const NumericMatrix& z,
    const IntegerMatrix& subset_vec_mat,
    const IntegerMatrix& alloc_mat,
    const NumericMatrix& alpha_mat,
    const List& shared_cache,
    const int maxpts = 20000,
    const double abseps = 1e-6,
    const double releps = 1e-6,
    const double jitter = 1e-10
) {
  const int n = z.nrow();
  const int p = z.ncol();
  const int M = subset_vec_mat.nrow();
  const int L = alloc_mat.nrow();

  SharedCacheParsed parsed = parse_shared_cache(shared_cache);
  const int B = parsed.B;
  const std::vector< std::vector<int> >& blocks = parsed.blocks;
  const std::vector<int>& k_block = parsed.k_block;
  const std::vector<LogicalVector>& ok_block = parsed.ok_block;
  const std::vector<NumericVector>& A_block = parsed.A_block;
  const std::vector<NumericVector>& V_block = parsed.V_block;

  std::vector< std::vector<int> > subset_idx;
  std::vector< std::vector<int> > subset_alloc_idx;
  std::vector< std::vector< std::vector<int> > > subset_block_pos;
  SEXP subset_meta_sexp = shared_cache.containsElementNamed("subset_meta")
    ? shared_cache["subset_meta"] : R_NilValue;
  if (subset_meta_sexp != R_NilValue) {
    SharedSubsetMetaParsed meta = parse_shared_subset_meta(as<List>(subset_meta_sexp), B);
    subset_idx = std::move(meta.subset_idx);
    subset_alloc_idx = std::move(meta.subset_alloc_idx);
    subset_block_pos = std::move(meta.subset_block_pos);
  } else {
    subset_idx = subset_idx_from_subset_vec(subset_vec_mat);
    subset_alloc_idx = subset_alloc_idx_from_alloc(subset_vec_mat, alloc_mat);
    subset_block_pos = subset_block_pos_from_subset_and_blocks(subset_vec_mat, blocks);
  }

  Environment mvtnorm = Environment::namespace_env("mvtnorm");
  Function pmvnorm = mvtnorm["pmvnorm"];
  Function TVPACK = mvtnorm["TVPACK"];
  RObject tvpack_alg = TVPACK();
  Environment pbiv = Environment::namespace_env("pbivnorm");
  Function pbivnorm = pbiv["pbivnorm"];

  NumericMatrix out(n, M);

  for (int i = 0; i < n; ++i) {
    for (int m = 0; m < M; ++m) {
      const std::vector<int>& U = subset_idx[m];
      if (U.empty()) {
        out(i, m) = NA_REAL;
        continue;
      }
      if (subset_row_has_nonfinite(z, i, U)) {
        out(i, m) = NA_REAL;
        continue;
      }

      double sum_prob = 0.0;
      const std::vector<int>& alloc_keep = subset_alloc_idx[m];
      for (int q = 0; q < static_cast<int>(alloc_keep.size()); ++q) {
        const int l = alloc_keep[q];
        double prob_l = 1.0;

        for (int b = 0; b < B; ++b) {
          const std::vector<int>& pos = subset_block_pos[m][b];
          if (pos.empty()) continue;
          if (!ok_block[b][l]) {
            prob_l = NA_REAL;
            break;
          }

          const std::vector<int>& idx = blocks[b];
          const int k = k_block[b];
          const NumericVector& A_arr = A_block[b];
          const NumericVector& V_arr = V_block[b];

          VectorXd z_b(k);
          for (int t = 0; t < k; ++t) z_b(t) = z(i, idx[t]);

          VectorXd mu_b(k);
          for (int a = 0; a < k; ++a) {
            double acc = 0.0;
            for (int c = 0; c < k; ++c) {
              const int off = l + L * (a + k * c);
              acc += A_arr[off] * z_b(c);
            }
            mu_b(a) = acc;
          }

          const int ku = static_cast<int>(pos.size());
          VectorXd mu_u(ku);
          MatrixXd V_u(ku, ku);
          VectorXd t_u(ku);
          for (int a = 0; a < ku; ++a) {
            const int pa = pos[a];
            const int ja = idx[pa];
            mu_u(a) = mu_b(pa);
            t_u(a) = (z(i, ja) >= 0.0) ? 1.0 : -1.0;
            for (int c = 0; c < ku; ++c) {
              const int off = l + L * (pa + k * pos[c]);
              V_u(a, c) = V_arr[off];
            }
          }

          double p_block = NA_REAL;
          if (!compute_block_prob_common(
                mu_u, V_u, t_u,
                pmvnorm, pbivnorm, tvpack_alg,
                maxpts, abseps, releps, jitter,
                p_block
              )) {
            prob_l = NA_REAL;
            break;
          }
          prob_l *= p_block;
        }

        if (!R_finite(prob_l)) continue;
        sum_prob += alpha_mat(i, l) * prob_l;
      }

      out(i, m) = 1.0 - sum_prob;
    }
  }

  return out;
}

// [[Rcpp::export]]
NumericVector lfsr_nondiag_shared_cached_bestsid_cpp(
    const NumericMatrix& z,
    const IntegerVector& sid,
    const IntegerMatrix& subset_vec_mat,
    const IntegerMatrix& alloc_mat,
    const NumericMatrix& alpha_mat,
    const List& shared_cache,
    const int maxpts = 20000,
    const double abseps = 1e-6,
    const double releps = 1e-6,
    const double jitter = 1e-10
) {
  const int n = z.nrow();
  const int p = z.ncol();
  const int M = subset_vec_mat.nrow();
  const int L = alloc_mat.nrow();
  if (sid.size() != n) {
    stop("sid length must equal nrow(z).");
  }
  if (alpha_mat.nrow() != n || alpha_mat.ncol() != L) {
    stop("alpha_mat dimension mismatch in lfsr_nondiag_shared_cached_bestsid_cpp.");
  }

  SharedCacheParsed parsed = parse_shared_cache(shared_cache);
  const int B = parsed.B;
  const std::vector< std::vector<int> >& blocks = parsed.blocks;
  const std::vector<int>& k_block = parsed.k_block;
  const std::vector<LogicalVector>& ok_block = parsed.ok_block;
  const std::vector<NumericVector>& A_block = parsed.A_block;
  const std::vector<NumericVector>& V_block = parsed.V_block;

  std::vector< std::vector<int> > subset_idx;
  std::vector< std::vector<int> > subset_alloc_idx;
  std::vector< std::vector< std::vector<int> > > subset_block_pos;
  std::vector< std::vector<int> > sid_active_blocks;
  SEXP subset_meta_sexp = shared_cache.containsElementNamed("subset_meta")
    ? shared_cache["subset_meta"] : R_NilValue;
  if (subset_meta_sexp != R_NilValue) {
    SharedSubsetMetaParsed meta = parse_shared_subset_meta(as<List>(subset_meta_sexp), B);
    subset_idx = std::move(meta.subset_idx);
    subset_alloc_idx = std::move(meta.subset_alloc_idx);
    subset_block_pos = std::move(meta.subset_block_pos);
    sid_active_blocks = std::move(meta.sid_active_blocks);
  } else {
    subset_idx = subset_idx_from_subset_vec(subset_vec_mat);
    subset_alloc_idx = subset_alloc_idx_from_alloc(subset_vec_mat, alloc_mat);
    subset_block_pos = subset_block_pos_from_subset_and_blocks(subset_vec_mat, blocks);
    sid_active_blocks.resize(M);
    for (int m = 0; m < M; ++m) {
      sid_active_blocks[m].reserve(B);
      for (int b = 0; b < B; ++b) {
        if (!subset_block_pos[m][b].empty()) sid_active_blocks[m].push_back(b);
      }
    }
  }

  Environment mvtnorm = Environment::namespace_env("mvtnorm");
  Function pmvnorm = mvtnorm["pmvnorm"];
  Function TVPACK = mvtnorm["TVPACK"];
  RObject tvpack_alg = TVPACK();
  Environment pbiv = Environment::namespace_env("pbivnorm");
  Function pbivnorm = pbiv["pbivnorm"];

  NumericVector out(n, NA_REAL);

  for (int i = 0; i < n; ++i) {
    const int sid_i = sid[i] - 1;
    if (sid_i < 0 || sid_i >= M) continue;

    const std::vector<int>& U = subset_idx[sid_i];
    if (U.empty()) continue;

    if (subset_row_has_nonfinite(z, i, U)) continue;

    const std::vector<int>& active_blocks = sid_active_blocks[sid_i];
    std::vector<VectorXd> z_block(active_blocks.size());
    std::vector<VectorXd> t_block(active_blocks.size());
    for (int bi = 0; bi < static_cast<int>(active_blocks.size()); ++bi) {
      const int b = active_blocks[bi];
      const std::vector<int>& idx = blocks[b];
      const std::vector<int>& pos = subset_block_pos[sid_i][b];
      const int k = k_block[b];
      z_block[bi].resize(k);
      for (int t = 0; t < k; ++t) z_block[bi](t) = z(i, idx[t]);
      const int ku = static_cast<int>(pos.size());
      t_block[bi].resize(ku);
      for (int a = 0; a < ku; ++a) {
        const int ja = idx[pos[a]];
        t_block[bi](a) = (z(i, ja) >= 0.0) ? 1.0 : -1.0;
      }
    }

    double sum_prob = 0.0;
    const std::vector<int>& alloc_keep = subset_alloc_idx[sid_i];
    for (int q = 0; q < static_cast<int>(alloc_keep.size()); ++q) {
      const int l = alloc_keep[q];
      double prob_l = 1.0;

      for (int bi = 0; bi < static_cast<int>(active_blocks.size()); ++bi) {
        const int b = active_blocks[bi];
        const std::vector<int>& pos = subset_block_pos[sid_i][b];
        if (!ok_block[b][l]) {
          prob_l = NA_REAL;
          break;
        }

        const std::vector<int>& idx = blocks[b];
        const int k = k_block[b];
        const NumericVector& A_arr = A_block[b];
        const NumericVector& V_arr = V_block[b];

        const VectorXd& z_b = z_block[bi];

        VectorXd mu_b(k);
        for (int a = 0; a < k; ++a) {
          double acc = 0.0;
          for (int c = 0; c < k; ++c) {
            const int off = l + L * (a + k * c);
            acc += A_arr[off] * z_b(c);
          }
          mu_b(a) = acc;
        }

        const int ku = static_cast<int>(pos.size());
        VectorXd mu_u(ku);
        MatrixXd V_u(ku, ku);
        const VectorXd& t_u = t_block[bi];
        for (int a = 0; a < ku; ++a) {
          const int pa = pos[a];
          mu_u(a) = mu_b(pa);
          for (int c = 0; c < ku; ++c) {
            const int off = l + L * (pa + k * pos[c]);
            V_u(a, c) = V_arr[off];
          }
        }

        double p_block = NA_REAL;
        if (!compute_block_prob_common(
              mu_u, V_u, t_u,
              pmvnorm, pbivnorm, tvpack_alg,
              maxpts, abseps, releps, jitter,
              p_block
            )) {
          prob_l = NA_REAL;
          break;
        }
        prob_l *= p_block;
      }

      if (!R_finite(prob_l)) continue;
      sum_prob += alpha_mat(i, l) * prob_l;
    }

    out[i] = 1.0 - sum_prob;
  }

  return out;
}
