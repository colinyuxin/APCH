// apch_stageB_mcmc.cpp  — Configuration Stage-B, collapsed Gibbs MCMC
//
// This implementation:
//
//   1. Builds the global log-likelihood matrix log_fmat (n x R) from block-wise
//      log-likelihood matrices.
//   2. Converts log_fmat to a numerically stable linear-scale table
//      f_tab(i, r) ∝ f_ir.
//   3. Runs a single mixsqp fit on f_tab to obtain an MLE phi_mle, and uses
//      only its null mass to define an initialization phi_init:
//           phi_init[null_r]      = phi_mle[null_r]
//           phi_init[r != null_r] = (1 - phi_mle[null_r]) / (R - 1).
//      If anything goes wrong with mixsqp, phi_init falls back to uniform.
//   4. Runs a single-chain collapsed Gibbs sampler on **all** effects
//      (no pre-screening):
//           P(z_i = r | z_-i, data) ∝ (alpha + n_r^{(-i)}) * f_ir
//      with alpha = 1 (uniform Dirichlet prior), 4000 sweeps, 1000 burn-in.
//   5. Uses post–burn-in samples (with optional thinning) to compute:
//        - delta: posterior mean of phi (Dirichlet(alpha + counts))
//        - w_mat: per-effect posterior configuration probabilities, obtained
//                 by averaging the conditional P(z_i = r | z_-i, data) over
//                 kept sweeps.
//        - phi_trace and sweep_index: MCMC trajectory of phi for diagnostics.

#include "calc_density.h"

// [[Rcpp::depends(Rcpp,RcppEigen,RcppParallel,mixsqp)]]
// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>
#include <Rmath.h>
#include <RcppEigen.h>
#include <RcppParallel.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>

using namespace Rcpp;
using namespace RcppParallel;

// ---------- worker: sum block log-likelihoods into global log_fmat ----------
struct StageBLogFWorker : public RcppParallel::Worker {
  std::vector<RcppParallel::RMatrix<double>> blocks;  // n x R_m for each block
  const std::vector<std::vector<int>>& gamma;         // R x M
  double* out;                                        // pointer to log_fmat
  int n, R, M;
  
  StageBLogFWorker(const std::vector<RcppParallel::RMatrix<double>>& blocks_,
                   const std::vector<std::vector<int>>& gamma_,
                   double* out_, int n_, int R_, int M_)
    : blocks(blocks_), gamma(gamma_), out(out_), n(n_), R(R_), M(M_) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      double* row_ptr = out + (std::size_t)i * R;
      for (int r = 0; r < R; ++r) {
        double acc = 0.0;
        for (int m = 0; m < M; ++m)
          acc += blocks[m](i, gamma[r][m]);
          row_ptr[r] = acc;
      }
    }
  }
};

// ====================== Stage-B MCMC (full, no pre-screening) ======================
//
// [[Rcpp::export]]
List block_stageB_mcmc_cpp(const List &block_logf,
                           const IntegerMatrix &cfg,
                           int p,
                           bool verbose = true) {
  (void)p;
  
  const int M = block_logf.size();
  int n = as<NumericMatrix>(block_logf[0]).nrow();
  
  // Determine per-block numbers of configs and the total number of global configs R.
  std::vector<int> Rm(M), block_size(M);
  int R = 1;
  for (int m = 0; m < M; ++m) {
    NumericMatrix Bm = block_logf[m];
    if (Bm.nrow() != n)
      stop("block_stageB_mcmc_cpp: all blocks must have same n.");
    Rm[m] = Bm.ncol();
    R    *= Rm[m];
    
    int bm = 0;
    while ((1 << bm) < Rm[m]) bm++;
    block_size[m] = bm;
  }
  
  // Enumerate gamma: for each global configuration r, which column index does it
  // correspond to in each block?
  std::vector<std::vector<int>> gamma_list(R, std::vector<int>(M));
  for (int r = 0; r < R; ++r) {
    int pos = 0;
    for (int m = 0; m < M; ++m) {
      int c  = 0;
      int bm = block_size[m];
      for (int j = 0; j < bm; ++j) {
        int bit = cfg(r, pos + j);
        if (bit)
          c |= (1 << (bm - j - 1));
      }
      gamma_list[r][m] = c;
      pos += bm;
    }
  }
  
  // -------- Build global log_fmat: n x R --------
  std::vector<RcppParallel::RMatrix<double>> Bm_list;
  Bm_list.reserve(M);
  for (int m = 0; m < M; ++m) {
    Bm_list.emplace_back(
      RcppParallel::RMatrix<double>(
        Rcpp::as<Rcpp::NumericMatrix>(block_logf[m])
      )
    );
  }
  
  RowMat log_fmat(n, R);
  {
    StageBLogFWorker worker(Bm_list, gamma_list, log_fmat.data(), n, R, M);
    if (n > 1)
      RcppParallel::parallelFor(0, (size_t)n, worker);
    else
      worker(0, (size_t)n);
  }
  
  // -------- Convert log_fmat to a stable linear-scale density table f_tab --------
  //
  // For each row i, we subtract row max and exponentiate:
  //   f_tab(i, r) = exp(log_fmat(i, r) - max_r log_fmat(i, r))
  // so f_tab(i, r) ∝ f_ir up to a row-specific constant factor.
  RowMat f_tab(n, R);
  for (int i = 0; i < n; ++i) {
    double mx = -std::numeric_limits<double>::infinity();
    for (int r = 0; r < R; ++r)
      mx = std::max(mx, log_fmat(i, r));
    
    for (int r = 0; r < R; ++r) {
      double v = log_fmat(i, r) - mx;
      f_tab(i, r) = std::isfinite(v) ? std::exp(v) : 0.0;
    }
  }
  // If memory is tight, log_fmat could be released here:
  //   log_fmat.resize(0, 0);
  
  // -------- Identify the global null configuration index null_r --------
  int p_cfg = cfg.ncol();
  int null_r = -1;
  for (int r = 0; r < cfg.nrow(); ++r) {
    bool all_zero = true;
    for (int j = 0; j < p_cfg; ++j) {
      if (cfg(r, j) != 0) {
        all_zero = false;
        break;
      }
    }
    if (all_zero) {
      null_r = r;
      break;
    }
  }
  if (null_r < 0) {
    // Should not happen in practice; fall back to column 0 defensively.
    null_r = 0;
  }
  
  // ====================== Optional: mixsqp-based initialization ======================
  //
  // We treat f_tab as the component density matrix L(i, r) for a finite mixture:
  //     x_i ~ sum_r phi_r f_ir
  // and use mixsqp to obtain an MLE of phi. Only the null mass is used:
  //   phi_init[null_r]      = phi_mle[null_r]
  //   phi_init[r != null_r] = (1 - phi_mle[null_r]) / (R - 1).
  // If anything fails, we fall back to uniform phi_init.
  std::vector<double> phi_init(R, 0.0);
  bool init_ok = false;
  
  try {
    NumericMatrix L_mle(n, R);
    for (int i = 0; i < n; ++i) {
      for (int r = 0; r < R; ++r) {
        L_mle(i, r) = f_tab(i, r);
      }
    }
    
    Environment ns_mixsqp = Environment::namespace_env("mixsqp");
    Function mixsqp_fun = ns_mixsqp["mixsqp"];
    
    NumericVector x0(R, 1.0 / std::max(1, R));
    List control = List::create(
      _["eps"]     = 1e-6,
      _["verbose"] = false
    );
    
    List fit = mixsqp_fun(
      _["L"]       = L_mle,
      _["x0"]      = x0,
      _["control"] = control
    );
    
    NumericVector x_hat = fit["x"];
    std::vector<double> phi_mle(R);
    double sump = 0.0;
    for (int r = 0; r < R; ++r) {
      double v = std::max((double)x_hat[r], 0.0);
      phi_mle[r] = v;
      sump      += v;
    }
    if (sump <= 0.0) sump = 1.0;
    for (int r = 0; r < R; ++r)
      phi_mle[r] /= sump;
    
    double null_mass = 0.0;
    if (null_r >= 0 && null_r < R)
      null_mass = std::max(0.0, std::min(1.0, phi_mle[null_r]));
    double nonnull_mass = 1.0 - null_mass;
    if (nonnull_mass < 0.0) nonnull_mass = 0.0;
    
    int R_nonnull = std::max(0, R - 1);
    double each = (R_nonnull > 0) ? (nonnull_mass / (double)R_nonnull) : 0.0;
    for (int r = 0; r < R; ++r) {
      if (r == null_r) phi_init[r] = null_mass;
      else             phi_init[r] = each;
    }
    init_ok = true;
    
    if (verbose) {
      Rcout << "[StageB-MCMC] mixsqp init: null mass = "
            << null_mass << ", non-null spread uniformly over "
            << R_nonnull << " configs.\n";
    }
  } catch (...) {
    init_ok = false;
  }
  
  if (!init_ok) {
    double invR = 1.0 / std::max(1, R);
    for (int r = 0; r < R; ++r)
      phi_init[r] = invR;
    if (verbose) {
      Rcout << "[StageB-MCMC] mixsqp init failed; using uniform init.\n";
    }
  }
  
  // ====================== Stage 1: collapsed Gibbs MCMC ======================
  const double alpha        = 1.0;   // Dirichlet(alpha,...,alpha)
  const int    n_sweeps     = 4000;  // total number of sweeps
  const int    burnin       = 1000;  // burn-in sweeps
  const int    sample_start = burnin;
  const int    thin         = 1;     // thinning interval
  
  const int max_kept = (n_sweeps > sample_start)
    ? ((n_sweeps - sample_start - 1) / thin + 1)
    : 0;
  
  if (verbose) {
    Rcout << "[StageB-MCMC] n=" << n
          << ", R=" << R
          << ", sweeps=" << n_sweeps
          << ", burnin=" << burnin
          << ", thin=" << thin
          << " (full MCMC, no prescreening)"
          << "\n";
  }
  
  // Latent configs z_i and counts n_r
  std::vector<int> z(n, 0);
  std::vector<int> counts(R, 0);
  
  // Accumulators for posterior summaries
  std::vector<double> phi_accum(R, 0.0);
  NumericMatrix       w_mat(n, R);          // will be divided by n_kept
  NumericMatrix       phi_trace(max_kept, R);
  int                 kept_index = 0;
  
  std::vector<double> tmp_prob(R);
  
  // ----- Initialization of z_i -----
  // Sample z_i from P(z_i | phi_init, f_tab(i, .)).
  for (int i = 0; i < n; ++i) {
    double sumw = 0.0;
    for (int r = 0; r < R; ++r) {
      double w = phi_init[r] * f_tab(i, r);
      tmp_prob[r] = w;
      sumw += w;
    }
    
    if (!(sumw > 0.0)) {
      double invR = 1.0 / std::max(1, R);
      for (int r = 0; r < R; ++r)
        tmp_prob[r] = invR;
      sumw = 1.0;
    } else {
      for (int r = 0; r < R; ++r)
        tmp_prob[r] /= sumw;
    }
    
    double u = R::runif(0.0, 1.0);
    double csum = 0.0;
    int zr = R - 1;
    for (int r = 0; r < R; ++r) {
      csum += tmp_prob[r];
      if (u <= csum) {
        zr = r;
        break;
      }
    }
    z[i] = zr;
    counts[zr] += 1;
  }
  
  // ----- MCMC sweeps -----
  for (int sweep = 0; sweep < n_sweeps; ++sweep) {
    const bool in_sample_window = (sweep >= sample_start &&
                                   ((sweep - sample_start) % thin == 0));
    
    // One full sweep over i
    for (int i = 0; i < n; ++i) {
      int old_r = z[i];
      if (old_r >= 0 && old_r < R) {
        counts[old_r] -= 1;
        if (counts[old_r] < 0) counts[old_r] = 0; // defensive
      }
      
      // Collapsed Gibbs:
      //   P(z_i = r | z_-i, data) ∝ (alpha + n_r^{(-i)}) * f_ir
      double sumw = 0.0;
      for (int r = 0; r < R; ++r) {
        double w = (alpha + (double)counts[r]) * f_tab(i, r);
        tmp_prob[r] = w;
        sumw += w;
      }
      
      if (!(sumw > 0.0)) {
        double invR = 1.0 / std::max(1, R);
        for (int r = 0; r < R; ++r)
          tmp_prob[r] = invR;
        sumw = 1.0;
      } else {
        for (int r = 0; r < R; ++r)
          tmp_prob[r] /= sumw;
      }
      
      // Sample new z_i
      double u = R::runif(0.0, 1.0);
      double csum = 0.0;
      int new_r = R - 1;
      for (int r = 0; r < R; ++r) {
        csum += tmp_prob[r];
        if (u <= csum) {
          new_r = r;
          break;
        }
      }
      z[i] = new_r;
      counts[new_r] += 1;
      
      // Accumulate per-effect posterior means
      if (in_sample_window) {
        for (int r = 0; r < R; ++r)
          w_mat(i, r) += tmp_prob[r];
      }
    } // end i loop
    
    // Accumulate phi trajectory: E[phi | counts] for Dirichlet(alpha + counts).
    if (in_sample_window && kept_index < max_kept) {
      double denom = alpha * (double)R + (double)n;
      for (int r = 0; r < R; ++r) {
        double val = (alpha + (double)counts[r]) / denom;
        phi_accum[r] += val;
        phi_trace(kept_index, r) = val;
      }
      kept_index += 1;
    }
    
    if (verbose && ((sweep + 1) % 50 == 0 || sweep == n_sweeps - 1)) {
      Rcout << "[StageB-MCMC] sweep " << (sweep + 1) << "/" << n_sweeps << "\r";
      if (sweep == n_sweeps - 1) Rcout << "\n";
    }
  } // end sweeps
  
  // ----- Summarize phi from MCMC -----
  int n_kept = kept_index;
  std::vector<double> phi_vec(R, 0.0);
  NumericMatrix phi_trace_out(std::max(1, n_kept), R);
  IntegerVector sweep_index(std::max(1, n_kept));
  
  if (n_kept > 0) {
    // Posterior mean of phi
    for (int r = 0; r < R; ++r)
      phi_vec[r] = phi_accum[r] / (double)n_kept;
    
    // Copy phi_trace and record the sweep indices
    for (int k = 0; k < n_kept; ++k) {
      for (int r = 0; r < R; ++r)
        phi_trace_out(k, r) = phi_trace(k, r);
      sweep_index[k] = sample_start + k * thin + 1;
    }
  } else {
    // Defensive fallback: if no sample was kept, use the last counts to form phi.
    double denom = alpha * (double)R + (double)n;
    for (int r = 0; r < R; ++r) {
      double val = (alpha + (double)counts[r]) / denom;
      phi_vec[r]          = val;
      phi_trace_out(0, r) = val;
    }
    sweep_index[0] = n_sweeps;
    n_kept = 1;
  }
  
  // Normalize w_mat to get posterior mean P(z_i = r | data).
  if (n_kept > 0) {
    for (int i = 0; i < n; ++i) {
      for (int r = 0; r < R; ++r)
        w_mat(i, r) /= (double)n_kept;
    }
  }
  
  return List::create(
    _["delta"]       = wrap(phi_vec),
    _["w_mat"]       = w_mat,
    _["phi_trace"]   = phi_trace_out,   // n_kept x R
    _["sweep_index"] = sweep_index,     // length n_kept
    _["converged"]   = true,            // single chain; R-hat not computed
    _["niter"]       = n_sweeps,
    _["solver"]      = std::string("mcmc")
  );
}


// [[Rcpp::export]]
Rcpp::NumericMatrix block_stageB_predict_cpp(const List &block_logf,
                                             const IntegerMatrix &cfg,
                                             const NumericVector &delta,
                                             int p,
                                             bool verbose = true) {
  const int M = block_logf.size();
  if (M < 1)
    stop("block_stageB_predict_cpp: block_logf must have at least one block.");
  
  // n：新数据的个数
  int n = as<NumericMatrix>(block_logf[0]).nrow();
  
  // 计算每个 block 的配置数 Rm，以及总配置数 R
  std::vector<int> Rm(M), block_size(M);
  int R = 1;
  for (int m = 0; m < M; ++m) {
    NumericMatrix Bm = block_logf[m];
    if (Bm.nrow() != n)
      stop("block_stageB_predict_cpp: all blocks must have same n.");
    Rm[m] = Bm.ncol();
    R    *= Rm[m];
    
    int bm = 0;
    while ((1 << bm) < Rm[m]) bm++;
    block_size[m] = bm;
  }
  
  if (cfg.nrow() != R)
    stop("block_stageB_predict_cpp: nrow(cfg) must equal total number of configs R.");
  
  if (delta.size() != R)
    stop("block_stageB_predict_cpp: length(delta) must equal total number of configs R.");
  
  // -------- Enumerate gamma: 与 block_stageB_mcmc_cpp 相同 ----------
  std::vector<std::vector<int>> gamma_list(R, std::vector<int>(M));
  for (int r = 0; r < R; ++r) {
    int pos = 0;
    for (int m = 0; m < M; ++m) {
      int c  = 0;
      int bm = block_size[m];
      for (int j = 0; j < bm; ++j) {
        int bit = cfg(r, pos + j);
        if (bit)
          c |= (1 << (bm - j - 1));
      }
      gamma_list[r][m] = c;
      pos += bm;
    }
  }
  
  // -------- Build global log_fmat: n x R --------
  std::vector<RcppParallel::RMatrix<double>> Bm_list;
  Bm_list.reserve(M);
  for (int m = 0; m < M; ++m) {
    Bm_list.emplace_back(
      RcppParallel::RMatrix<double>(
        Rcpp::as<Rcpp::NumericMatrix>(block_logf[m])
      )
    );
  }
  
  RowMat log_fmat(n, R);
  {
    StageBLogFWorker worker(Bm_list, gamma_list, log_fmat.data(), n, R, M);
    if (n > 1)
      RcppParallel::parallelFor(0, (size_t)n, worker);
    else
      worker(0, (size_t)n);
  }
  
  // -------- 转成稳定的线性尺度 f_tab(i,r) ∝ f_ir --------
  RowMat f_tab(n, R);
  for (int i = 0; i < n; ++i) {
    double mx = -std::numeric_limits<double>::infinity();
    for (int r = 0; r < R; ++r)
      mx = std::max(mx, log_fmat(i, r));
    
    for (int r = 0; r < R; ++r) {
      double v = log_fmat(i, r) - mx;
      f_tab(i, r) = std::isfinite(v) ? std::exp(v) : 0.0;
    }
  }
  
  // -------- 归一化 delta，作为固定先验 φ_r --------
  std::vector<double> phi(R);
  double ssum = 0.0;
  for (int r = 0; r < R; ++r) {
    double v = std::max((double)delta[r], 0.0);
    phi[r] = v;
    ssum  += v;
  }
  if (ssum <= 0.0) {
    double invR = 1.0 / std::max(1, R);
    for (int r = 0; r < R; ++r)
      phi[r] = invR;
  } else {
    for (int r = 0; r < R; ++r)
      phi[r] /= ssum;
  }
  
  if (verbose) {
    Rcout << "[StageB-predict] n=" << n
          << ", R=" << R
          << ", M=" << M
          << " (fixed delta, no MCMC)\n";
  }
  
  // -------- 对每个 i 计算 P(z_i = r | x_i; phi, f_tab) --------
  Rcpp::NumericMatrix w_mat(n, R);
  for (int i = 0; i < n; ++i) {
    double row_sum = 0.0;
    for (int r = 0; r < R; ++r) {
      double val = phi[r] * f_tab(i, r);
      w_mat(i, r) = val;
      row_sum    += val;
    }
    if (row_sum > 0.0) {
      for (int r = 0; r < R; ++r)
        w_mat(i, r) /= row_sum;
    } else {
      double invR = 1.0 / std::max(1, R);
      for (int r = 0; r < R; ++r)
        w_mat(i, r) = invR;
    }
  }
  
  return w_mat;
}

