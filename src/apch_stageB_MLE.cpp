// apch_stageB_MLE.cpp — Configuration Stage-B
// 1) block_stageB_cpp : EM / mixsqp 优化版（原有）

#include "calc_density.h"
#include "squarem.h"

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

// ---------- globals for SQUAREM ----------
static int    n_obs_glob;   // n
static int    n_cfg_glob;   // R
static RowMat logmat_glob;  // n x R

// ---------- EM fixed-point for phi ----------
static std::vector<double> fixpt_phi(std::vector<double> phi) {
  const int R = n_cfg_glob;
  
  // log-phi with small floor
  std::vector<double> log_phi(R);
  for (int r = 0; r < R; ++r) {
    phi[r] = std::max(phi[r], 1e-15);
    log_phi[r] = std::log(phi[r]);
  }
  
  // E-step: responsibilities summed over i
  std::vector<double> n_r(R, 0.0), lvec(R);
  for (int i = 0; i < n_obs_glob; ++i) {
    for (int r = 0; r < R; ++r)
      lvec[r] = logmat_glob(i, r) + log_phi[r];
    const double lse = lse_row(lvec);
    for (int r = 0; r < R; ++r) {
      if (std::isfinite(lvec[r]) && std::isfinite(lse))
        n_r[r] += std::exp(lvec[r] - lse);
    }
  }
  
  for (int r = 0; r < R; ++r)
    phi[r] = n_r[r];
  
  return phi;
}

// log-likelihood objective for SQUAREM
static double objfn_phi(std::vector<double> phi) {
  const int R = n_cfg_glob;
  
  // normalize phi (log-space)
  std::vector<double> log_phi(R);
  double sump = 0.0;
  for (int r = 0; r < R; ++r) {
    phi[r] = std::max(phi[r], 1e-15);
    sump += phi[r];
  }
  const double lZ = std::log(sump);
  for (int r = 0; r < R; ++r)
    log_phi[r] = std::log(phi[r]) - lZ;
  
  // average log-likelihood
  double ll = 0.0;
  std::vector<double> lvec(R);
  for (int i = 0; i < n_obs_glob; ++i) {
    for (int r = 0; r < R; ++r)
      lvec[r] = logmat_glob(i, r) + log_phi[r];
    ll += lse_row(lvec);
  }
  
  return ll / static_cast<double>(std::max(1, n_obs_glob));
}

// ---------- sum blocks to global log_fmat ----------
struct StageBLogFWorker : public RcppParallel::Worker {
  std::vector<RcppParallel::RMatrix<double>> blocks;  // n x Rm for each block
  const std::vector<std::vector<int>>& gamma;         // R x M
  double* out;
  int n, R, M;
  
  StageBLogFWorker(const std::vector<RcppParallel::RMatrix<double>>& blocks_,
                   const std::vector<std::vector<int>>& gamma_,
                   double* out_,
                   int n_,
                   int R_,
                   int M_)
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

// ---------- solver switch (EM / mixsqp), returns phi ----------
static std::vector<double> solve_phi(
    const RowMat&       log_fmat,
    const NumericVector& phi_init,
    const std::string&  solver,
    double              max_iter,
    double              tol,
    bool                verbose_unused,  // currently unused
    // outputs (optional)
    double*             out_ll         = nullptr,
    int*                out_fe         = nullptr,
    bool*               out_converged  = nullptr
) {
  (void)verbose_unused;  // silence unused parameter warning
  
  const int n = log_fmat.rows();
  const int R = log_fmat.cols();
  
  if (solver == "em") {
    // ----- EM + SQUAREM 分支 -----
    logmat_glob = log_fmat;
    n_obs_glob  = n;
    n_cfg_glob  = R;
    
    sqctl.tol     = tol;
    sqctl.maxiter = max_iter;
    
    int    it = 0, fe = 0, oe = 0;
    double ll = 0.0;
    
    std::vector<double> phi0 = as<std::vector<double>>(phi_init);
    if ((int)phi0.size() != R)
      phi0.assign(R, 1.0 / std::max(1, R));
    
    std::vector<double> phi_vec =
      squarem1(phi0, fixpt_phi, objfn_phi, it, fe, oe, ll, false /*quiet*/);
    
    if (out_ll)        *out_ll       = ll;
    if (out_fe)        *out_fe       = fe;
    if (out_converged) *out_converged = (fe < sqctl.maxiter);
    
    return phi_vec;
  }
  
  // ----- mixsqp 分支：原始 ML 解，安静输出 -----
  NumericMatrix L(n, R);
  for (int i = 0; i < n; ++i) {
    double mx = -std::numeric_limits<double>::infinity();
    for (int r = 0; r < R; ++r)
      mx = std::max(mx, log_fmat(i, r));
    for (int r = 0; r < R; ++r) {
      const double v = log_fmat(i, r) - mx;
      L(i, r) = std::isfinite(v) ? std::exp(v) : 0.0;
    }
  }
  
  Environment ns_mixsqp  = Environment::namespace_env("mixsqp");
  Function    mixsqp_fun = ns_mixsqp["mixsqp"];
  
  NumericVector x0(R, 1.0 / std::max(1, R));
  if (phi_init.size() == R)
    x0 = clone(phi_init);
  
  List fit = mixsqp_fun(
    Named("L")      = L,
    Named("x0")     = x0,
    Named("control") = List::create(_["eps"] = tol)
  );
  
  NumericVector x = fit["x"];
  std::vector<double> phi_vec(R);
  for (int r = 0; r < R; ++r)
    phi_vec[r] = std::max((double)x[r], 1e-15);
  
  if (out_ll)        *out_ll       = NA_REAL;
  if (out_fe)        *out_fe       = 0;
  if (out_converged) *out_converged = true;
  
  return phi_vec;
}

// ====================== Stage-B (EM / mixsqp 版本) ======================
//
// block_logf:  list of block log-likelihood matrices, each n x R_m
// cfg       :  configuration bits (R x sum block_size) for mapping gamma -> block col
// phi_init  :  initial phi (length R) or empty (will default to uniform)
// p         :  number of traits (unused here; encoded via cfg & blocks)
//
// [[Rcpp::export]]
List block_stageB_cpp(const List         &block_logf,
                      const IntegerMatrix &cfg,
                      const NumericVector &phi_init,
                      int                  p,
                      double               max_iter = 1e6,
                      double               tol      = 1e-6,
                      bool                 verbose  = true,
                      std::string          solver   = "mixsqp") {
  (void)p;
  (void)cfg;
  (void)verbose;
  
  const int M = block_logf.size();
  int n = as<NumericMatrix>(block_logf[0]).nrow();
  
  std::vector<int> Rm(M), block_size(M);
  int R = 1;
  
  for (int m = 0; m < M; ++m) {
    NumericMatrix Bm = block_logf[m];
    if (Bm.nrow() != n)
      stop("block_stageB_cpp: all blocks must have same n.");
    Rm[m] = Bm.ncol();
    R    *= Rm[m];
    
    int bm = 0;
    while ((1 << bm) < Rm[m]) bm++;
    block_size[m] = bm;
  }
  
  // enumerate gamma
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
  
  // build global log_fmat
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
  
  // solve phi (delta)
  double ll   = NA_REAL;
  int    fe   = 0;
  bool   conv = true;
  
  std::vector<double> phi_vec =
    solve_phi(log_fmat, phi_init, solver, max_iter, tol,
              false /*verbose_unused*/, &ll, &fe, &conv);
  
  // normalize phi
  double ssum = 0.0;
  for (double &v : phi_vec) {
    v = std::max(v, 1e-15);
    ssum += v;
  }
  if (ssum <= 0.0) ssum = 1.0;
  for (double &v : phi_vec)
    v /= ssum;
  
  // responsibilities w_mat
  NumericMatrix w_mat(n, R);
  std::vector<double> log_phi(R);
  for (int r = 0; r < R; ++r)
    log_phi[r] = std::log(std::max(phi_vec[r], 1e-15));
  
  std::vector<double> lvec(R);
  for (int i = 0; i < n; ++i) {
    for (int r = 0; r < R; ++r)
      lvec[r] = log_fmat(i, r) + log_phi[r];
    double lse = lse_row(lvec);
    for (int r = 0; r < R; ++r)
      w_mat(i, r) = std::exp(lvec[r] - lse);
  }
  
  return List::create(
    _["delta"]     = wrap(phi_vec),
    _["w_mat"]     = w_mat,
    _["loglik"]    = ll,
    _["converged"] = conv,
    _["niter"]     = fe,
    _["solver"]    = solver
  );
}

