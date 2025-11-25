// apch_stageA.cpp
#include "calc_density.h"
#include "squarem.h"

// [[Rcpp::depends(RcppEigen,RcppParallel,mixsqp)]]
// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>
#include <Rmath.h>
#include <RcppEigen.h>
#include <RcppParallel.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <functional>
#include <iomanip>
#include <limits>
#include <atomic>
#include <mutex>

#define EIGEN_DONT_VECTORIZE

using namespace Rcpp;
using namespace RcppParallel;

// RowMat / lse_row() are declared in calc_density.h
using Eigen::MatrixXd;
using Eigen::VectorXd;

// ---------- globals for Stage-A EM ----------
static int    n_obs_glob;        // n
static int    n_cfg_block_glob;  // L
static RowMat loglik_block_glob; // n x L

// ---------- eta -> 0/1 pattern within block ----------
inline int config_to_r(const std::vector<int> &eta) {
  int r = 0;
  for (int j = 0; j < (int)eta.size(); ++j) {
    const int bit = (std::abs(eta[j]) > 0) ? 1 : 0;
    r = (r << 1) | bit;
  }
  return r;
}

// ---------- sub-likelihood builders ----------
RowMat calc_sub_logmat(const NumericMatrix &X,
                       const NumericMatrix &S,
                       const NumericMatrix &Rcor,
                       const std::vector< std::vector<int> > &eta_list,
                       const std::vector< std::vector<double> > &grids,
                       std::string prior) {
  if (prior == "normal")
    return logmat_normal(X, S, Rcor, eta_list, grids);
  else if (prior == "uniform" || prior == "halfuniform" ||
           prior == "+uniform" || prior == "-uniform")
    return logmat_uniform(X, S, Rcor, eta_list, grids, prior);
  else if (prior == "halfnormal")
    return logmat_halfnormal(X, S, Rcor, eta_list, grids);
  else {
    int n = X.nrow(), L = (int)eta_list.size();
    return RowMat(n, L);
  }
}

RowMat calc_sub_logmat_xwas(const NumericMatrix &X,
                            const NumericMatrix &S,
                            const List &Rcor,
                            const std::vector< std::vector<int> > &eta_list,
                            const std::vector< std::vector<double> > &grids,
                            const std::string &prior) {
  if (prior == "normal")
    return logmat_normal_xwas(X, S, Rcor, eta_list, grids);
  else if (prior == "uniform" || prior == "halfuniform" ||
           prior == "+uniform" || prior == "-uniform")
    return logmat_uniform_xwas(X, S, Rcor, eta_list, grids, prior);
  else if (prior == "halfnormal")
    return logmat_halfnormal_xwas(X, S, Rcor, eta_list, grids);
  else {
    int n = X.nrow(), L = (int)eta_list.size();
    return RowMat(n, L);
  }
}

// ---------- chunked worker with progress ----------
struct SubLogMatWorker : public RcppParallel::Worker {
  const NumericMatrix &X;
  const NumericMatrix &S;
  const NumericMatrix &Rcor;
  const std::vector<std::vector<int>> &eta_list;
  const std::vector<std::vector<double>> &grids;
  const std::string prior;
  int n, L, chunk_cols;
  double *out_data;
  std::atomic<int> &processed;
  std::mutex &mtx;
  bool allow_print;
  
  SubLogMatWorker(const NumericMatrix &X_, const NumericMatrix &S_,
                  const NumericMatrix &Rcor_,
                  const std::vector<std::vector<int>> &eta_list_,
                  const std::vector<std::vector<double>> &grids_,
                  const std::string &prior_, int n_, int L_, int chunk_cols_,
                  double *out_data_, std::atomic<int> &processed_,
                  std::mutex &mtx_, bool allow_print_)
    : X(X_), S(S_), Rcor(Rcor_), eta_list(eta_list_), grids(grids_),
      prior(prior_), n(n_), L(L_), chunk_cols(chunk_cols_),
      out_data(out_data_), processed(processed_), mtx(mtx_),
      allow_print(allow_print_) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t ci = begin; ci < end; ++ci) {
      const int start   = static_cast<int>(ci) * chunk_cols;
      const int endcol  = std::min(start + chunk_cols, L);
      const int width   = endcol - start;
      
      std::vector<std::vector<int>> eta_sub(
          eta_list.begin() + start, eta_list.begin() + endcol
      );
      
      RowMat sub = calc_sub_logmat(X, S, Rcor, eta_sub, grids, prior);
      
      for (int i = 0; i < n; ++i) {
        const int row_off = i * L;
        for (int c = 0; c < width; ++c)
          out_data[row_off + (start + c)] = sub(i, c);
      }
      
      if (allow_print) {
        int after = processed.fetch_add(width) + width;
        std::lock_guard<std::mutex> lock(mtx);
        double pct = 100.0 * after / std::max(1, L);
        Rcpp::Rcout << "\r[StageA] precomputing log-likelihood: "
                    << std::setw(6) << std::fixed << std::setprecision(1)
                    << pct << "% (" << after << "/" << L << ")" << std::flush;
      } else {
        processed.fetch_add(width);
      }
    }
  }
};

RowMat calc_sub_logmat_progress(const NumericMatrix &X,
                                const NumericMatrix &S,
                                const NumericMatrix &Rcor,
                                const std::vector<std::vector<int>> &eta_list,
                                const std::vector<std::vector<double>> &grids,
                                const std::string &prior,
                                int chunk_cols = 64,
                                bool show_progress = true,
                                bool use_parallel = true) {
  const int n = X.nrow();
  const int L = (int)eta_list.size();
  if (L == 0) {
    RowMat Z(n, 0);
    return Z;
  }
  if (chunk_cols <= 0) chunk_cols = 64;
  const int n_chunks = (L + chunk_cols - 1) / chunk_cols;
  
  RowMat out(n, L);
  double *out_data = out.data();
  
  std::atomic<int> processed(0);
  std::mutex mtx;
  bool allow_print = show_progress && !use_parallel;
  
  SubLogMatWorker worker(
      X, S, Rcor, eta_list, grids, prior,
      n, L, chunk_cols, out_data, processed, mtx, allow_print
  );
  
  if (use_parallel && n_chunks > 1) {
    RcppParallel::parallelFor((size_t)0, (size_t)n_chunks, worker);
    if (show_progress) {
      Rcpp::Rcout << "\r[StageA] precomputing log-likelihood: 100.0% ("
                  << L << "/" << L << ")" << std::endl;
    }
  } else {
    worker((size_t)0, (size_t)n_chunks);
    if (show_progress) Rcpp::Rcout << std::endl;
  }
  
  return out;
}

// ---------- EM fixed-point for Stage-A ----------
static std::vector<double> fixpt_block(std::vector<double> omega) {
  const int L = n_cfg_block_glob;
  std::vector<double> log_w(L);
  
  for (int l = 0; l < L; ++l) {
    omega[l] = std::max(omega[l], 1e-15);
    log_w[l] = std::log(omega[l]);
  }
  
  std::vector<double> n_l(L, 0.0), lvec(L);
  for (int i = 0; i < n_obs_glob; ++i) {
    for (int l = 0; l < L; ++l)
      lvec[l] = loglik_block_glob(i, l) + log_w[l];
    const double lse = lse_row(lvec);
    for (int l = 0; l < L; ++l) {
      if (std::isfinite(lvec[l]) && std::isfinite(lse))
        n_l[l] += std::exp(lvec[l] - lse);
    }
  }
  for (int l = 0; l < L; ++l) omega[l] = n_l[l];
  return omega;
}

static double objfn_block(std::vector<double> omega) {
  const int L = n_cfg_block_glob;
  std::vector<double> log_w(L);
  double sumw = 0.0;
  
  for (int l = 0; l < L; ++l) {
    omega[l] = std::max(omega[l], 1e-15);
    sumw += omega[l];
  }
  const double lZ = std::log(sumw);
  
  for (int l = 0; l < L; ++l)
    log_w[l] = std::log(omega[l]) - lZ;
  
  double ll = 0.0;
  std::vector<double> lvec(L);
  
  for (int i = 0; i < n_obs_glob; ++i) {
    for (int l = 0; l < L; ++l)
      lvec[l] = loglik_block_glob(i, l) + log_w[l];
    ll += lse_row(lvec);
  }
  return ll / static_cast<double>(std::max(1, n_obs_glob));
}

// ---------- build block_logf ----------
struct StageABlockLogFWorker : public RcppParallel::Worker {
  const RowMat &logL;
  const std::vector<int> &r_of_l;
  const std::vector<double> &log_omega;
  const std::vector<double> &pi_r;
  int n, L, R;
  RcppParallel::RMatrix<double> out;
  
  StageABlockLogFWorker(const RowMat &logL_,
                        const std::vector<int> &r_of_l_,
                        const std::vector<double> &log_omega_,
                        const std::vector<double> &pi_r_,
                        int n_, int L_, int R_,
                        Rcpp::NumericMatrix &out_)
    : logL(logL_), r_of_l(r_of_l_), log_omega(log_omega_),
      pi_r(pi_r_), n(n_), L(L_), R(R_), out(out_) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    const double NEG_INF = -std::numeric_limits<double>::infinity();
    for (std::size_t i = begin; i < end; ++i) {
      std::vector<double> maxv(R, NEG_INF), sumexp(R, 0.0);
      for (int l = 0; l < L; ++l) {
        const int r = r_of_l[l];
        const double val = log_omega[l] + logL(i, l);
        if (val > maxv[r]) {
          sumexp[r] = (maxv[r] == NEG_INF)
          ? 1.0
          : sumexp[r] * std::exp(maxv[r] - val) + 1.0;
          maxv[r] = val;
        } else {
          sumexp[r] += std::exp(val - maxv[r]);
        }
      }
      for (int r = 0; r < R; ++r) {
        if (pi_r[r] > 0.0 && maxv[r] != NEG_INF) {
          const double lse = maxv[r] + std::log(sumexp[r]);
          out(i, r) = lse - std::log(pi_r[r]);
        } else {
          out(i, r) = R_NegInf;
        }
      }
    }
  }
};

// ---------- Stage-A solver switch (EM / mixsqp with eta pruning) ----------
static std::vector<double>
  solve_omega(const RowMat &logL,
              const NumericVector &eta_init,
              const std::string &solver,
              double max_iter,
              double tol,
              bool verbose) {
    
    const int n = logL.rows();
    const int L = logL.cols();
    
    loglik_block_glob = logL;
    n_obs_glob        = n;
    n_cfg_block_glob  = L;
    
    // -------- EM branch: keep旧逻辑，不做剪枝 --------
    if (solver == "em") {
      sqctl.tol     = tol;
      sqctl.maxiter = max_iter;
      
      int it = 0, fe = 0, oe = 0;
      double ll = 0.0;
      
      std::vector<double> omega0 = as<std::vector<double>>(eta_init);
      std::vector<double> omega_vec =
        squarem1(omega0, fixpt_block, objfn_block, it, fe, oe, ll, verbose);
      
      return omega_vec;
    }
    
    // -------- mixsqp branch: 先全量 fit，然后按 1e-6 剪枝，再在子列上二次 fit --------
    // 稳定到线性域
    Rcpp::NumericMatrix L_stab(n, L);
    for (int i = 0; i < n; ++i) {
      double m = -std::numeric_limits<double>::infinity();
      for (int l = 0; l < L; ++l)
        m = std::max(m, logL(i, l));
      for (int l = 0; l < L; ++l) {
        const double v = logL(i, l) - m;
        L_stab(i, l) = std::isfinite(v) ? std::exp(v) : 0.0;
      }
    }
    
    Rcpp::Environment ns_mixsqp = Rcpp::Environment::namespace_env("mixsqp");
    Rcpp::Function    mixsqp    = ns_mixsqp["mixsqp"];
    
    // 第一次：在完整 L 列上跑 mixsqp（静音）
    Rcpp::List ctl0 = Rcpp::List::create(
      Rcpp::_["eps"]     = tol,
      Rcpp::_["verbose"] = false
    );
    
    Rcpp::List fit0 = mixsqp(
      Rcpp::_["L"]       = L_stab,
      Rcpp::_["x0"]      = eta_init,
      Rcpp::_["control"] = ctl0
    );
    
    Rcpp::NumericVector x0 = fit0["x"];
    
    // 归一化，确保非负
    double sumx0 = 0.0;
    for (int l = 0; l < L; ++l) {
      double v = std::max((double)x0[l], 0.0);
      x0[l] = v;
      sumx0 += v;
    }
    if (sumx0 <= 0.0) sumx0 = 1.0;
    for (int l = 0; l < L; ++l)
      x0[l] /= sumx0;
    
    // 按相对权重阈值剪枝
    const double thr = 1e-6;
    std::vector<int> keep;
    keep.reserve(L);
    for (int l = 0; l < L; ++l) {
      if (x0[l] >= thr) keep.push_back(l);
    }
    
    if (keep.empty()) {
      // 极端情况：如果全 < thr，就至少保留一个最大值
      int argmax = 0;
      double best = x0[0];
      for (int l = 1; l < L; ++l) {
        if (x0[l] > best) {
          best   = x0[l];
          argmax = l;
        }
      }
      keep.push_back(argmax);
    }
    
    const int L_kept = (int)keep.size();
    
    if (verbose) {
      Rcpp::Rcout << "[StageA] eta total=" << L
                  << ", kept=" << L_kept
                  << " (thr=" << thr << "), mixsqp converged"
                  << std::endl;
    }
    
    // 构造剪枝后的 L_stab 子矩阵
    Rcpp::NumericMatrix L_pruned(n, L_kept);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < L_kept; ++j) {
        const int col = keep[j];
        L_pruned(i, j) = L_stab(i, col);
      }
    }
    
    // 剪枝后初始值：用 x0 的子集并重新归一化
    Rcpp::NumericVector x_init_pruned(L_kept);
    double sum_init = 0.0;
    for (int j = 0; j < L_kept; ++j) {
      x_init_pruned[j] = x0[keep[j]];
      sum_init += x_init_pruned[j];
    }
    if (sum_init <= 0.0) sum_init = 1.0;
    for (int j = 0; j < L_kept; ++j)
      x_init_pruned[j] /= sum_init;
    
    // 第二次：只在保留列上跑 mixsqp（同样静音）
    Rcpp::List ctl = Rcpp::List::create(
      Rcpp::_["eps"]     = tol,
      Rcpp::_["verbose"] = false
    );
    
    Rcpp::List fit = mixsqp(
      Rcpp::_["L"]       = L_pruned,
      Rcpp::_["x0"]      = x_init_pruned,
      Rcpp::_["control"] = ctl
    );
    
    Rcpp::NumericVector x = fit["x"];
    
    // 映射回原始长度 L：未保留的列设为 0，最后统一加 floor 并归一化
    std::vector<double> out(L, 0.0);
    for (int j = 0; j < L_kept; ++j) {
      const int col = keep[j];
      double v = std::max((double)x[j], 0.0);
      out[col] = v;
    }
    
    double ssum = 0.0;
    for (int l = 0; l < L; ++l) {
      out[l] = std::max(out[l], 1e-15);
      ssum  += out[l];
    }
    if (ssum <= 0.0) ssum = 1.0;
    for (int l = 0; l < L; ++l)
      out[l] /= ssum;
    
    return out;
  }

// [[Rcpp::export]]
List block_stageA_cpp(const NumericMatrix &X,
                      const NumericMatrix &S,
                      const NumericMatrix &Rcor,
                      const List &sgL,
                      const std::vector<std::vector<int>> &eta_list,
                      const NumericVector &eta_init,
                      std::string prior = "normal",
                      std::string solver = "mixsqp",
                      double max_iter = 1e6,
                      double tol = 1e-6,
                      bool progress = true,
                      int progress_chunk = 64,
                      bool use_parallel = true,
                      int n_threads = 8,
                      bool verbose = true) {
  
  const int n = X.nrow();
  const int p = X.ncol();
  const int L = (int)eta_list.size();
  (void)n_threads;
  
  // pack grids
  std::vector<std::vector<double>> grids(p);
  for (int j = 0; j < p; ++j) {
    NumericVector g = sgL[j];
    grids[j].assign(g.begin(), g.end());
  }
  
  // 提示开始计算 log-likelihood
  if (verbose) {
    Rcpp::Rcout << "[StageA] start precomputing log-likelihood "
                << "(n=" << n << ", p=" << p << ", L=" << L << ") ..."
                << std::endl;
  }
  
  // precompute logL
  RowMat logL = calc_sub_logmat_progress(
    X, S, Rcor, eta_list, grids, prior,
    progress_chunk, progress, use_parallel
  );
  
  if (verbose) {
    Rcpp::Rcout << "[StageA] finished precomputing log-likelihood." << std::endl;
  }
  
  // solve omega (with mixsqp+prune or EM)
  std::vector<double> omega_vec =
    solve_omega(logL, eta_init, solver, max_iter, tol, verbose);
  
  // normalize
  double ssum = 0.0;
  for (double &v : omega_vec) {
    v = std::max(v, 1e-15);
    ssum += v;
  }
  if (ssum <= 0.0) ssum = 1.0;
  for (double &v : omega_vec) v /= ssum;
  
  // map to block configurations and produce block_logf
  std::vector<int> r_of_l(L);
  for (int l = 0; l < L; ++l)
    r_of_l[l] = config_to_r(eta_list[l]);
  
  const int R = 1 << p;
  
  std::vector<double> pi_r(R, 0.0), log_omega(L);
  for (int l = 0; l < L; ++l) {
    pi_r[r_of_l[l]] += omega_vec[l];
    log_omega[l]     = std::log(omega_vec[l]);
  }
  
  Rcpp::NumericMatrix block_logf(n, R);
  {
    StageABlockLogFWorker worker(
        logL, r_of_l, log_omega, pi_r, n, L, R, block_logf
    );
    if (use_parallel && n > 1)
      RcppParallel::parallelFor(0, (size_t)n, worker);
    else
      worker(0, (size_t)n);
  }
  
  return List::create(
    _["omega"]      = wrap(omega_vec),
    _["block_logf"] = block_logf,
    _["solver"]     = solver
  );
}

// [[Rcpp::export]]
List block_stageA_xwas_cpp(const NumericMatrix &X,
                           const NumericMatrix &S,
                           const List &Rcor_list,
                           const List &sgL,
                           const std::vector<std::vector<int>> &eta_list,
                           const NumericVector &eta_init,
                           std::string prior = "normal",
                           std::string solver = "mixsqp",
                           double max_iter = 1e6,
                           double tol = 1e-6,
                           bool progress = true,
                           int progress_chunk = 64,
                           bool use_parallel = true,
                           int n_threads = 8,
                           bool verbose = true) {
  
  const int n = X.nrow();
  const int p = X.ncol();
  const int L = (int)eta_list.size();
  (void)n_threads;
  (void)progress;
  (void)progress_chunk;
  (void)use_parallel;
  
  if (Rcor_list.size() != n)
    stop("block_stageA_xwas_cpp: Rcor_list length must equal nrow(X).");
  
  // pack grids
  std::vector<std::vector<double>> grids(p);
  for (int j = 0; j < p; ++j) {
    NumericVector g = sgL[j];
    grids[j].assign(g.begin(), g.end());
  }
  
  // 提示开始 / 结束计算 log-likelihood（xwas 版本没有 chunk 进度条）
  if (verbose) {
    Rcpp::Rcout << "[StageA-xwas] start precomputing log-likelihood "
                << "(n=" << n << ", p=" << p << ", L=" << L << ") ..."
                << std::endl;
  }
  
  RowMat logL = calc_sub_logmat_xwas(
    X, S, Rcor_list, eta_list, grids, prior
  );
  
  if (verbose) {
    Rcpp::Rcout << "[StageA-xwas] finished precomputing log-likelihood."
                << std::endl;
  }
  
  // solve omega
  std::vector<double> omega_vec =
    solve_omega(logL, eta_init, solver, max_iter, tol, verbose);
  
  // normalize
  double ssum = 0.0;
  for (double &v : omega_vec) {
    v = std::max(v, 1e-15);
    ssum += v;
  }
  if (ssum <= 0.0) ssum = 1.0;
  for (double &v : omega_vec) v /= ssum;
  
  // map to block configurations and produce block_logf
  std::vector<int> r_of_l(L);
  for (int l = 0; l < L; ++l)
    r_of_l[l] = config_to_r(eta_list[l]);
  
  const int R = 1 << p;
  
  std::vector<double> pi_r(R, 0.0), log_omega(L);
  for (int l = 0; l < L; ++l) {
    pi_r[r_of_l[l]] += omega_vec[l];
    log_omega[l]     = std::log(omega_vec[l]);
  }
  
  Rcpp::NumericMatrix block_logf(n, R);
  {
    StageABlockLogFWorker worker(
        logL, r_of_l, log_omega, pi_r, n, L, R, block_logf
    );
    if (n > 1)
      RcppParallel::parallelFor(0, (size_t)n, worker);
    else
      worker(0, (size_t)n);
  }
  
  return List::create(
    _["omega"]      = wrap(omega_vec),
    _["block_logf"] = block_logf,
    _["solver"]     = solver
  );
}




// [[Rcpp::export]]
Rcpp::NumericMatrix block_stageA_predict_cpp(const NumericMatrix &X,
                                          const NumericMatrix &S,
                                          const NumericMatrix &Rcor,
                                          const List &sgL,
                                          const std::vector<std::vector<int>> &eta_list,
                                          const NumericVector &omega,
                                          std::string prior = "normal",
                                          bool progress = true,
                                          int progress_chunk = 64,
                                          bool use_parallel = true,
                                          bool verbose = true) {
  const int n = X.nrow();
  const int p = X.ncol();
  const int L = (int)eta_list.size();
  
  if (S.nrow() != n || S.ncol() != p)
    stop("block_stageA_eval_cpp: X and S must have same dimensions.");
  if (Rcor.nrow() != p || Rcor.ncol() != p)
    stop("block_stageA_eval_cpp: Rcor must be a p x p matrix.");
  if (sgL.size() != p)
    stop("block_stageA_eval_cpp: length(sgL) must equal ncol(X).");
  if (omega.size() != L)
    stop("block_stageA_eval_cpp: length(omega) must equal length(eta_list).");
  
  // pack grids
  std::vector<std::vector<double>> grids(p);
  for (int j = 0; j < p; ++j) {
    NumericVector g = sgL[j];
    grids[j].assign(g.begin(), g.end());
  }
  
  if (verbose) {
    Rcpp::Rcout << "[StageA-eval] start precomputing log-likelihood "
                << "(n=" << n << ", p=" << p << ", L=" << L << ") ..."
                << std::endl;
  }
  
  // 只算 logL，不再 EM / mixsqp 更新 omega
  RowMat logL = calc_sub_logmat_progress(
    X, S, Rcor, eta_list, grids, prior,
    progress_chunk, progress, use_parallel
  );
  
  if (verbose) {
    Rcpp::Rcout << "[StageA-eval] finished precomputing log-likelihood."
                << std::endl;
  }
  
  // 与 block_stageA_cpp 中相同的步骤：用固定 omega 构造 block_logf
  std::vector<int> r_of_l(L);
  for (int l = 0; l < L; ++l)
    r_of_l[l] = config_to_r(eta_list[l]);
  
  const int R = 1 << p;
  
  std::vector<double> pi_r(R, 0.0), log_omega(L);
  for (int l = 0; l < L; ++l) {
    double w = std::max( (double)omega[l], 1e-15 );
    log_omega[l] = std::log(w);
    pi_r[r_of_l[l]] += w;
  }
  
  Rcpp::NumericMatrix block_logf(n, R);
  {
    StageABlockLogFWorker worker(
        logL, r_of_l, log_omega, pi_r, n, L, R, block_logf
    );
    if (use_parallel && n > 1)
      RcppParallel::parallelFor(0, (size_t)n, worker);
    else
      worker(0, (size_t)n);
  }
  
  return block_logf;
}
