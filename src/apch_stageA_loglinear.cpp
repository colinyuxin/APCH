// apch_stageA_loglinear.cpp — Stage-A loglinear + L2 (meta / xwas)
// 只实现：
//   - block_stageA_loglinear_cpp      (meta 模式，有全局 R_cor)
//   - block_stageA_xwas_loglinear_cpp (xwas 模式，有 per-effect C_i)
//
// 依赖：RowMat, lse_row, calc_sub_logmat_* 都来自 calc_density.h / apch_stageA.cpp.
//
// [[Rcpp::depends(RcppEigen,RcppParallel)]]
// [[Rcpp::plugins(cpp17)]]

#include "calc_density.h"

#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <random>

using namespace Rcpp;
using namespace RcppParallel;

// ========== 外部函数声明（在 apch_stageA.cpp 里定义） ==========

// 这些在你现有的 apch_stageA.cpp 里已经实现了，这里只做 forward declaration。
RowMat calc_sub_logmat(const NumericMatrix &X,
                       const NumericMatrix &S,
                       const NumericMatrix &Rcor,
                       const std::vector< std::vector<int> > &eta_list,
                       const std::vector< std::vector<double> > &grids,
                       std::string prior);

RowMat calc_sub_logmat_xwas(const NumericMatrix &X,
                            const NumericMatrix &S,
                            const List &Rcor,
                            const std::vector< std::vector<int> > &eta_list,
                            const std::vector< std::vector<double> > &grids,
                            const std::string &prior);

RowMat calc_sub_logmat_progress(const NumericMatrix &X,
                                const NumericMatrix &S,
                                const NumericMatrix &Rcor,
                                const std::vector<std::vector<int>> &eta_list,
                                const std::vector<std::vector<double>> &grids,
                                const std::string &prior,
                                int  chunk_cols,
                                bool show_progress,
                                bool use_parallel);

// ========== 共用小工具：softmax、config 编码、block_logf worker ==========

// softmax: R^L -> Δ^{L-1}
static inline Eigen::VectorXd softmax_vec(const Eigen::VectorXd &psi) {
  const int L = psi.size();
  double mx = psi.maxCoeff();
  Eigen::ArrayXd exps = (psi.array() - mx).exp();
  double Z = exps.sum();
  if (!(Z > 0.0)) {
    return Eigen::VectorXd::Ones(L) / (double)L;
  }
  return exps / Z;
}

// eta (长度 p，取 {0,1,...}) -> 对应 block 配置 r \in {0,...,2^p - 1}
inline int config_to_r(const std::vector<int> &eta) {
  int r = 0;
  for (int j = 0; j < (int)eta.size(); ++j) {
    int bit = (std::abs(eta[j]) > 0) ? 1 : 0;
    r = (r << 1) | bit;
  }
  return r;
}

// 构造 block_logf: 每行 i，对每个 block 配置 r 把子配置 logL(h) + log ω_h 聚合
struct StageABlockLogFWorker : public RcppParallel::Worker {
  const RowMat &logL;                     // n x L
  const std::vector<int>    &r_of_l;      // length L
  const std::vector<double> &log_omega;   // length L, log ω_h
  const std::vector<double> &pi_r;        // length R, π_r = ∑_h ω_h
  int n, L, R;
  RcppParallel::RMatrix<double> out;      // n x R
  
  StageABlockLogFWorker(const RowMat &logL_,
                        const std::vector<int>    &r_of_l_,
                        const std::vector<double> &log_omega_,
                        const std::vector<double> &pi_r_,
                        int n_, int L_, int R_,
                        Rcpp::NumericMatrix &out_)
    : logL(logL_), r_of_l(r_of_l_), log_omega(log_omega_), pi_r(pi_r_),
      n(n_), L(L_), R(R_), out(out_) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    const double NEG_INF = -std::numeric_limits<double>::infinity();
    for (std::size_t i = begin; i < end; ++i) {
      std::vector<double> maxv(R, NEG_INF), sumexp(R, 0.0);
      for (int l = 0; l < L; ++l) {
        const int r   = r_of_l[l];
        const double v = log_omega[l] + logL(i, l);
        if (v > maxv[r]) {
          sumexp[r] = (maxv[r] == NEG_INF)
          ? 1.0
          : sumexp[r] * std::exp(maxv[r] - v) + 1.0;
          maxv[r] = v;
        } else {
          sumexp[r] += std::exp(v - maxv[r]);
        }
      }
      for (int r = 0; r < R; ++r) {
        if (pi_r[r] > 0.0 && maxv[r] != NEG_INF) {
          double lse = maxv[r] + std::log(sumexp[r]);
          // 按原 block_stageA_cpp 的语义：除以 π_r，得到“条件于该 config”的 log-likelihood
          out(i, r) = lse - std::log(pi_r[r]);
        } else {
          out(i, r) = R_NegInf;
        }
      }
    }
  }
};

// ========== StageA loglinear：E-step 并行累加 N_h = ∑_i w_{ih} ===========

struct StageALoglinearEWorker : public RcppParallel::Worker {
  const RowMat &logL;                    // n x L
  const std::vector<double> &log_omega;  // length L
  int L;
  std::vector<double> N_local;           // length L
  
  StageALoglinearEWorker(const RowMat &logL_,
                         const std::vector<double> &log_omega_,
                         int L_)
    : logL(logL_), log_omega(log_omega_), L(L_), N_local(L_, 0.0) {}
  
  StageALoglinearEWorker(const StageALoglinearEWorker &other,
                         RcppParallel::Split)
    : logL(other.logL),
      log_omega(other.log_omega),
      L(other.L),
      N_local(L, 0.0) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    std::vector<double> lvec(L);
    for (std::size_t i = begin; i < end; ++i) {
      for (int h = 0; h < L; ++h) {
        lvec[h] = logL(i, h) + log_omega[h];
      }
      const double lse = lse_row(lvec);
      for (int h = 0; h < L; ++h) {
        double w = std::exp(lvec[h] - lse);
        if (std::isfinite(w)) {
          N_local[h] += w;
        }
      }
    }
  }
  
  void join(const StageALoglinearEWorker &rhs) {
    for (int h = 0; h < L; ++h) {
      N_local[h] += rhs.N_local[h];
    }
  }
};

static Eigen::VectorXd
stageA_E_accumulate_N(const RowMat &logL,
                      const Eigen::VectorXd &log_omega_eig,
                      bool use_parallel)
{
  const int n = logL.rows();
  const int L = logL.cols();
  std::vector<double> log_omega(L);
  for (int h = 0; h < L; ++h) log_omega[h] = log_omega_eig[h];
  
  StageALoglinearEWorker worker(logL, log_omega, L);
  if (use_parallel && n > 1) {
    RcppParallel::parallelReduce((std::size_t)0, (std::size_t)n, worker);
  } else {
    worker((std::size_t)0, (std::size_t)n);
  }
  
  Eigen::VectorXd N(L);
  for (int h = 0; h < L; ++h) N[h] = worker.N_local[h];
  return N;
}

// ========== 仿 StageB 的 λ 诊断工具：A(v), power / inverse-power ==========

static inline void apply_A(const Eigen::VectorXd& v,
                           Eigen::VectorXd& out,
                           const Eigen::VectorXd& delta,
                           const Eigen::VectorXd& lam_r,
                           double n) {
  // out = n*(delta ⊙ v - (delta^T v) * delta) + lam_r ⊙ v
  const int L = v.size();
  double s = delta.dot(v);
  out = (n * (delta.array() * v.array()).matrix()) - (n * s) * delta;
  out.array() += lam_r.array() * v.array();
}

static double power_max_eig(int L,
                            const Eigen::VectorXd& delta,
                            const Eigen::VectorXd& lam_r,
                            double n,
                            int iters = 50) {
  std::mt19937_64 rng(12345);
  std::normal_distribution<> nd(0.0,1.0);
  Eigen::VectorXd v(L);
  for (int i = 0; i < L; ++i) v[i] = nd(rng);
  v.normalize();
  
  Eigen::VectorXd Av(L);
  for (int t = 0; t < iters; ++t) {
    apply_A(v, Av, delta, lam_r, n);
    double normAv = Av.norm();
    if (normAv <= 0) break;
    v = Av / normAv;
  }
  apply_A(v, Av, delta, lam_r, n);
  double num = v.dot(Av);
  double den = v.squaredNorm();
  return (den > 0) ? (num / den) : std::numeric_limits<double>::quiet_NaN();
}

static Eigen::VectorXd cg_solve_A(const Eigen::VectorXd& b,
                                  const Eigen::VectorXd& delta,
                                  const Eigen::VectorXd& lam_r,
                                  double n,
                                  int maxit = 200, double tol = 1e-6) {
  const int L = b.size();
  Eigen::VectorXd x = Eigen::VectorXd::Zero(L);
  Eigen::VectorXd r = b;
  Eigen::VectorXd p = r;
  Eigen::VectorXd Ap(L);
  double rr_old = r.dot(r);
  double rr0 = rr_old;
  if (rr_old == 0.0) return x;
  for (int k = 0; k < maxit; ++k) {
    apply_A(p, Ap, delta, lam_r, n);
    double denom = p.dot(Ap);
    if (std::abs(denom) < 1e-18) break;
    double alpha = rr_old / denom;
    x.noalias() += alpha * p;
    r.noalias() -= alpha * Ap;
    double rr_new = r.dot(r);
    if (std::sqrt(rr_new) <= tol * std::sqrt(rr0)) break;
    double beta = rr_new / rr_old;
    p = r + beta * p;
    rr_old = rr_new;
  }
  return x;
}

static double inverse_power_min_eig(int L,
                                    const Eigen::VectorXd& delta,
                                    const Eigen::VectorXd& lam_r,
                                    double n,
                                    int iters = 3) {
  std::mt19937_64 rng(67890);
  std::normal_distribution<> nd(0.0,1.0);
  Eigen::VectorXd v(L);
  for (int i = 0; i < L; ++i) v[i] = nd(rng);
  v.normalize();
  
  for (int t = 0; t < iters; ++t) {
    Eigen::VectorXd z = cg_solve_A(v, delta, lam_r, n, 200, 1e-6);
    double nz = z.norm();
    if (nz <= 0) break;
    v = z / nz;
  }
  Eigen::VectorXd Av(L);
  apply_A(v, Av, delta, lam_r, n);
  double num = v.dot(Av);
  double den = v.squaredNorm();
  return (den > 0) ? (num / den) : std::numeric_limits<double>::quiet_NaN();
}

// ========== 核心：给定 logL，解 softmax(psi) + L2 的 ω_h = δ_h(psi) ===========
//
// maximize_psi   Q(psi)
//   = ∑_h N_h psi_h - n_eff log ∑_k exp(psi_k) - (lambda_A/2) ||psi||^2.
//
// 梯度： g_h = N_h - n_eff δ_h - lambda_A psi_h.
// Hessian：H = -n_eff (Diag(δ) - δδ^T) - lambda_A I.
// 我们解 A step = g, A = -H = n_eff(Diag(δ) - δδ^T) + lambda_A I.
//
// 注意：这里通过数据自动估计 λ，并写回 lambda_A（按引用传入），用于后续诊断/返回。
//
static std::vector<double>
  solve_omega_loglinear(const RowMat              &logL,
                        const Rcpp::NumericVector &eta_init,
                        double                    &lambda_A,   // <- 现在按引用传入
                        int                        max_em,
                        double                     tol,
                        bool                       use_parallel,
                        bool                       verbose)
  {
    const int n = logL.rows();
    const int L = logL.cols();
    
    // ---------- 预阶段：在均匀先验下跑一次 E 步，得到 delta0 ----------
    Eigen::VectorXd log_omega_uniform = Eigen::VectorXd::Constant(L, -std::log((double)L));
    Eigen::VectorXd N0 = stageA_E_accumulate_N(logL, log_omega_uniform, use_parallel);
    Eigen::VectorXd delta0 = (N0.array() / (double)n).matrix();
    
    // ---------- 估计 A0 的最大特征值，A0 = n*(Diag(delta0) - delta0*delta0^T) ----------
    const double kappa_target = 1.0e4;   // 目标条件数，与 StageB 保持一致
    Eigen::VectorXd lam_zero = Eigen::VectorXd::Zero(L);
    double eigmax_A0 = power_max_eig(L, delta0, lam_zero, (double)n, 50);
    if (!std::isfinite(eigmax_A0) || eigmax_A0 <= 0.0) {
      // 极端退化的保护：用 n/L 做个粗近似
      eigmax_A0 = (double)n / std::max(1, L);
    }
    
    // ---------- 初始各向同性 λ，使 1 + eigmax(A0)/λ ≈ κ* ----------
    double base_lambda = eigmax_A0 / (kappa_target - 1.0);
    if (!(base_lambda > 0.0)) base_lambda = 1e-6;
    
    // ---------- 小步调优 λ：保证 κ(A) ≲ κ* ----------
    const int tune_max_iter = 5;
    double eigmax_est = std::numeric_limits<double>::quiet_NaN();
    double eigmin_est = std::numeric_limits<double>::quiet_NaN();
    double kappa_est  = std::numeric_limits<double>::quiet_NaN();
    
    for (int it = 0; it < tune_max_iter; ++it) {
      Eigen::VectorXd lam_r_t = Eigen::VectorXd::Constant(L, base_lambda);
      eigmax_est = power_max_eig(L, delta0, lam_r_t, (double)n, 40);
      eigmin_est = inverse_power_min_eig(L, delta0, lam_r_t, (double)n, 3);
      if (std::isfinite(eigmax_est) && std::isfinite(eigmin_est) && eigmin_est > 0.0)
        kappa_est = eigmax_est / eigmin_est;
      else
        kappa_est = std::numeric_limits<double>::infinity();
      
      if (kappa_est <= kappa_target) break;
      
      double scale = std::max(1.05, std::min(4.0, kappa_est / kappa_target));
      base_lambda *= scale; // 只增加 λ
    }
    
    // 最终使用的 λ（完全由上面的诊断给出）
    lambda_A = base_lambda;
    
    if (verbose) {
      Rcpp::Rcout << "[StageA-loglinear][lambda-auto] "
                  << "n=" << n
                  << ", L=" << L
                  << ", eigmax(A0)~" << eigmax_A0
                  << ", λ=" << lambda_A
                  << ", κ_est~" << kappa_est
                  << std::endl;
    }
    
    // ---------- 正式 EM + Newton 更新 psi ----------
    // --- 初始化 psi：从 eta_init（如果有）映射到 softmax 的对数 ---
    Eigen::VectorXd psi(L);
    if (eta_init.size() == L) {
      double sum_eta = 0.0;
      for (int h = 0; h < L; ++h) {
        double v = std::max((double)eta_init[h], 1e-15);
        sum_eta += v;
      }
      if (sum_eta <= 0.0) sum_eta = 1.0;
      double mean_log = 0.0;
      for (int h = 0; h < L; ++h) {
        double v = std::max((double)eta_init[h], 1e-15) / sum_eta;
        psi[h] = std::log(v);
        mean_log += psi[h];
      }
      mean_log /= (double)L;
      for (int h = 0; h < L; ++h) psi[h] -= mean_log;
    } else {
      psi.setZero();
    }
    
    Eigen::VectorXd delta = softmax_vec(psi);
    Eigen::VectorXd log_omega(L);
    for (int h = 0; h < L; ++h) {
      log_omega[h] = std::log(std::max(delta[h], 1e-300));
    }
    
    const double eps_rel = 1e-12;
    bool converged = false;
    int  it_em = 0;
    
    auto Q_value = [&](const Eigen::VectorXd &psi_test,
                       const Eigen::VectorXd &N_vec,
                       double n_eff) {
      const int Lloc = psi_test.size();
      double mx = psi_test.maxCoeff();
      Eigen::ArrayXd exps = (psi_test.array() - mx).exp();
      double Z = exps.sum();
      if (!(Z > 0.0)) Z = 1.0;
      double lse = mx + std::log(Z);
      double quad = psi_test.squaredNorm();
      return N_vec.dot(psi_test) - n_eff * lse - 0.5 * lambda_A * quad;
    };
    
    for (it_em = 0; it_em < max_em; ++it_em) {
      // ----- E-step: N_h = ∑_i w_{ih}（并行） -----
      Eigen::VectorXd N = stageA_E_accumulate_N(logL, log_omega, use_parallel);
      double n_eff = N.sum();
      if (!(n_eff > 0.0)) n_eff = (double)n;
      
      // ----- M-step: psi 上的牛顿一步 -----
      Eigen::VectorXd g(L);
      for (int h = 0; h < L; ++h) {
        g[h] = N[h] - n_eff * delta[h] - lambda_A * psi[h];
      }
      
      Eigen::MatrixXd A = Eigen::MatrixXd::Zero(L, L);
      for (int h = 0; h < L; ++h) {
        A(h, h) = n_eff * delta[h] + lambda_A;
      }
      A.noalias() -= n_eff * (delta * delta.transpose());
      for (int h = 0; h < L; ++h) {
        A(h, h) += 1e-12; // 数值 jitter
      }
      
      Eigen::VectorXd step(L);
      {
        Eigen::LLT<Eigen::MatrixXd> llt(A);
        if (llt.info() == Eigen::Success) {
          step = llt.solve(g);
        } else {
          Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
          if (ldlt.info() == Eigen::Success) {
            step = ldlt.solve(g);
          } else {
            Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
            if (lu.isInvertible()) {
              step = lu.solve(g);
            } else {
              step.setZero();
            }
          }
        }
      }
      
      double Q0 = Q_value(psi, N, n_eff);
      
      // line search 保证 Q 不降
      double t = 1.0;
      Eigen::VectorXd psi_new(L), delta_new(L);
      for (int bt = 0; bt < 20; ++bt) {
        psi_new   = psi + t * step;
        delta_new = softmax_vec(psi_new);
        double Q1 = Q_value(psi_new, N, n_eff);
        if (Q1 >= Q0) break;
        t *= 0.5;
      }
      
      Eigen::VectorXd delta_old = delta;
      double max_rel = 0.0;
      for (int h = 0; h < L; ++h) {
        double denom = std::max(eps_rel, delta_old[h]);
        double rel   = std::abs(delta_new[h] - delta_old[h]) / denom;
        if (rel > max_rel) max_rel = rel;
      }
      double step_norm = (t * step).norm();
      
      psi   = psi_new;
      delta = delta_new;
      
      for (int h = 0; h < L; ++h) {
        log_omega[h] = std::log(std::max(delta[h], 1e-300));
      }
      
      if (verbose && ((it_em + 1) % 10 == 0)) {
        Rcpp::Rcout << "  [StageA-loglinear] iter=" << (it_em + 1)
                    << "  max rel|Δδ|=" << max_rel
                    << "  ||step||=" << step_norm
                    << std::endl;
      }
      
      if (max_rel <= tol || step_norm <= tol) {
        converged = true;
        ++it_em;
        break;
      }
    }
    
    if (verbose) {
      Rcpp::Rcout << "[StageA-loglinear] "
                  << (converged ? "converged" : "stopped without convergence")
                  << " (niter=" << it_em
                  << ", lambda=" << lambda_A << ")"
                  << std::endl;
    }
    
    // ----- 打包成 ω 向量 -----
    std::vector<double> omega_vec(L);
    double sumw = 0.0;
    for (int h = 0; h < L; ++h) {
      double v = std::max(delta[h], 1e-15);
      omega_vec[h] = v;
      sumw += v;
    }
    if (!(sumw > 0.0)) sumw = 1.0;
    for (int h = 0; h < L; ++h) omega_vec[h] /= sumw;
    
    return omega_vec;
  }

// ========== 导出：meta 模式 StageA-loglinear ==========
//
// 注意：lambda_A 参数传进来只是占位，真正使用的 λ 完全由 solve_omega_loglinear 自动估计，
// 并通过引用写回 lambda_A，便于返回给 R 端做诊断。
// [[Rcpp::export]]
Rcpp::List block_stageA_loglinear_cpp(const NumericMatrix &X,
                                      const NumericMatrix &S,
                                      const NumericMatrix &Rcor,
                                      const List &sgL,
                                      const std::vector<std::vector<int>> &eta_list,
                                      const NumericVector &eta_init,
                                      std::string prior = "normal",
                                      double lambda_A   = 0.05,  // <- 初始值无关紧要，会被自动 λ 覆盖
                                      int    max_em     = 500,
                                      double tol        = 1e-2,
                                      bool   progress   = true,
                                      int    progress_chunk = 64,
                                      bool   use_parallel   = true,
                                      int    n_threads      = 8,
                                      bool   verbose        = true)
{
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
  
  if (verbose) {
    Rcpp::Rcout << "[StageA-loglinear] start precomputing log-likelihood "
                << "(n=" << n << ", p=" << p << ", L=" << L << ") ..."
                << std::endl;
  }
  
  // 与原 block_stageA_cpp 相同：用 chunked + parallel 的 logL
  RowMat logL = calc_sub_logmat_progress(
    X, S, Rcor, eta_list, grids, prior,
    progress_chunk, progress, use_parallel
  );
  
  if (verbose) {
    Rcpp::Rcout << "[StageA-loglinear] finished precomputing log-likelihood."
                << std::endl;
  }
  
  // ---- 求 ω（loglinear + L2，λ 自动估计）----
  lambda_A = std::max(lambda_A, 0.0); // 只是确保非负，实际会被 solve_omega_loglinear 覆盖
  int max_iter_em = std::max(1, max_em);
  
  std::vector<double> omega_vec =
    solve_omega_loglinear(logL, eta_init, lambda_A,
                          max_iter_em, tol, use_parallel, verbose);
  
  // map h -> block 配置 r
  std::vector<int> r_of_l(L);
  for (int l = 0; l < L; ++l)
    r_of_l[l] = config_to_r(eta_list[l]);
  
  const int R = 1 << p;
  std::vector<double> pi_r(R, 0.0), log_omega(L);
  for (int l = 0; l < L; ++l) {
    int r = r_of_l[l];
    pi_r[r] += omega_vec[l];
    log_omega[l] = std::log(omega_vec[l]);
  }
  
  NumericMatrix block_logf(n, R);
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
    _["solver"]     = std::string("loglinear"),
    _["lambda"]     = lambda_A  // 返回最终自动估计的 λ
  );
}

// ========== 导出：xwas 模式 StageA-loglinear ==========
//
// 同样，lambda_A 参数仅为占位；真正 λ 由 solve_omega_loglinear 自动估计，
// 并通过引用写回 lambda_A。
// [[Rcpp::export]]
Rcpp::List block_stageA_xwas_loglinear_cpp(const NumericMatrix &X,
                                           const NumericMatrix &S,
                                           const List &Rcor_list,
                                           const List &sgL,
                                           const std::vector<std::vector<int>> &eta_list,
                                           const NumericVector &eta_init,
                                           std::string prior = "normal",
                                           double lambda_A   = 0.05,  // <- 会被覆盖
                                           int    max_em     = 500,
                                           double tol        = 1e-2,
                                           bool   progress   = true,
                                           int    progress_chunk = 64,
                                           bool   use_parallel   = true,
                                           int    n_threads      = 8,
                                           bool   verbose        = true)
{
  const int n = X.nrow();
  const int p = X.ncol();
  const int L = (int)eta_list.size();
  (void)n_threads;
  (void)progress;
  (void)progress_chunk;
  (void)use_parallel;  // xwas 下 logL 本身就不 chunk
  
  if ((int)Rcor_list.size() != n)
    stop("block_stageA_xwas_loglinear_cpp: Rcor_list length must equal nrow(X).");
  
  // pack grids
  std::vector<std::vector<double>> grids(p);
  for (int j = 0; j < p; ++j) {
    NumericVector g = sgL[j];
    grids[j].assign(g.begin(), g.end());
  }
  
  if (verbose) {
    Rcpp::Rcout << "[StageA-xwas-loglinear] start precomputing log-likelihood "
                << "(n=" << n << ", p=" << p << ", L=" << L << ") ..."
                << std::endl;
  }
  
  RowMat logL = calc_sub_logmat_xwas(
    X, S, Rcor_list, eta_list, grids, prior
  );
  
  if (verbose) {
    Rcpp::Rcout << "[StageA-xwas-loglinear] finished precomputing log-likelihood."
                << std::endl;
  }
  
  lambda_A = std::max(lambda_A, 0.0);
  int max_iter_em = std::max(1, max_em);
  
  std::vector<double> omega_vec =
    solve_omega_loglinear(logL, eta_init, lambda_A,
                          max_iter_em, tol, /*use_parallel=*/true, verbose);
  
  std::vector<int> r_of_l(L);
  for (int l = 0; l < L; ++l)
    r_of_l[l] = config_to_r(eta_list[l]);
  
  const int R = 1 << p;
  std::vector<double> pi_r(R, 0.0), log_omega(L);
  for (int l = 0; l < L; ++l) {
    int r = r_of_l[l];
    pi_r[r] += omega_vec[l];
    log_omega[l] = std::log(omega_vec[l]);
  }
  
  NumericMatrix block_logf(n, R);
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
    _["solver"]     = std::string("loglinear"),
    _["lambda"]     = lambda_A   // 同样返回自动 λ
  );
}

