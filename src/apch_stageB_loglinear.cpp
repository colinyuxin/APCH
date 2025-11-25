// apch_stageB_loglinear.cpp — Configuration Stage-B
// (saturated multinomial + isotropic ridge + diagnostics + auto-lambda + parallel E-step)
//
// [[Rcpp::depends(Rcpp,RcppEigen,RcppParallel)]]
// [[Rcpp::plugins(cpp17)]]

#include "calc_density.h"   // RowMat, lse_row()
#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <string>
#include <cmath>
#include <random>

using namespace Rcpp;
using namespace RcppParallel;

// ---------- worker: sum block log-likelihoods into global log_fmat ----------
struct StageBLogFWorker : public RcppParallel::Worker {
  std::vector<RcppParallel::RMatrix<double>> blocks;  // n x Rm for each block
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
        for (int m = 0; m < M; ++m) {
          acc += blocks[m](i, gamma[r][m]);
        }
        row_ptr[r] = acc;
      }
    }
  }
};

// ---------- softmax utilities (saturated: eta = theta) ----------
static inline Eigen::VectorXd softmax_from_theta(const Eigen::VectorXd& theta) {
  const int R = theta.size();
  double mx = theta.maxCoeff();
  Eigen::ArrayXd exps = (theta.array() - mx).exp();
  double Z = exps.sum();
  if (!(Z > 0.0)) return Eigen::VectorXd::Ones(R) / (double)R;
  return exps / Z;
}

// ---------- parallel E-step worker: accumulate N without forming w_mat ----------
struct EAccumWorker : public RcppParallel::Worker {
  const RowMat& log_fmat;                 // n x R, read-only
  const std::vector<double>& log_delta;   // length R, read-only
  int R;
  std::vector<double> N_local;            // thread-local accumulator
  
  EAccumWorker(const RowMat& log_fmat_,
               const std::vector<double>& log_delta_,
               int R_)
    : log_fmat(log_fmat_), log_delta(log_delta_), R(R_), N_local(R_, 0.0) {}
  
  EAccumWorker(const EAccumWorker& other, RcppParallel::Split)
    : log_fmat(other.log_fmat), log_delta(other.log_delta),
      R(other.R), N_local(other.R, 0.0) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    std::vector<double> lvec(R);
    for (std::size_t i = begin; i < end; ++i) {
      for (int r = 0; r < R; ++r) lvec[r] = log_fmat(i, r) + log_delta[r];
      const double lse = lse_row(lvec);
      for (int r = 0; r < R; ++r) {
        double v = std::exp(lvec[r] - lse);
        if (std::isfinite(v)) N_local[r] += v;
      }
    }
  }
  
  void join(const EAccumWorker& rhs) {
    for (int r = 0; r < R; ++r) N_local[r] += rhs.N_local[r];
  }
};

// E-step (parallel): accumulate N = sum_i w_i
static Eigen::VectorXd E_accumulate_N(const RowMat& log_fmat,
                                      const Eigen::VectorXd& log_delta_eig) {
  const int n = log_fmat.rows();
  const int R = log_fmat.cols();
  std::vector<double> log_delta(R);
  for (int r = 0; r < R; ++r) log_delta[r] = log_delta_eig[r];
  EAccumWorker worker(log_fmat, log_delta, R);
  RcppParallel::parallelReduce((std::size_t)0, (std::size_t)n, worker);
  Eigen::VectorXd N(R);
  for (int r = 0; r < R; ++r) N[r] = worker.N_local[r];
  return N;
}

// helper: generate pretty names for theta by cfg bits
static std::vector<std::string> cfg_names(const IntegerMatrix& cfg) {
  const int R = cfg.nrow();
  const int p = cfg.ncol();
  std::vector<std::string> nm; nm.reserve(R);
  for (int r = 0; r < R; ++r) {
    std::string s = "cfg_";
    for (int j = 0; j < p; ++j) s.push_back(char('0' + (cfg(r, j) ? 1 : 0)));
    nm.push_back(s);
  }
  return nm;
}

// --------- helpers: A(v) and simple solvers for diagnostics ----------
static inline void apply_A(const Eigen::VectorXd& v,
                           Eigen::VectorXd& out,
                           const Eigen::VectorXd& delta,
                           const Eigen::VectorXd& lam_r,
                           double n) {
  // out = n*(delta ⊙ v - (delta^T v) * delta) + lam_r ⊙ v
  const int R = v.size();
  double s = delta.dot(v);
  out = (n * (delta.array() * v.array()).matrix()) - (n * s) * delta;
  out.array() += lam_r.array() * v.array();
}

static double power_max_eig(int R,
                            const Eigen::VectorXd& delta,
                            const Eigen::VectorXd& lam_r,
                            double n,
                            int iters = 50) {
  std::mt19937_64 rng(12345);
  std::normal_distribution<> nd(0.0,1.0);
  Eigen::VectorXd v(R);
  for (int i = 0; i < R; ++i) v[i] = nd(rng);
  v.normalize();
  
  Eigen::VectorXd Av(R);
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
  const int R = b.size();
  Eigen::VectorXd x = Eigen::VectorXd::Zero(R);
  Eigen::VectorXd r = b;
  Eigen::VectorXd p = r;
  Eigen::VectorXd Ap(R);
  double rr_old = r.dot(r);
  double rr0 = rr_old;
  if (rr_old == 0.0) return x;
  for (int k = 0; k < maxit; ++k) {
    apply_A(p, Ap, delta, lam_r, n);   // <--- 这里改成这样
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


static double inverse_power_min_eig(int R,
                                    const Eigen::VectorXd& delta,
                                    const Eigen::VectorXd& lam_r,
                                    double n,
                                    int iters = 3) {
  std::mt19937_64 rng(67890);
  std::normal_distribution<> nd(0.0,1.0);
  Eigen::VectorXd v(R);
  for (int i = 0; i < R; ++i) v[i] = nd(rng);
  v.normalize();
  
  for (int t = 0; t < iters; ++t) {
    Eigen::VectorXd z = cg_solve_A(v, delta, lam_r, n, 200, 1e-6);
    double nz = z.norm();
    if (nz <= 0) break;
    v = z / nz;
  }
  Eigen::VectorXd Av(R);
  apply_A(v, Av, delta, lam_r, n);
  double num = v.dot(Av);
  double den = v.squaredNorm();
  return (den > 0) ? (num / den) : std::numeric_limits<double>::quiet_NaN();
}

// ====================== 主函数（各向同性岭；目标 κ*=10000） ======================
//
// [[Rcpp::export]]
Rcpp::List block_stageB_loglinear_cpp(const Rcpp::List& block_logf,
                                      const Rcpp::IntegerMatrix& cfg, // R x p (MSB)
                                      double lambda      = NA_REAL,   // 可选起点；若非正则则忽略
                                      int    max_em      = 500,
                                      double tol         = 1e-2,      // 相对误差阈值（对 delta）
                                      bool   verbose     = true) {
  const int M = block_logf.size();
  if (M < 1) Rcpp::stop("block_stageB_loglinear_cpp: block_logf is empty.");
  int n = Rcpp::as<Rcpp::NumericMatrix>(block_logf[0]).nrow();
  
  // --- block sizes & gamma mapping ---
  std::vector<int> Rm(M), block_size(M);
  int R = 1;
  for (int m = 0; m < M; ++m) {
    Rcpp::NumericMatrix Bm = block_logf[m];
    if (Bm.nrow() != n)
      Rcpp::stop("block_stageB_loglinear_cpp: all blocks must have same n.");
    Rm[m] = Bm.ncol();
    R    *= Rm[m];
    int bm = 0; while ((1 << bm) < Rm[m]) bm++;
    block_size[m] = bm;
  }
  if (cfg.nrow() != R) Rcpp::stop("block_stageB_loglinear_cpp: nrow(cfg) must equal R.");
  if (cfg.ncol() <= 0) Rcpp::stop("block_stageB_loglinear_cpp: cfg must have p >= 1 columns.");
  
  // map global config r -> each block's column
  std::vector<std::vector<int>> gamma_list(R, std::vector<int>(M));
  for (int r = 0; r < R; ++r) {
    int pos = 0;
    for (int m = 0; m < M; ++m) {
      int c = 0, bm = block_size[m];
      for (int j = 0; j < bm; ++j) if (cfg(r, pos + j)) c |= (1 << (bm - j - 1));
      gamma_list[r][m] = c;
      pos += bm;
    }
  }
  
  // --- build global log_fmat: n x R ---
  std::vector<RcppParallel::RMatrix<double>> Bm_list; Bm_list.reserve(M);
  for (int m = 0; m < M; ++m)
    Bm_list.emplace_back(RcppParallel::RMatrix<double>(Rcpp::as<Rcpp::NumericMatrix>(block_logf[m])));
  RowMat log_fmat(n, R);
  {
    StageBLogFWorker worker(Bm_list, gamma_list, log_fmat.data(), n, R, M);
    if (n > 1) RcppParallel::parallelFor(0, (size_t)n, worker);
    else       worker(0, (size_t)n);
  }
  
  // --- saturated model params ---
  Eigen::VectorXd theta = Eigen::VectorXd::Zero(R);
  Eigen::VectorXd delta = Eigen::VectorXd::Ones(R) / (double)R;
  Eigen::VectorXd log_delta = (delta.array().max(1e-300)).log().matrix();
  
  // ===================== 预诊断：进入 EM 前 =====================
  // 1) 均匀 prior 下的 E 步（并行） → delta0
  Eigen::VectorXd log_delta_uniform = Eigen::VectorXd::Constant(R, -std::log((double)R));
  Eigen::VectorXd N0 = E_accumulate_N(log_fmat, log_delta_uniform);
  Eigen::VectorXd delta0 = (N0.array() / (double)n).matrix();
  
  // 熵与有效配置数
  double H = 0.0;
  for (int r = 0; r < R; ++r) {
    double pr = std::max(1e-300, (double)delta0[r]);
    H -= pr * std::log(pr);
  }
  double H_norm = H / std::log((double)R);
  double N_eff  = std::exp(H);
  
  // 2) 目标条件数（固定）
  const double kappa_target = 1.0e4;
  
  // 3) 用 power 估计 A0 的最大特征值，A0 = n*(Diag(delta0)-delta0*delta0^T)
  Eigen::VectorXd lam_zero = Eigen::VectorXd::Zero(R);
  double eigmax_A0 = power_max_eig(R, delta0, lam_zero, (double)n, 50);
  if (!std::isfinite(eigmax_A0) || eigmax_A0 <= 0.0) {
    // 退化保护：用 n/R 的解析近似
    eigmax_A0 = (double)n / std::max(1, R);
  }
  
  // 4) 初始各向同性 λ：使 1 + eigmax(A0)/λ ≈ κ*
  double base_lambda = std::isfinite(lambda) && (lambda > 0.0)
    ? lambda
  : std::max(1e-12, eigmax_A0 / (kappa_target - 1.0));
  
  // 5) 小步校准（确保 κ(A) ≤ κ*）
  const int tune_max_iter = 5;
  double eigmax_est = std::numeric_limits<double>::quiet_NaN();
  double eigmin_est = std::numeric_limits<double>::quiet_NaN();
  double kappa_est  = std::numeric_limits<double>::quiet_NaN();
  
  for (int it = 0; it < tune_max_iter; ++it) {
    Eigen::VectorXd lam_r_t = Eigen::VectorXd::Constant(R, base_lambda);
    eigmax_est = power_max_eig(R, delta0, lam_r_t, (double)n, 40);
    eigmin_est = inverse_power_min_eig(R, delta0, lam_r_t, (double)n, 3);
    if (std::isfinite(eigmax_est) && std::isfinite(eigmin_est) && eigmin_est > 0.0)
      kappa_est = eigmax_est / eigmin_est;
    else
      kappa_est = std::numeric_limits<double>::infinity();
    
    if (kappa_est <= kappa_target) break;
    
    double scale = std::max(1.05, std::min(4.0, kappa_est / kappa_target));
    base_lambda *= scale; // 只放大各向同性 λ
  }
  
  // 6) 固定最终 lam_r、Gersh 上界等诊断量
  Eigen::VectorXd lam_r = Eigen::VectorXd::Constant(R, base_lambda);
  double min_lam = base_lambda;
  
  double max_disc = 0.0;
  for (int r = 0; r < R; ++r) {
    double v = 2.0 * (double)n * delta0[r] * (1.0 - delta0[r]) + base_lambda;
    if (v > max_disc) max_disc = v;
  }
  double kappa_gersh = (min_lam > 0.0) ? (max_disc / min_lam) : std::numeric_limits<double>::infinity();
  
  if (verbose) {
    Rcpp::Rcout << "[StageB-loglinear][pre-EM] R=" << R
                << "  min(lambda_r)=" << min_lam
                << "  Gersh_kappa<=" << kappa_gersh
                << "  (power kappa~" << kappa_est << ", A0_eigmax~" << eigmax_A0 << ")"
                << "  Heff=" << N_eff
                << "  Hnorm=" << H_norm
                << "  | chosen lambda=" << base_lambda
                << "\n";
  }
  
  // ============= 正式 EM 循环（E-step 并行，其它不变） =============
  if (verbose) {
    Rcpp::Rcout << "[StageB-loglinear] n=" << n
                << ", R=" << R
                << ", p=" << cfg.ncol()
                << ", (saturated model)"
                << ", lambda=" << base_lambda
                << "\n";
  }
  
  const double eps_rel = 1e-12;
  bool   converged = false;
  double last_max_rel = std::numeric_limits<double>::infinity();
  int it_em = 0;
  
  for (; it_em < max_em; ++it_em) {
    // E-step (parallel)
    Eigen::VectorXd N = E_accumulate_N(log_fmat, log_delta);
    
    // M-step
    Eigen::VectorXd g = N - (double)n * delta - lam_r.cwiseProduct(theta);
    
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(R, R);
    for (int r = 0; r < R; ++r) A(r, r) = (double)n * delta[r] + lam_r[r];
    A.noalias() -= (double)n * (delta * delta.transpose());
    for (int r = 0; r < R; ++r) A(r, r) += 1e-12;
    
    Eigen::VectorXd step(R);
    {
      Eigen::LLT<Eigen::MatrixXd> llt(A);
      if (llt.info() == Eigen::Success) step = llt.solve(g);
      else {
        Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
        if (ldlt.info() == Eigen::Success) step = ldlt.solve(g);
        else {
          Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
          step = lu.isInvertible() ? (lu.solve(g)).eval() : Eigen::VectorXd::Zero(R);
        }
      }
    }
    
    auto Q_value = [&](const Eigen::VectorXd& th) {
      double mx = th.maxCoeff();
      double lse = mx + std::log((th.array() - mx).exp().sum());
      double quad = 0.0; for (int r = 0; r < R; ++r) quad += lam_r[r] * th[r] * th[r];
      return N.dot(th) - (double)n * lse - 0.5 * quad;
    };
    
    double Q0 = Q_value(theta);
    double t = 1.0;
    Eigen::VectorXd theta_new, delta_new;
    for (int bt = 0; bt < 20; ++bt) {
      theta_new = theta + t * step;
      delta_new = softmax_from_theta(theta_new);
      double Q1 = Q_value(theta_new);
      if (Q1 >= Q0) break;
      t *= 0.5;
    }
    
    Eigen::VectorXd delta_old = delta;
    delta = delta_new;
    log_delta = (delta.array().max(1e-300)).log().matrix();
    
    double max_rel = 0.0;
    for (int r = 0; r < R; ++r) {
      double denom = std::max(eps_rel, delta_old[r]);
      double rel   = std::abs(delta[r] - delta_old[r]) / denom;
      if (rel > max_rel) max_rel = rel;
    }
    last_max_rel = max_rel;
    
    double step_eff_norm = (t * step).norm();
    theta = theta_new;
    
    if (verbose && ((it_em + 1) % 10 == 0)) {
      Rcpp::Rcout << "  [EM] iter=" << (it_em + 1)
                  << "  max rel|Δδ|=" << max_rel
                  << "  ||θ||=" << theta.norm()
                  << std::endl;
    }
    
    if (max_rel <= tol || step_eff_norm <= tol) {
      converged = true;
      it_em++;
      break;
    }
  }
  
  if (verbose && !converged) {
    Rcpp::Rcout << "  [EM] reached max_em=" << max_em
                << "  without convergence; last max rel|Δδ|=" << last_max_rel
                << std::endl;
  }
  
  // posterior responsibilities w_mat（串行）
  Rcpp::NumericMatrix w_mat(n, R);
  std::vector<double> lvec(R), log_phi(R);
  for (int r = 0; r < R; ++r) log_phi[r] = std::log(std::max(delta[r], 1e-300));
  for (int i = 0; i < n; ++i) {
    for (int r = 0; r < R; ++r) lvec[r] = log_fmat(i, r) + log_phi[r];
    const double lse = lse_row(lvec);
    for (int r = 0; r < R; ++r) w_mat(i, r) = std::exp(lvec[r] - lse);
  }
  
  // diagnostics/outputs
  std::vector<int> layer_hamm(R, 0);
  for (int r = 0; r < R; ++r) {
    int s = 0; for (int j = 0; j < cfg.ncol(); ++j) s += (cfg(r, j) != 0);
    layer_hamm[r] = s;
  }
  
  Rcpp::NumericVector delta_out(R);
  for (int r = 0; r < R; ++r) delta_out[r] = delta[r];
  
  auto nms = cfg_names(cfg);
  Rcpp::NumericVector theta_out(R);
  Rcpp::CharacterVector theta_nm(R);
  for (int r = 0; r < R; ++r) { theta_out[r] = theta[r]; theta_nm[r] = nms[r]; }
  theta_out.attr("names") = theta_nm;
  
  Rcpp::IntegerVector layer_vec(R);
  for (int r = 0; r < R; ++r) layer_vec[r] = layer_hamm[r];
  
  std::vector<double> layer_mass_vec(cfg.ncol() + 1, 0.0);
  for (int r = 0; r < R; ++r) layer_mass_vec[layer_hamm[r]] += delta[r];
  Rcpp::NumericVector layer_mass(layer_mass_vec.begin(), layer_mass_vec.end());
  Rcpp::CharacterVector lm_names(cfg.ncol() + 1);
  for (int s = 0; s <= cfg.ncol(); ++s) lm_names[s] = std::to_string(s);
  layer_mass.attr("names") = lm_names;
  
  Rcpp::List diagnostics = Rcpp::List::create(
    Rcpp::_["kappa_gersh_upper"] = kappa_gersh,
    Rcpp::_["eigmax_est"]        = eigmax_est,
    Rcpp::_["eigmin_est"]        = eigmin_est,
    Rcpp::_["kappa_est"]         = kappa_est,
    Rcpp::_["A0_eigmax"]         = eigmax_A0,
    Rcpp::_["H_norm"]            = H_norm,
    Rcpp::_["N_eff"]             = N_eff
  );
  
  return Rcpp::List::create(
    Rcpp::_["delta"]        = delta_out,
    Rcpp::_["theta"]        = theta_out,
    Rcpp::_["layer"]        = layer_vec,      // 诊断
    Rcpp::_["layer_mass"]   = layer_mass,     // 诊断
    Rcpp::_["w_mat"]        = w_mat,
    Rcpp::_["converged"]    = converged,
    Rcpp::_["niter"]        = it_em,
    Rcpp::_["lambda"]       = base_lambda,    // 最终各向同性 λ
    Rcpp::_["diagnostics"]  = diagnostics
  );
}

