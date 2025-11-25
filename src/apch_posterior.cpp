//apch_posterior.cpp

// [[Rcpp::depends(Rcpp,RcppEigen)]]
// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <limits>
#include <algorithm>

using namespace Rcpp;

// small eps for variance
static inline double _eps_var() { return 1e-12; }

// compute vbar per feature from grids + pi (only non-zero slab)
static std::vector<double>
  vbar_from_grids_cpp(const Rcpp::List& grid_list, const Rcpp::List& pi_list) {
    const int p = grid_list.size();
    std::vector<double> vbar(p, 0.0);
    for (int j = 0; j < p; ++j) {
      Rcpp::NumericVector sig = grid_list[j];
      Rcpp::NumericVector pik = pi_list[j];
      double mass = 0.0, acc = 0.0;
      if (sig.size() == pik.size() && sig.size() > 0) {
        for (int t = 0; t < sig.size(); ++t) {
          const double s = sig[t];
          const double w = pik[t];
          if (s > 0.0 && std::isfinite(w)) {
            acc  += w * (s * s);
            mass += w;
          }
        }
      }
      double vb = (mass > 0.0 ? (acc / mass) : 0.0);
      if (!std::isfinite(vb) || vb <= 0.0) vb = _eps_var();
      vbar[j] = vb;
    }
    return vbar;
  }

// ---------- meta version ----------
  // [[Rcpp::export]]
Rcpp::List gaussian_posterior_cpp(const Rcpp::NumericMatrix &X,
                                  const Rcpp::NumericMatrix &S,
                                  const Rcpp::NumericMatrix &R_cor,
                                  const Rcpp::NumericMatrix &w_mat,
                                  const Rcpp::IntegerMatrix &cfg,
                                  const Rcpp::List &grid_list,
                                  const Rcpp::List &pi_list,
                                  bool return_cov_full = true,
                                  double w_thresh = 1e-12) {
  const int n = X.nrow();
  const int p = X.ncol();
  const int R = w_mat.ncol();
  if (S.nrow() != n || S.ncol() != p)
    Rcpp::stop("gaussian_posterior_cpp: dim(S) must equal dim(X).");
  if (cfg.ncol() != p) Rcpp::stop("cfg must have p columns.");
  if (cfg.nrow() != R) Rcpp::stop("nrow(cfg) must equal ncol(w_mat).");
  
  std::vector<double> vbar = vbar_from_grids_cpp(grid_list, pi_list);
  
  Eigen::MatrixXd Rcor = Rcpp::as<Eigen::MatrixXd>(R_cor);
  Rcor = 0.5 * (Rcor + Rcor.transpose());
  Eigen::LLT<Eigen::MatrixXd> lltR(Rcor);
  if (lltR.info() != Eigen::Success) {
    const double jitter = 1e-10;
    Rcor += jitter * Eigen::MatrixXd::Identity(p, p);
    lltR.compute(Rcor);
    if (lltR.info() != Eigen::Success) Rcpp::stop("LLT on R_cor failed.");
  }
  Eigen::MatrixXd R_inv = lltR.solve(Eigen::MatrixXd::Identity(p, p));
  
  std::vector<std::vector<int>> active_sets(R);
  for (int r = 0; r < R; ++r) {
    std::vector<int> Sidx; Sidx.reserve(p);
    for (int j = 0; j < p; ++j) if (cfg(r, j) != 0) Sidx.push_back(j);
    active_sets[r] = std::move(Sidx);
  }
  
  Rcpp::NumericMatrix post_mean(n, p);
  Rcpp::NumericMatrix var_diag(n, p);
  Rcpp::NumericVector cov_full;
  bool need_full = return_cov_full;
  if (need_full) {
    cov_full = Rcpp::NumericVector(p * p * n);
    Rcpp::IntegerVector dim = Rcpp::IntegerVector::create(p, p, n);
    cov_full.attr("dim") = dim;
  }
  
  Rcpp::List dnX = X.attr("dimnames");
  Rcpp::CharacterVector eff_names, feat_names;
  if (dnX.size() == 2) { eff_names = dnX[0]; feat_names = dnX[1]; }
  if (feat_names.size() == p) { Rcpp::colnames(post_mean) = feat_names; Rcpp::colnames(var_diag) = feat_names; }
  if (eff_names.size() == n) { Rcpp::rownames(post_mean) = eff_names; Rcpp::rownames(var_diag) = eff_names; }
  
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd xi(p), si(p);
    double smin_pos = std::numeric_limits<double>::infinity();
    for (int j = 0; j < p; ++j) {
      const double xij = X(i, j), sij = S(i, j);
      xi(j) = xij; si(j) = sij;
      if (std::isfinite(sij) && sij > 0.0) smin_pos = std::min(smin_pos, sij);
    }
    if (!std::isfinite(smin_pos)) smin_pos = 1.0;
    for (int j = 0; j < p; ++j) if (!std::isfinite(si(j)) || si(j) <= 0.0) si(j) = smin_pos;
    
    Eigen::VectorXd u  = xi.array() / si.array();
    Eigen::VectorXd Rinv_u = R_inv * u;
    
    Eigen::VectorXd mu_sum = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd diag_V_acc = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd diag_mu2_acc = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd S2_sum;
    if (need_full) S2_sum = Eigen::MatrixXd::Zero(p, p);
    
    for (int r = 0; r < R; ++r) {
      const double w = w_mat(i, r);
      if (!(w > w_thresh)) continue;
      const std::vector<int>& Sidx = active_sets[r];
      const int k = (int)Sidx.size();
      if (k == 0) continue;
      
      Eigen::VectorXd sS(k), Q_Sx(k);
      for (int a = 0; a < k; ++a) {
        const int ja = Sidx[a];
        sS(a)   = si(ja);
        Q_Sx(a) = Rinv_u(ja) / si(ja);
      }
      Eigen::MatrixXd Q_SS(k, k);
      for (int a = 0; a < k; ++a)
        for (int b = 0; b < k; ++b) {
          const int ja = Sidx[a], jb = Sidx[b];
          Q_SS(a, b) = R_inv(ja, jb) / (sS(a) * sS(b));
        }
      
      Eigen::MatrixXd D_inv = Eigen::MatrixXd::Zero(k, k);
      for (int a = 0; a < k; ++a) {
        const double vb = std::max(vbar[Sidx[a]], _eps_var());
        D_inv(a, a) = 1.0 / vb;
      }
      
      Eigen::MatrixXd Prec_S = Q_SS; Prec_S += D_inv;
      Eigen::LLT<Eigen::MatrixXd> lltS(Prec_S);
      if (lltS.info() != Eigen::Success) {
        Prec_S.diagonal().array() += 1e-12; lltS.compute(Prec_S);
        if (lltS.info() != Eigen::Success)
          Rcpp::stop("gaussian_posterior_cpp: LLT on Prec_S failed (i=%d, r=%d).", i+1, r+1);
      }
      
      Eigen::VectorXd M_S = lltS.solve(Q_Sx);
      
      for (int a = 0; a < k; ++a) {
        const int ja = Sidx[a]; const double m = M_S(a);
        mu_sum(ja) += w * m;
        if (!need_full) diag_mu2_acc(ja) += w * (m * m);
      }
      
      Eigen::MatrixXd V_S = lltS.solve(Eigen::MatrixXd::Identity(k, k));
      if (need_full) {
        for (int a = 0; a < k; ++a)
          for (int b = 0; b < k; ++b)
            S2_sum(Sidx[a], Sidx[b]) += w * V_S(a, b);
            for (int a = 0; a < k; ++a) {
              const int ja = Sidx[a]; const double ma = M_S(a);
              for (int b = 0; b < k; ++b) {
                const int jb = Sidx[b];
                S2_sum(ja, jb) += w * (ma * M_S(b));
              }
            }
      } else {
        for (int a = 0; a < k; ++a) {
          const int ja = Sidx[a];
          diag_V_acc(ja) += w * V_S(a, a);
        }
      }
    }
    
    for (int j = 0; j < p; ++j) post_mean(i, j) = mu_sum(j);
    
    if (need_full) {
      for (int a = 0; a < p; ++a)
        for (int b = 0; b < p; ++b) {
          const double val = S2_sum(a, b) - mu_sum(a) * mu_sum(b);
          const std::size_t idx = (std::size_t)a + (std::size_t)p * (std::size_t)b + (std::size_t)p * (std::size_t)p * (std::size_t)i;
          cov_full[idx] = val;
        }
      for (int j = 0; j < p; ++j)
        var_diag(i, j) = cov_full[j + p * j + p * p * i];
    } else {
      for (int j = 0; j < p; ++j) {
        double vj = diag_V_acc(j) + diag_mu2_acc(j) - mu_sum(j) * mu_sum(j);
        var_diag(i, j) = vj;
      }
    }
  }
  
  Rcpp::List out;
  out["mean"]     = post_mean;
  out["var_diag"] = var_diag;
  if (need_full) out["cov_full"] = cov_full; else out["cov_full"] = R_NilValue;
  out["note"] = R_NilValue;
  return out;
}

// ---------- xwas version ----------
  // [[Rcpp::export]]
Rcpp::List gaussian_posterior_xwas_cpp(const Rcpp::NumericMatrix &X,
                                       const Rcpp::NumericMatrix &S,
                                       const Rcpp::List &C_list,
                                       const Rcpp::NumericMatrix &w_mat,
                                       const Rcpp::IntegerMatrix &cfg,
                                       const Rcpp::List &grid_list,
                                       const Rcpp::List &pi_list,
                                       bool return_cov_full = true,
                                       double w_thresh = 1e-12) {
  const int n = X.nrow();
  const int p = X.ncol();
  const int R = w_mat.ncol();
  if (S.nrow() != n || S.ncol() != p)
    Rcpp::stop("gaussian_posterior_xwas_cpp: dim(S) must equal dim(X).");
  if (cfg.ncol() != p)
    Rcpp::stop("gaussian_posterior_xwas_cpp: cfg must have p columns.");
  if (cfg.nrow() != R)
    Rcpp::stop("gaussian_posterior_xwas_cpp: nrow(cfg) must equal ncol(w_mat).");
  if (C_list.size() != n)
    Rcpp::stop("gaussian_posterior_xwas_cpp: C_list length must equal nrow(X).");
  
  std::vector<double> vbar = vbar_from_grids_cpp(grid_list, pi_list);
  
  std::vector<Rcpp::NumericMatrix> C_mats(n);
  for (int i = 0; i < n; ++i) {
    C_mats[i] = Rcpp::as<Rcpp::NumericMatrix>(C_list[i]);
    if (C_mats[i].nrow() != p || C_mats[i].ncol() != p)
      Rcpp::stop("gaussian_posterior_xwas_cpp: each C_list[[i]] must be p x p.");
  }
  
  Rcpp::NumericMatrix post_mean(n, p);
  Rcpp::NumericMatrix var_diag(n, p);
  Rcpp::NumericVector cov_full;
  bool need_full = return_cov_full;
  if (need_full) {
    cov_full = Rcpp::NumericVector(p * p * n);
    Rcpp::IntegerVector dim = Rcpp::IntegerVector::create(p, p, n);
    cov_full.attr("dim") = dim;
  }
  
  Rcpp::List dnX = X.attr("dimnames");
  Rcpp::CharacterVector eff_names, feat_names;
  if (dnX.size() == 2) { eff_names = dnX[0]; feat_names = dnX[1]; }
  if (feat_names.size() == p) { Rcpp::colnames(post_mean) = feat_names; Rcpp::colnames(var_diag) = feat_names; }
  if (eff_names.size() == n) { Rcpp::rownames(post_mean) = eff_names; Rcpp::rownames(var_diag) = eff_names; }
  
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(p, p);
  
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd xi(p), si(p);
    double smin_pos = std::numeric_limits<double>::infinity();
    for (int j = 0; j < p; ++j) {
      const double xij = X(i, j), sij = S(i, j);
      xi(j) = xij; si(j) = sij;
      if (std::isfinite(sij) && sij > 0.0) smin_pos = std::min(smin_pos, sij);
    }
    if (!std::isfinite(smin_pos)) smin_pos = 1.0;
    for (int j = 0; j < p; ++j) if (!std::isfinite(si(j)) || si(j) <= 0.0) si(j) = smin_pos;
    
    Eigen::MatrixXd Rcor_i = Rcpp::as<Eigen::MatrixXd>(C_mats[i]);
    Rcor_i = 0.5 * (Rcor_i + Rcor_i.transpose());
    Eigen::LLT<Eigen::MatrixXd> lltR(Rcor_i);
    if (lltR.info() != Eigen::Success) {
      const double jitter = 1e-10; Rcor_i += jitter * Eigen::MatrixXd::Identity(p, p);
      lltR.compute(Rcor_i);
      if (lltR.info() != Eigen::Success) Rcpp::stop("LLT on C_list[[i]] failed (i=%d).", i+1);
    }
    Eigen::MatrixXd R_inv = lltR.solve(I);
    
    Eigen::VectorXd u  = xi.array() / si.array();
    Eigen::VectorXd Rinv_u = R_inv * u;
    
    Eigen::VectorXd mu_sum = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd diag_V_acc = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd diag_mu2_acc = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd S2_sum;
    if (need_full) S2_sum = Eigen::MatrixXd::Zero(p, p);
    
    std::vector<std::vector<int>> active_sets(R);
    for (int r = 0; r < R; ++r) {
      std::vector<int> Sidx; Sidx.reserve(p);
      for (int j = 0; j < p; ++j) if (cfg(r, j) != 0) Sidx.push_back(j);
      active_sets[r] = std::move(Sidx);
    }
    
    for (int r = 0; r < R; ++r) {
      const double w = w_mat(i, r);
      if (!(w > w_thresh)) continue;
      const std::vector<int>& Sidx = active_sets[r];
      const int k = (int)Sidx.size();
      if (k == 0) continue;
      
      Eigen::VectorXd sS(k), Q_Sx(k);
      for (int a = 0; a < k; ++a) { const int ja = Sidx[a]; sS(a) = si(ja); Q_Sx(a) = Rinv_u(ja) / si(ja); }
      Eigen::MatrixXd Q_SS(k, k);
      for (int a = 0; a < k; ++a)
        for (int b = 0; b < k; ++b) {
          const int ja = Sidx[a], jb = Sidx[b];
          Q_SS(a, b) = R_inv(ja, jb) / (sS(a) * sS(b));
        }
      
      Eigen::MatrixXd D_inv = Eigen::MatrixXd::Zero(k, k);
      for (int a = 0; a < k; ++a) { const double vb = std::max(vbar[Sidx[a]], _eps_var()); D_inv(a, a) = 1.0 / vb; }
      
      Eigen::MatrixXd Prec_S = Q_SS; Prec_S += D_inv;
      Eigen::LLT<Eigen::MatrixXd> lltS(Prec_S);
      if (lltS.info() != Eigen::Success) {
        Prec_S.diagonal().array() += 1e-12; lltS.compute(Prec_S);
        if (lltS.info() != Eigen::Success)
          Rcpp::stop("gaussian_posterior_xwas_cpp: LLT on Prec_S failed (i=%d, r=%d).", i+1, r+1);
      }
      
      Eigen::VectorXd M_S = lltS.solve(Q_Sx);
      
      for (int a = 0; a < k; ++a) {
        const int ja = Sidx[a]; const double m = M_S(a);
        mu_sum(ja) += w * m;
        if (!need_full) diag_mu2_acc(ja) += w * (m * m);
      }
      
      Eigen::MatrixXd V_S = lltS.solve(Eigen::MatrixXd::Identity(k, k));
      if (need_full) {
        for (int a = 0; a < k; ++a)
          for (int b = 0; b < k; ++b)
            S2_sum(Sidx[a], Sidx[b]) += w * V_S(a, b);
            for (int a = 0; a < k; ++a) {
              const int ja = Sidx[a]; const double ma = M_S(a);
              for (int b = 0; b < k; ++b) S2_sum(ja, Sidx[b]) += w * (ma * M_S(b));
            }
      } else {
        for (int a = 0; a < k; ++a) {
          const int ja = Sidx[a];
          diag_V_acc(ja) += w * V_S(a, a);
        }
      }
    }
    
    for (int j = 0; j < p; ++j) post_mean(i, j) = mu_sum(j);
    
    if (need_full) {
      for (int a = 0; a < p; ++a)
        for (int b = 0; b < p; ++b) {
          const double val = S2_sum(a, b) - mu_sum(a) * mu_sum(b);
          const std::size_t idx = (std::size_t)a + (std::size_t)p * (std::size_t)b + (std::size_t)p * (std::size_t)p * (std::size_t)i;
          cov_full[idx] = val;
        }
      for (int j = 0; j < p; ++j)
        var_diag(i, j) = cov_full[j + p * j + p * p * i];
    } else {
      for (int j = 0; j < p; ++j) {
        double vj = diag_V_acc(j) + diag_mu2_acc(j) - mu_sum(j) * mu_sum(j);
        var_diag(i, j) = vj;
      }
    }
  }
  
  Rcpp::List out;
  out["mean"]     = post_mean;
  out["var_diag"] = var_diag;
  if (need_full) out["cov_full"] = cov_full; else out["cov_full"] = R_NilValue;
  out["note"] = R_NilValue;
  return out;
}
