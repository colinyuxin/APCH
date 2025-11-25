//calc_density.cpp
#include "calc_density.h"
#include "include/miwa_impl.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

// constants
const double LOG2PI = std::log(2.0 * M_PI);
const double LOG_PI = std::log(M_PI);
const double LOG_2  = std::log(2.0);

// -----------------------------------------------------------------------------
// calculate the joint density under the non-independent scenario
// -----------------------------------------------------------------------------

// logarithm of the density function of the mixture normal distribution: log N(x | 0, S)
double ldmvnorm0(const VectorXd& x, const MatrixXd& S) {
  LLT<MatrixXd> chol(S);
  MatrixXd      L = chol.matrixL();

  double logdet = 2.0 * L.diagonal().array().log().sum();
  VectorXd y    = chol.solve(x);

  return -0.5 * (x.size() * LOG2PI + logdet + x.dot(y));
}

// conditional Gaussian distribution
// given idx_null is fixed, calculate the conditional distribution of the active part and the marginal density corresponding to the null distribution
void conditional_gaussian(const Eigen::VectorXd &x,
                          const Eigen::VectorXd &mu,
                          const Eigen::MatrixXd &Sigma,
                          const std::vector<int> &idx_active,
                          const std::vector<int> &idx_null,
                          Eigen::VectorXd &mu_free,
                          Eigen::MatrixXd &Sigma_free,
                          Eigen::VectorXd &x_act,
                          double &log_val) {

  int k = (int)idx_active.size();
  int m = (int)idx_null.size();

  mu_free   = Eigen::VectorXd::Zero(k);
  Sigma_free= Eigen::MatrixXd::Zero(k, k);
  x_act     = Eigen::VectorXd::Zero(k);

  if (m == 0) {
    // no null distribution, directly take the sub-block
    for (int u = 0; u < k; ++u) {
      int j = idx_active[u];
      mu_free(u) = mu(j);
      x_act(u)   = x(j) - mu_free(u);
      for (int v = 0; v < k; ++v)
        Sigma_free(u,v) = Sigma(idx_active[u], idx_active[v]);
    }
  } else {
    // decompose into block matrices
    Eigen::MatrixXd Sigma_ff(k,k), Sigma_fc(k,m), Sigma_cc(m,m);
    Eigen::VectorXd x_fix(m), mu_fix(m);

    for (int u=0; u<k; ++u) {
      int j = idx_active[u];
      mu_free(u) = mu(j);
      for (int v=0; v<k; ++v)
        Sigma_ff(u,v) = Sigma(idx_active[u], idx_active[v]);
      for (int v=0; v<m; ++v)
        Sigma_fc(u,v) = Sigma(idx_active[u], idx_null[v]);
    }
    for (int u=0; u<m; ++u) {
      int j = idx_null[u];
      x_fix(u) = x(j);
      mu_fix(u)= mu(j);
      for (int v=0; v<m; ++v)
        Sigma_cc(u,v) = Sigma(idx_null[u], idx_null[v]);
    }

    // marginal density of the null distribution
    log_val += ldmvnorm0(x_fix - mu_fix, Sigma_cc);

    // conditional distribution of the active part
    Eigen::MatrixXd Sigma_cc_inv = Sigma_cc.inverse();
    mu_free   = mu_free + Sigma_fc * Sigma_cc_inv * (x_fix - mu_fix);
    Sigma_free= Sigma_ff - Sigma_fc * Sigma_cc_inv * Sigma_fc.transpose();

    for (int u=0; u<k; ++u){
      // avoid singular matrix
      Sigma_free(u, u) += 1e-8;
      x_act(u) = x(idx_active[u]) - mu_free(u);
    }
  }
}

// marginal density of the null distribution * integral of the conditional distribution of the non-null distribution
double normal_cdf_logval(const NumericVector &xrow, // zero vector
                         const NumericMatrix &Sigma,
                         std::vector<int> &idx_active,
                         std::vector<int> &idx_null,
                         const std::vector<double> &lower_full,
                         const std::vector<double> &upper_full,
                         const NumericVector &mean_full) {
  Eigen::VectorXd x_eig    = as<Eigen::VectorXd>(xrow);
  Eigen::MatrixXd Sigma_eig= as<Eigen::MatrixXd>(Sigma);
  Eigen::VectorXd mean_eig = as<Eigen::VectorXd>(mean_full);

  double log_val = 0.0;

  Eigen::VectorXd mu_free, x_act;
  Eigen::MatrixXd Sigma_free;
  conditional_gaussian(x_eig, mean_eig, Sigma_eig, idx_active, idx_null,
                       mu_free, Sigma_free, x_act, log_val);

  if (idx_active.empty()) return log_val;

  int k = (int)idx_active.size();

  // take the lower/upper bounds and mean on the active dimensions
  std::vector<double> lower(k), upper(k), mean(k);
  for (int u=0; u<k; ++u) {
    int j = idx_active[u];
    lower[u] = lower_full[j];
    upper[u] = upper_full[j];
    mean[u]  = mu_free(u);
  }
  //  miwa method
  double prob = mvn_cdf_miwa_cpp(lower, upper, mean, Sigma_free);
  log_val += std::log(prob);
  return log_val;
}


// halfnormal conditional distribution + update of mean and covariance for convolved Gaussian
void halfnormal_posterior(const Eigen::MatrixXd &Sigma,
                          const Eigen::VectorXd &y_act,
                          const Eigen::MatrixXd &D_inv,
                          double &multiplier,
                          NumericVector &mean,
                          NumericMatrix &Sigma_A) {
  int k = (int)y_act.size();
  LLT<MatrixXd> lltSigma(Sigma);
  MatrixXd Sigma_inv = lltSigma.solve(MatrixXd::Identity(k, k));
  MatrixXd A = D_inv + Sigma_inv;
  LLT<MatrixXd> lltA(A);
  MatrixXd A_inv = lltA.solve(MatrixXd::Identity(k, k));

  // log-det
  MatrixXd LA = lltA.matrixL();
  MatrixXd LS = lltSigma.matrixL();
  double logdetA    = 2.0 * LA.diagonal().array().log().sum();
  double logdetSigma= 2.0 * LS.diagonal().array().log().sum();
  multiplier += -0.5 * logdetA - 0.5 * logdetSigma;

  // quadratic form
  VectorXd v = lltSigma.solve(y_act);
  VectorXd w = lltA.solve(v);
  VectorXd u = lltSigma.solve(w);
  double quad = 0.5 * ( y_act.dot(u - v) );
  multiplier += quad;

  mean    = as<NumericVector>(wrap(w));
  Sigma_A = as<NumericMatrix>(wrap(A_inv));
}

// -----------------------------------------------------------------------------
// calculation of log-likelihood for various priors
// -----------------------------------------------------------------------------

// normal prior: Evaluate only on the observed sub-dimensions O_i
RowMat logmat_normal(const NumericMatrix &X,
                     const NumericMatrix &S,
                     const NumericMatrix &Rcor,
                     const std::vector< std::vector<int> > &eta_list,
                     const std::vector< std::vector<double> > &grids) {
  const int n = X.nrow();
  const int p = X.ncol();
  const int L = (int)eta_list.size();
  RowMat logd(n, L);

  for (int i = 0; i < n; ++i) {
    // observed indices O_i
    std::vector<int> O; O.reserve(p);
    for (int j = 0; j < p; ++j) {
      double x = X(i,j), s = S(i,j);
      if (R_finite(x) && R_finite(s)) O.push_back(j);
    }
    if (O.empty()) { // complete missingness: Neutral evidence
      for (int l = 0; l < L; ++l) logd(i,l) = 0.0;
      continue;
    }
    const int o = (int)O.size();

    // Σ_OO & x_O
    Eigen::MatrixXd Sigma_OO(o, o);
    for (int a = 0; a < o; ++a) {
      for (int b = 0; b <= a; ++b) {
        int ja = O[a], jb = O[b];
        double val = S(i,ja) * S(i,jb) * Rcor(ja,jb);
        Sigma_OO(a,b) = val; Sigma_OO(b,a) = val;
      }
    }
    Eigen::VectorXd x_O(o);
    for (int a = 0; a < o; ++a) x_O(a) = X(i, O[a]);

    // subconfiguration
    for (int l = 0; l < L; ++l) {
      // add prior variance on O
      Eigen::MatrixXd Sigma = Sigma_OO; // copy
      for (int a = 0; a < o; ++a) {
        int j = O[a];
        int g = eta_list[l][j];
        double s2 = (g == 0 ? 0.0 : grids[j][g] * grids[j][g]);
        Sigma(a,a) += s2;
      }
      logd(i,l) = ldmvnorm0(x_O, Sigma);
    }
  }
  return logd;
}

// Uniform / HalfUniform / +Uniform / -Uniform prior: evaluate only on O_i
RowMat logmat_uniform(const NumericMatrix &X,
                      const NumericMatrix &S,
                      const NumericMatrix &Rcor,
                      const std::vector< std::vector<int> > &eta_list,
                      const std::vector< std::vector<double> > &grids,
                      std::string priors) {
  const int n = X.nrow();
  const int p = X.ncol();
  const int L = (int)eta_list.size();
  RowMat logd(n, L);

  for (int i = 0; i < n; ++i) {
    // observed indices O_i
    std::vector<int> O; O.reserve(p);
    for (int j = 0; j < p; ++j) {
      double x = X(i,j), s = S(i,j);
      if (R_finite(x) && R_finite(s)) O.push_back(j);
    }
    if (O.empty()) { for (int l=0;l<L;++l) logd(i,l) = 0.0; continue; }
    const int o = (int)O.size();

    // Σ_OO
    NumericMatrix Sigma_OO(o, o);
    for (int a = 0; a < o; ++a) {
      for (int b = 0; b <= a; ++b) {
        int ja = O[a], jb = O[b];
        double val = S(i,ja) * S(i,jb) * Rcor(ja,jb);
        Sigma_OO(a,b) = val; Sigma_OO(b,a) = val;
      }
    }

    // mean_full_O = X_O; xrow_O = zero vector
    NumericVector mean_full_O(o), xrow_O(o);
    for (int a = 0; a < o; ++a) mean_full_O[a] = X(i, O[a]);

    for (int l = 0; l < L; ++l) {
      // active/null coordinates (local indices restricted to O)
      std::vector<int> idx_active_O, idx_null_O;
      idx_active_O.reserve(o); idx_null_O.reserve(o);
      double multiplier = 0.0;

      // lower/upper bounds of o-dimension (integration region)
      std::vector<double> lower_O(o, 0.0), upper_O(o, 0.0);
      bool any_act = false;
      for (int a = 0; a < o; ++a) {
        int j = O[a];
        int g = eta_list[l][j];
        if (g != 0) {
          any_act = true;
          idx_active_O.push_back(a);
          double A = grids[j][g];
          if (priors == "uniform") {
            lower_O[a] = -A; upper_O[a] =  A;
          } else { // halfuniform / +uniform / -uniform
            if (A > 0) { lower_O[a] = 0.0; upper_O[a] =  A; }
            else       { lower_O[a] =  A;  upper_O[a] = 0.0; }
          }
          multiplier += -std::log(std::fabs(upper_O[a]-lower_O[a]));
        } else {
          idx_null_O.push_back(a);
        }
      }

      // no active coordinates on observed dimensions → degenerate to null: density of N(0, Σ_OO)
      if (!any_act) {
        Eigen::VectorXd x_O = as<Eigen::VectorXd>(wrap(mean_full_O)); // X_O
        Eigen::MatrixXd S_O = as<Eigen::MatrixXd>(wrap(Sigma_OO));
        logd(i,l) = ldmvnorm0(x_O, S_O);
        continue;
      }

      double log_val = normal_cdf_logval(xrow_O, Sigma_OO, idx_active_O, idx_null_O,
                                         lower_O, upper_O, mean_full_O);
      logd(i,l) = log_val + multiplier;
    }
  }
  return logd;
}

// HalfNormal prior: evaluate only on O_i (and retain the sign direction)
RowMat logmat_halfnormal(const NumericMatrix &X,
                         const NumericMatrix &S,
                         const NumericMatrix &Rcor,
                         const std::vector< std::vector<int> > &eta_list,
                         const std::vector< std::vector<double> > &grids) {
  const int n = X.nrow();
  const int p = X.ncol();
  const int L = (int)eta_list.size();
  RowMat logd(n, L);

  for (int i = 0; i < n; ++i) {
    // observed indices O_i
    std::vector<int> O; O.reserve(p);
    for (int j = 0; j < p; ++j) {
      double x = X(i,j), s = S(i,j);
      if (R_finite(x) && R_finite(s)) O.push_back(j);
    }
    if (O.empty()) { for (int l=0;l<L;++l) logd(i,l) = 0.0; continue; }
    const int o = (int)O.size();

    // Σ_OO
    NumericMatrix Sigma_OO(o, o);
    for (int a=0; a<o; ++a) {
      for (int b=0; b<=a; ++b) {
        int ja = O[a], jb = O[b];
        double val = S(i,ja) * S(i,jb) * Rcor(ja,jb);
        Sigma_OO(a,b) = val; Sigma_OO(b,a) = val;
      }
    }
    // x_O
    NumericVector x_O(o);
    for (int a=0; a<o; ++a) x_O[a] = X(i, O[a]);

    for (int l = 0; l < L; ++l) {
      // active/null coordinates (local indices restricted to O)
      std::vector<int> idx_active_O, idx_null_O;
      idx_active_O.reserve(o); idx_null_O.reserve(o);

      double multiplier0 = 0.0;
      std::vector<double> var_sig_O(o, 0.0);
      std::vector<int>    sign_dir_O;  // +1: positive half-normal, -1: negative half-normal
      sign_dir_O.reserve(o);

      bool any_act = false;
      for (int a=0; a<o; ++a) {
        int j = O[a];
        int g = eta_list[l][j];
        double sigma = grids[j][g];
        var_sig_O[a] = (g==0 ? 0.0 : sigma*sigma);
        if (g != 0) {
          any_act = true;
          idx_active_O.push_back(a);
          multiplier0 += 0.5*LOG_2 - std::log(std::fabs(sigma)) - 0.5*LOG_PI;
          sign_dir_O.push_back( sigma >= 0.0 ? +1 : -1 );
        } else {
          idx_null_O.push_back(a);
        }
      }

      // no observed active coordinates → degenerate to null
      if (!any_act) {
        Eigen::VectorXd vx = as<Eigen::VectorXd>(x_O);
        Eigen::MatrixXd Sm = as<Eigen::MatrixXd>(wrap(Sigma_OO));
        logd(i,l) = ldmvnorm0(vx, Sm);
        continue;
      }

      // conditional Gaussian (o-dimension)
      double multiplier = multiplier0;
      NumericVector mean_O(o); // all zeros
      Eigen::VectorXd mu_free, x_act;
      Eigen::MatrixXd Sigma_free;
      {
        Eigen::VectorXd x_eig = as<Eigen::VectorXd>(x_O);
        Eigen::VectorXd mu_eig= Eigen::VectorXd::Zero(o);
        Eigen::MatrixXd Sigma_eig = as<Eigen::MatrixXd>(wrap(Sigma_OO));
        conditional_gaussian(x_eig, mu_eig, Sigma_eig, idx_active_O, idx_null_O,
                             mu_free, Sigma_free, x_act, multiplier);
      }

      // D_inv (active dimensions only)
      const int k = (int)idx_active_O.size();
      Eigen::MatrixXd D_inv = Eigen::MatrixXd::Zero(k,k);
      for (int u = 0; u < k; ++u) {
        double vs = var_sig_O[ idx_active_O[u] ];
        if (vs <= 0.0) vs = 1e-12;
        D_inv(u,u) = 1.0 / vs;
      }

      // half-normal posterior & truncated integration
      NumericVector mean_post;
      NumericMatrix Sigma_A;
      halfnormal_posterior(Sigma_free, x_act, D_inv, multiplier, mean_post, Sigma_A);

      // set the integration interval of active dimensions according to the sign of sigma
      std::vector<double> lower_act(k), upper_act(k), mean_std(k);
      for (int u=0; u<k; ++u) {
        if (sign_dir_O[u] >= 0) {
          lower_act[u] = 0.0;    upper_act[u] = 1000.0;
        } else {
          lower_act[u] = -1000.0; upper_act[u] = 0.0;
        }
        mean_std[u] = mean_post[u];
      }

      double prob = mvn_cdf_miwa_cpp(lower_act, upper_act, mean_std,
                                     as<Eigen::MatrixXd>(Sigma_A));
      double log_val = std::log(prob);
      logd(i,l) = log_val + multiplier;
    }
  }
  return logd;
}



// ============================================================================
// xwas versions: per-effect R_cor (C_i), supplied as List
// ============================================================================

// Normal prior with per-effect R_cor (C_i)
RowMat logmat_normal_xwas(const NumericMatrix &X,
                          const NumericMatrix &S,
                          const List &Rcor_list,
                          const std::vector< std::vector<int> > &eta_list,
                          const std::vector< std::vector<double> > &grids) {
  const int n = X.nrow();
  const int p = X.ncol();
  const int L = (int)eta_list.size();
  RowMat logd(n, L);
  
  if (Rcor_list.size() != n)
    stop("logmat_normal_xwas: Rcor_list length must equal nrow(X).");
  
  // Wrap each element as NumericMatrix (no copy; just SEXP handle)
  std::vector<NumericMatrix> R_list(n);
  for (int i = 0; i < n; ++i) {
    R_list[i] = Rcpp::as<NumericMatrix>(Rcor_list[i]);
    if (R_list[i].nrow() != p || R_list[i].ncol() != p)
      stop("logmat_normal_xwas: each Rcor_list[[i]] must be p x p.");
  }
  
  for (int i = 0; i < n; ++i) {
    NumericMatrix Rcor_i = R_list[i];
    
    // observed indices O_i
    std::vector<int> O; O.reserve(p);
    for (int j = 0; j < p; ++j) {
      double x = X(i,j), s = S(i,j);
      if (R_finite(x) && R_finite(s)) O.push_back(j);
    }
    if (O.empty()) {
      for (int l = 0; l < L; ++l) logd(i,l) = 0.0;
      continue;
    }
    const int o = (int)O.size();
    
    // Σ_OO & x_O
    Eigen::MatrixXd Sigma_OO(o, o);
    for (int a = 0; a < o; ++a) {
      for (int b = 0; b <= a; ++b) {
        int ja = O[a], jb = O[b];
        double val = S(i,ja) * S(i,jb) * Rcor_i(ja,jb);
        Sigma_OO(a,b) = val; Sigma_OO(b,a) = val;
      }
    }
    Eigen::VectorXd x_O(o);
    for (int a = 0; a < o; ++a) x_O(a) = X(i, O[a]);
    
    // subconfiguration
    for (int l = 0; l < L; ++l) {
      Eigen::MatrixXd Sigma = Sigma_OO; // copy
      for (int a = 0; a < o; ++a) {
        int j = O[a];
        int g = eta_list[l][j];
        double s2 = (g == 0 ? 0.0 : grids[j][g] * grids[j][g]);
        Sigma(a,a) += s2;
      }
      logd(i,l) = ldmvnorm0(x_O, Sigma);
    }
  }
  return logd;
}

// Uniform / HalfUniform / +Uniform / -Uniform prior with per-effect R_cor (C_i)
RowMat logmat_uniform_xwas(const NumericMatrix &X,
                           const NumericMatrix &S,
                           const List &Rcor_list,
                           const std::vector< std::vector<int> > &eta_list,
                           const std::vector< std::vector<double> > &grids,
                           std::string priors) {
  const int n = X.nrow();
  const int p = X.ncol();
  const int L = (int)eta_list.size();
  RowMat logd(n, L);
  
  if (Rcor_list.size() != n)
    stop("logmat_uniform_xwas: Rcor_list length must equal nrow(X).");
  
  std::vector<NumericMatrix> R_list(n);
  for (int i = 0; i < n; ++i) {
    R_list[i] = Rcpp::as<NumericMatrix>(Rcor_list[i]);
    if (R_list[i].nrow() != p || R_list[i].ncol() != p)
      stop("logmat_uniform_xwas: each Rcor_list[[i]] must be p x p.");
  }
  
  for (int i = 0; i < n; ++i) {
    NumericMatrix Rcor_i = R_list[i];
    
    // observed indices O_i
    std::vector<int> O; O.reserve(p);
    for (int j = 0; j < p; ++j) {
      double x = X(i,j), s = S(i,j);
      if (R_finite(x) && R_finite(s)) O.push_back(j);
    }
    if (O.empty()) { for (int l=0;l<L;++l) logd(i,l) = 0.0; continue; }
    const int o = (int)O.size();
    
    // Σ_OO
    NumericMatrix Sigma_OO(o, o);
    for (int a = 0; a < o; ++a) {
      for (int b = 0; b <= a; ++b) {
        int ja = O[a], jb = O[b];
        double val = S(i,ja) * S(i,jb) * Rcor_i(ja,jb);
        Sigma_OO(a,b) = val; Sigma_OO(b,a) = val;
      }
    }
    
    // mean_full_O = X_O; xrow_O = zero vector
    NumericVector mean_full_O(o), xrow_O(o);
    for (int a = 0; a < o; ++a) mean_full_O[a] = X(i, O[a]);
    
    for (int l = 0; l < L; ++l) {
      // active/null coordinates (local indices restricted to O)
      std::vector<int> idx_active_O, idx_null_O;
      idx_active_O.reserve(o); idx_null_O.reserve(o);
      double multiplier = 0.0;
      
      // lower/upper bounds of o-dimension (integration region)
      std::vector<double> lower_O(o, 0.0), upper_O(o, 0.0);
      bool any_act = false;
      for (int a = 0; a < o; ++a) {
        int j = O[a];
        int g = eta_list[l][j];
        if (g != 0) {
          any_act = true;
          idx_active_O.push_back(a);
          double A = grids[j][g];
          if (priors == "uniform") {
            lower_O[a] = -A; upper_O[a] =  A;
          } else { // halfuniform / +uniform / -uniform
            if (A > 0) { lower_O[a] = 0.0; upper_O[a] =  A; }
            else       { lower_O[a] =  A;  upper_O[a] = 0.0; }
          }
          multiplier += -std::log(std::fabs(upper_O[a]-lower_O[a]));
        } else {
          idx_null_O.push_back(a);
        }
      }
      
      // no active coordinates on observed dimensions → degenerate to null: density of N(0, Σ_OO)
      if (!any_act) {
        Eigen::VectorXd x_O = as<Eigen::VectorXd>(wrap(mean_full_O)); // X_O
        Eigen::MatrixXd S_O = as<Eigen::MatrixXd>(wrap(Sigma_OO));
        logd(i,l) = ldmvnorm0(x_O, S_O);
        continue;
      }
      
      double log_val = normal_cdf_logval(xrow_O, Sigma_OO, idx_active_O, idx_null_O,
                                         lower_O, upper_O, mean_full_O);
      logd(i,l) = log_val + multiplier;
    }
  }
  return logd;
}

// HalfNormal prior with per-effect R_cor (C_i)
RowMat logmat_halfnormal_xwas(const NumericMatrix &X,
                              const NumericMatrix &S,
                              const List &Rcor_list,
                              const std::vector< std::vector<int> > &eta_list,
                              const std::vector< std::vector<double> > &grids) {
  const int n = X.nrow();
  const int p = X.ncol();
  const int L = (int)eta_list.size();
  RowMat logd(n, L);
  
  if (Rcor_list.size() != n)
    stop("logmat_halfnormal_xwas: Rcor_list length must equal nrow(X).");
  
  std::vector<NumericMatrix> R_list(n);
  for (int i = 0; i < n; ++i) {
    R_list[i] = Rcpp::as<NumericMatrix>(Rcor_list[i]);
    if (R_list[i].nrow() != p || R_list[i].ncol() != p)
      stop("logmat_halfnormal_xwas: each Rcor_list[[i]] must be p x p.");
  }
  
  for (int i = 0; i < n; ++i) {
    NumericMatrix Rcor_i = R_list[i];
    
    // observed indices O_i
    std::vector<int> O; O.reserve(p);
    for (int j = 0; j < p; ++j) {
      double x = X(i,j), s = S(i,j);
      if (R_finite(x) && R_finite(s)) O.push_back(j);
    }
    if (O.empty()) { for (int l=0;l<L;++l) logd(i,l) = 0.0; continue; }
    const int o = (int)O.size();
    
    // Σ_OO
    NumericMatrix Sigma_OO(o, o);
    for (int a=0; a<o; ++a) {
      for (int b=0; b<=a; ++b) {
        int ja = O[a], jb = O[b];
        double val = S(i,ja) * S(i,jb) * Rcor_i(ja,jb);
        Sigma_OO(a,b) = val; Sigma_OO(b,a) = val;
      }
    }
    // x_O
    NumericVector x_O(o);
    for (int a=0; a<o; ++a) x_O[a] = X(i, O[a]);
    
    for (int l = 0; l < L; ++l) {
      // active/null coordinates (local indices restricted to O)
      std::vector<int> idx_active_O, idx_null_O;
      idx_active_O.reserve(o); idx_null_O.reserve(o);
      
      double multiplier0 = 0.0;
      std::vector<double> var_sig_O(o, 0.0);
      std::vector<int>    sign_dir_O;  // +1: positive half-normal, -1: negative half-normal
      sign_dir_O.reserve(o);
      
      bool any_act = false;
      for (int a=0; a<o; ++a) {
        int j = O[a];
        int g = eta_list[l][j];
        double sigma = grids[j][g];
        var_sig_O[a] = (g==0 ? 0.0 : sigma*sigma);
        if (g != 0) {
          any_act = true;
          idx_active_O.push_back(a);
          multiplier0 += 0.5*LOG_2 - std::log(std::fabs(sigma)) - 0.5*LOG_PI;
          sign_dir_O.push_back( sigma >= 0.0 ? +1 : -1 );
        } else {
          idx_null_O.push_back(a);
        }
      }
      
      // no observed active coordinates → degenerate to null
      if (!any_act) {
        Eigen::VectorXd vx = as<Eigen::VectorXd>(x_O);
        Eigen::MatrixXd Sm = as<Eigen::MatrixXd>(wrap(Sigma_OO));
        logd(i,l) = ldmvnorm0(vx, Sm);
        continue;
      }
      
      // conditional Gaussian (o-dimension)
      double multiplier = multiplier0;
      NumericVector mean_O(o); // all zeros
      Eigen::VectorXd mu_free, x_act;
      Eigen::MatrixXd Sigma_free;
      {
        Eigen::VectorXd x_eig = as<Eigen::VectorXd>(x_O);
        Eigen::VectorXd mu_eig= Eigen::VectorXd::Zero(o);
        Eigen::MatrixXd Sigma_eig = as<Eigen::MatrixXd>(wrap(Sigma_OO));
        conditional_gaussian(x_eig, mu_eig, Sigma_eig, idx_active_O, idx_null_O,
                             mu_free, Sigma_free, x_act, multiplier);
      }
      
      // D_inv (active dimensions only)
      const int k = (int)idx_active_O.size();
      Eigen::MatrixXd D_inv = Eigen::MatrixXd::Zero(k,k);
      for (int u = 0; u < k; ++u) {
        double vs = var_sig_O[ idx_active_O[u] ];
        if (vs <= 0.0) vs = 1e-12;
        D_inv(u,u) = 1.0 / vs;
      }
      
      // half-normal posterior & truncated integration
      NumericVector mean_post;
      NumericMatrix Sigma_A;
      halfnormal_posterior(Sigma_free, x_act, D_inv, multiplier, mean_post, Sigma_A);
      
      // set the integration interval of active dimensions according to the sign of sigma
      std::vector<double> lower_act(k), upper_act(k), mean_std(k);
      for (int u=0; u<k; ++u) {
        if (sign_dir_O[u] >= 0) {
          lower_act[u] = 0.0;     upper_act[u] = 1000.0;
        } else {
          lower_act[u] = -1000.0; upper_act[u] = 0.0;
        }
        mean_std[u] = mean_post[u];
      }
      
      double prob = mvn_cdf_miwa_cpp(lower_act, upper_act, mean_std,
                                     as<Eigen::MatrixXd>(Sigma_A));
      double log_val = std::log(prob);
      logd(i,l) = log_val + multiplier;
    }
  }
  return logd;
}

