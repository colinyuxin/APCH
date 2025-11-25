//calc_density.h
#pragma once

// [[Rcpp::depends(Rcpp)]]
// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <Rmath.h>
#include <RcppEigen.h>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace Rcpp;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::LLT;

using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

extern const double LOG2PI;
extern const double LOG_PI;
extern const double LOG_2;

inline double phi(double x)     { return R::dnorm(x, 0.0, 1.0, false); }
inline double logphi(double x)  { return R::dnorm(x, 0.0, 1.0, true);  }
inline double Phi(double x)     { return R::pnorm(x, 0.0, 1.0, true, false); }
inline double logPhi(double x)  { return R::pnorm(x, 0.0, 1.0, true, true); }
inline bool   isNA(double x)    { return Rcpp::traits::is_na<REALSXP>(x); }

// stably compute log(sum(exp(v)))
static inline double lse_row(const std::vector<double>& v) {
  double m = *std::max_element(v.begin(), v.end());
  if (!std::isfinite(m)) return m;
  double s = 0.0;
  for (double x : v) if (std::isfinite(x)) s += std::exp(x - m);
  return m + std::log(s);
}

// -----------------------------------------------------------------------------
// One-dimensional probability density
// -----------------------------------------------------------------------------

// stably compute Phi(a) - Phi(b)
inline double pnorm_diff(double a, double b) {
  if (a == b) return 0.0;
  if (a < b) return -pnorm_diff(b, a);
  
  double logPa  = R::pnorm(a, 0.0, 1.0, /*lower_tail=*/1, /*log_p=*/1);
  double logPb  = R::pnorm(b, 0.0, 1.0, /*lower_tail=*/1, /*log_p=*/1);
  double logMax = std::max(logPa, logPb);
  double v1     = std::exp(logPa - logMax);
  double v2     = std::exp(logPb - logMax);
  return (v1 - v2) * std::exp(logMax);
}

// 1) Normal prior N(0, sigma^2) conv N(0, s^2) => N(0, s^2 + sigma^2)
inline double conv_normal_normal_density(double x, double s, double sigma) {
  double var = s*s + sigma*sigma;
  double sd  = std::sqrt(var);
  return R::dnorm(x, 0.0, sd, false);
}

// 2) Symmetric uniform prior U[-a, a] conv N(0, s^2).
//    f_obs(x) = [Phi((x + a)/s) - Phi((x - a)/s)] / (2a)
inline double conv_uniform_normal_density(double x, double s, double a) {
  if (a <= 0.0) return conv_normal_normal_density(x, s, 0.0);
  double u1   = (x + a) / s;
  double u2   = (x - a) / s;
  double diff = pnorm_diff(u1, u2);
  return diff / (2.0 * a);
}

// 3) Half-uniform, param > 0 => U[0, a]; param < 0 => U[-a,0].
//    For U[0,a]:   f_obs(x) = [Phi(x/s) - Phi((x - a)/s)] / a
//    For U[-a,0]:  f_obs(x) = [Phi((x + a)/s) - Phi(x/s)] / a
inline double conv_halfuniform_normal_density(double x, double s, double signed_a) {
  if (signed_a == 0.0) return conv_normal_normal_density(x, s, 0.0);
  double a = std::fabs(signed_a);
  if (signed_a > 0.0) {
    double u1 =  x      / s;
    double u2 = (x - a) / s;
    double diff = pnorm_diff(u1, u2);
    return diff / a;
  } else {
    double u1 = (x + a) / s;
    double u2 =  x      / s;
    double diff = pnorm_diff(u1, u2);
    return diff / a;
  }
}

// 4) +uniform（U[0, a]）与 -uniform（U[-a, 0]）
inline double conv_pos_uniform_density(double x, double s, double a) {
  if (a <= 0.0) return conv_normal_normal_density(x, s, 0.0);
  double u1   =  x      / s;
  double u2   = (x - a) / s;
  double diff = pnorm_diff(u1, u2);
  return diff / a;
}

inline double conv_neg_uniform_density(double x, double s, double a) {
  if (a <= 0.0) return conv_normal_normal_density(x, s, 0.0);
  double u1   = (x + a) / s;
  double u2   =  x      / s;
  double diff = pnorm_diff(u1, u2);
  return diff / a;
}

// 5) Half-normal prior conv Normal(0,s^2).
//    f_obs(x) = sqrt(2/pi) * 1/sqrt(s^2 + sigma^2) * exp(-x^2/(2*(s^2+sigma^2)))
//               * Phi( x * sigma / ( s * sqrt(s^2+sigma^2) ) )
inline double conv_halfnormal_normal_density(double x, double s, double signed_sigma) {
  if (signed_sigma == 0.0) return conv_normal_normal_density(x, s, 0.0);
  double sigma   = std::fabs(signed_sigma);
  double var     = s*s + sigma*sigma;
  double logpref = 0.5*(std::log(2.0/M_PI) - std::log(var));
  double exponent= - (x*x) / (2.0*var);
  double denom   = std::sqrt(var);
  double arg     = (x * sigma) / (s * denom);
  double logF    = (signed_sigma >= 0.0) ? logPhi(arg) : logPhi(-arg);
  double logdens = logpref + exponent + logF;
  return std::exp(logdens);
}

// One-dimensional convolution kernel selector
inline double conv_density_by_family(double x, double s, double param, std::string mixcompdist) {
  if (mixcompdist == "normal")        return conv_normal_normal_density(x, s, param);
  else if (mixcompdist == "uniform")  return conv_uniform_normal_density(x, s, std::fabs(param));
  else if (mixcompdist == "halfuniform") return conv_halfuniform_normal_density(x, s, param);
  else if (mixcompdist == "+uniform") return conv_pos_uniform_density(x, s, std::fabs(param));
  else if (mixcompdist == "-uniform") return conv_neg_uniform_density(x, s, std::fabs(param));
  else if (mixcompdist == "halfnormal") return conv_halfnormal_normal_density(x, s, param);
  // fallback: treat as normal
  return conv_normal_normal_density(x, s, std::fabs(param));
}

// -----------------------------------------------------------------------------
// Multivariate normal distribution and conditional distributio
// -----------------------------------------------------------------------------

double ldmvnorm0(const VectorXd& x, const MatrixXd& S);

// truncated MVN integration (Miwa-style)
double mvn_cdf_miwa_cpp(const std::vector<double>& lower,
                        const std::vector<double>& upper,
                        const std::vector<double>& mean,
                        const Eigen::MatrixXd& Sigma,
                        double rel_tol = 1e-6,
                        int    max_depth = 12);

void conditional_gaussian(const Eigen::VectorXd &x,
                          const Eigen::VectorXd &mu,
                          const Eigen::MatrixXd &Sigma,
                          const std::vector<int> &idx_active,
                          const std::vector<int> &idx_null,
                          Eigen::VectorXd &mu_free,
                          Eigen::MatrixXd &Sigma_free,
                          Eigen::VectorXd &x_act,
                          double &log_val);

double normal_cdf_logval(const NumericVector &xrow,
                         const NumericMatrix &Sigma,
                         std::vector<int> &idx_active,
                         std::vector<int> &idx_null, 
                         const std::vector<double> &lower_full,
                         const std::vector<double> &upper_full,
                         const NumericVector &mean_full);

void halfnormal_posterior(const Eigen::MatrixXd &Sigma,
                          const Eigen::VectorXd &y_act,
                          const Eigen::MatrixXd &D_inv,
                          double &multiplier,
                          NumericVector &mean,
                          NumericMatrix &Sigma_A);

// log-likelihood matrix (global C)
RowMat logmat_normal(const NumericMatrix &X,
                     const NumericMatrix &S,
                     const NumericMatrix &Rcor,
                     const std::vector< std::vector<int> > &eta_list,
                     const std::vector< std::vector<double> > &grids);

RowMat logmat_uniform(const NumericMatrix &X,
                      const NumericMatrix &S,
                      const NumericMatrix &Rcor,
                      const std::vector< std::vector<int> > &eta_list,
                      const std::vector< std::vector<double> > &grids,
                      std::string priors);

RowMat logmat_halfnormal(const NumericMatrix &X,
                         const NumericMatrix &S,
                         const NumericMatrix &Rcor,
                         const std::vector< std::vector<int> > &eta_list,
                         const std::vector< std::vector<double> > &grids);

// xwas variants: per-effect noise matrices supplied as a list (length n)
RowMat logmat_normal_xwas(const NumericMatrix &X,
                          const NumericMatrix &S,
                          const List &Rcor_list,
                          const std::vector< std::vector<int> > &eta_list,
                          const std::vector< std::vector<double> > &grids);

RowMat logmat_uniform_xwas(const NumericMatrix &X,
                           const NumericMatrix &S,
                           const List &Rcor_list,
                           const std::vector< std::vector<int> > &eta_list,
                           const std::vector< std::vector<double> > &grids,
                           std::string priors);

RowMat logmat_halfnormal_xwas(const NumericMatrix &X,
                              const NumericMatrix &S,
                              const List &Rcor_list,
                              const std::vector< std::vector<int> > &eta_list,
                              const std::vector< std::vector<double> > &grids);
