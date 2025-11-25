#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>

#include <Rmath.h>
#include <RcppEigen.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// safe Φ and Φ⁻¹ (avoiding 0/1 overflow)
static inline double safe_pnorm(double x) {
  return R::pnorm(x, 0.0, 1.0, /*lower=*/1, /*log_p=*/0);
}
static inline double safe_qnorm(double u) {
  const double eps = 1e-15;
  if (u <= 0.0) u = eps;
  else if (u >= 1.0) u = 1.0 - eps;
  return R::qnorm(u, 0.0, 1.0, /*lower=*/1, /*log_p=*/0);
}

// adaptive simpson (performed in U space, interval [ua, ub], integrand is g(u))
struct AdaptiveSimpson {
  int  max_depth;
  double eps;
  
  AdaptiveSimpson(double tol, int depth) : max_depth(depth), eps(tol) {}
  
  static inline double simpson_est(double fa, double fm, double fb, double a, double b) {
    return (b - a) * (fa + 4.0*fm + fb) / 6.0;
  }
  
  double recurse(const std::function<double(double)>& g,
                 double a, double b, double fa, double fm, double fb,
                 double whole, int depth) {
    const double m  = 0.5 * (a + b);
    const double lm = 0.5 * (a + m);
    const double rm = 0.5 * (m + b);
    
    const double flm = g(lm);
    const double frm = g(rm);
    
    const double left  = simpson_est(fa, flm, fm, a, m);
    const double right = simpson_est(fm, frm, fb, m, b);
    const double delta = left + right - whole;
    
    if (depth <= 0 || std::fabs(delta) < 15.0 * eps) {
      // richardson extrapolation
      return left + right + delta / 15.0;
    }
    return recurse(g, a, m, fa, flm, fm, left,  depth - 1)
      + recurse(g, m, b, fm, frm, fb, right, depth - 1);
  }
  
  double integrate(const std::function<double(double)>& g, double a, double b) {
    if (a >= b) return 0.0;
    const double m  = 0.5 * (a + b);
    const double fa = g(a), fm = g(m), fb = g(b);
    const double whole = simpson_est(fa, fm, fb, a, b);
    return recurse(g, a, b, fa, fm, fb, whole, max_depth);
  }
};

// miwa-style recursor
struct MiwaRecursor {
  int d;
  MatrixXd L;                    // Cholesky of correlation R (lower-triangular)
  std::vector<double> zl, zu;    // normalized bounds in Z space
  double rel_tol;
  int    max_depth;
  
  MiwaRecursor(const MatrixXd& R,
               const std::vector<double>& lower_z,
               const std::vector<double>& upper_z,
               double tol, int depth)
    : d((int)lower_z.size()),
      L(R),
      zl(lower_z),
      zu(upper_z),
      rel_tol(tol),
      max_depth(depth) {}
  
  // compute a_k/b_k given prefix v[0..k-2]
  inline void compute_bounds_k(int k, const std::vector<double>& v_prefix,
                               double& a, double& b) const {
    // k index from 1..d , v_prefix length = k-1
    double sum = 0.0;
    for (int j = 1; j <= k-1; ++j) sum += L(k-1, j-1) * v_prefix[j-1];
    const double Lkk = L(k-1, k-1);
    a = (zl[k-1] - sum) / Lkk;
    b = (zu[k-1] - sum) / Lkk;
    if (a > b) std::swap(a, b);
  }
  
  // recursion: probability from layer k (1-based)
  double rec(int k, const std::vector<double>& v_prefix) const {
    double a, b;
    compute_bounds_k(k, v_prefix, a, b);
    if (a >= b) return 0.0;
    
    if (k == d) {
      // last layer: Φ(b) - Φ(a)
      return safe_pnorm(b) - safe_pnorm(a);
    }
    
    // integrate φ(v_k) * rec(k+1) with respect to v_k for v_k
    // variable substitution: u = Φ(v_k), then dv_k * φ(v_k) = du, interval [ua, ub]
    const double ua = safe_pnorm(a);
    const double ub = safe_pnorm(b);
    
    // g(u) = rec(k+1, v_prefix ⊕ v_k), with v_k = Φ^{-1}(u)
    auto g = [&](double u)->double {
      double vk = safe_qnorm(u);
      std::vector<double> next = v_prefix;
      next.push_back(vk);
      return rec(k + 1, next);
    };
    
    // absolute tolerance scaled by interval length
    double abs_tol = std::max(1e-14, rel_tol * (ub - ua));
    AdaptiveSimpson integ(abs_tol, max_depth);
    return integ.integrate(g, ua, ub);
  }
};

// standardize to correlation matrix R, and transform bounds to Z space
static inline void standardize_to_R(const Eigen::MatrixXd& Sigma,
                                    const std::vector<double>& lower,
                                    const std::vector<double>& upper,
                                    const std::vector<double>& mean,
                                    Eigen::MatrixXd& R,
                                    std::vector<double>& zl,
                                    std::vector<double>& zu) {
  const int d = (int)lower.size();
  zl.resize(d);
  zu.resize(d);
  
  // Σ → R
  Eigen::VectorXd sd(d);
  for (int i = 0; i < d; ++i) {
    double v = Sigma(i,i);
    sd(i) = std::sqrt(std::max(v, 0.0));
  }
  R = Sigma;
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) {
      double denom = sd(i) * sd(j);
      if (denom > 0) R(i,j) /= denom;
      else           R(i,j)  = (i==j ? 1.0 : 0.0);
    }
    
    // transformed rectangular bounds in Z space
    for (int i = 0; i < d; ++i) {
      double s = (sd(i) > 0 ? sd(i) : 1.0);
      zl[i] = (std::isfinite(lower[i]) ? (lower[i] - mean[i]) / s : -INFINITY);
      zu[i] = (std::isfinite(upper[i]) ? (upper[i] - mean[i]) / s :  INFINITY);
      if (zl[i] > zu[i]) std::swap(zl[i], zu[i]);
    }
}

// public interface
double mvn_cdf_miwa_cpp(const std::vector<double>& lower,
                        const std::vector<double>& upper,
                        const std::vector<double>& mean,
                        const Eigen::MatrixXd& Sigma,
                        double rel_tol,
                        int    max_depth) {
  const int d = (int)lower.size();
  if ((int)upper.size() != d || (int)mean.size() != d || Sigma.rows() != d || Sigma.cols() != d)
    throw std::runtime_error("mvn_cdf_miwa_cpp: dimension mismatch.");
  if (d == 0) return 1.0;
  
  // Σ → R, and transform bounds to Z space
  Eigen::MatrixXd R = Sigma;
  std::vector<double> zl, zu;
  standardize_to_R(Sigma, lower, upper, mean, R, zl, zu);
  
  // make sure R is positive-definite by adding a small jitter
  Eigen::LLT<Eigen::MatrixXd> llt;
  double jitter = 0.0;
  for (int it = 0; it < 5; ++it) {
    llt.compute(R);
    if (llt.info() == Eigen::Success) break;
    jitter = (jitter == 0.0 ? 1e-10 : jitter * 10.0);
    R.diagonal().array() += jitter;
  }
  if (llt.info() != Eigen::Success) {
    // extreme degeneracy: fall back to independent approximation (only when R is approximately diagonal)
    double prod = 1.0;
    for (int i = 0; i < d; ++i) {
      double pb = safe_pnorm(zu[i]);
      double pa = safe_pnorm(zl[i]);
      prod *= std::max(0.0, pb - pa);
    }
    return prod;
  }
  
  // take the lower triangular L
  Eigen::MatrixXd L = llt.matrixL();
  
  // recursive integration
  MiwaRecursor mr(L, zl, zu, rel_tol, max_depth);
  std::vector<double> none;
  return mr.rec(1, none);
}
