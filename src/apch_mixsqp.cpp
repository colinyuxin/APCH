// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

namespace {

inline void apch_mixem_update(const mat& L, const vec& w, vec& x,
                              vec& xnew, vec& scratch) {
  const double e = 1e-15;
  const uword n = L.n_rows;
  const uword m = L.n_cols;
  xnew.zeros();

  for (uword i = 0; i < n; ++i) {
    double row_max = 0.0;
    for (uword j = 0; j < m; ++j) {
      const double v = L(i, j) * (x[j] + e);
      scratch[j] = v;
      if (v > row_max) {
        row_max = v;
      }
    }

    if (!(row_max > 0) || !std::isfinite(row_max)) {
      xnew += (w[i] / static_cast<double>(m));
      continue;
    }

    double denom = 0.0;
    const double inv_row_max = 1.0 / row_max;
    for (uword j = 0; j < m; ++j) {
      denom += scratch[j] * inv_row_max + e;
    }

    const double scale = w[i] / denom;
    for (uword j = 0; j < m; ++j) {
      xnew[j] += scale * (scratch[j] * inv_row_max + e);
    }
  }

  x = xnew;
}

inline double apch_compute_objective(const mat& L, const vec& w, const vec& x, double eps) {
  vec u = L * x;
  u += eps;
  if (u.min() <= 0) {
    stop("Objective is -Inf");
  }
  return -sum(w % log(u));
}

inline void apch_compute_objective_grad(const mat& L, const vec& w, const vec& x,
                                        double eps, double& obj, vec& g, mat& H,
                                        mat& Z) {
  vec u = L * x;
  u += eps;
  if (u.min() <= 0) {
    stop("Objective is -Inf");
  }
  obj = -sum(w % log(u));
  g = -trans(L) * (w / u);
  Z = L.each_col() % (sqrt(w) / u);
  H = trans(Z) * Z;
}

inline void apch_feasible_stepsize(const vec& x, const vec& p, int& j, double& a) {
  const uvec i = find(p < 0);
  a = 1.0;
  j = -1;
  if (!i.is_empty()) {
    const vec t = -x(i) / p(i);
    j = static_cast<int>(t.index_min());
    if (t[j] < 1.0) {
      a = t[j];
    }
    j = static_cast<int>(i[j]);
  }
}

inline double apch_init_hessian_correction(const mat& H, double a0) {
  const double d = H.diag().min();
  return (d > a0) ? 0.0 : (a0 - d);
}

inline void apch_compute_activeset_searchdir(const mat& H, const vec& y, vec& p,
                                             mat& B, double ainc) {
  const double a0 = 1e-15;
  const double amax = 1e15;
  const int n = y.n_elem;
  const mat I(n, n, fill::eye);
  mat R(n, n);

  double a = apch_init_hessian_correction(H, a0);
  while (true) {
    B = H + a * I;
    if (a * ainc > amax) {
      break;
    } else if (chol(R, B)) {
      break;
    } else if (a <= 0) {
      a = a0;
    } else {
      a *= ainc;
    }
  }

  p = solve(B, -y);
}

inline int apch_activesetqp(const mat& H, const vec& g, vec& y, int maxiter,
                            double zerosearchdir, double tol, double ainc) {
  const int m = g.n_elem;
  double a;
  int k;
  bool add_to_working_set;
  vec b(m);
  vec p(m);
  vec bs(m);
  vec ps(m);
  mat Hs(m, m);
  mat Bs(m, m);
  uvec i(m);
  uvec j(m);
  uvec t = (y > 0);

  int iter = 0;
  for (; iter < maxiter; ++iter) {
    i = find(t != 0);
    j = find(t == 0);
    y(j).fill(0);

    Hs = H(i, i);
    b = g;
    b(i) += Hs * y(i);
    bs = b(i);

    p.fill(0);
    apch_compute_activeset_searchdir(Hs, bs, ps, Bs, ainc);
    p(i) = ps;

    a = 1.0;
    add_to_working_set = false;

    if (norm(p, "inf") <= zerosearchdir) {
      b = g + H * y;
      if (j.is_empty()) {
        ++iter;
        break;
      } else if (b(j).min() >= -tol) {
        ++iter;
        break;
      } else {
        k = static_cast<int>(j(b(j).index_min()));
        t(k) = 1;
      }
    } else {
      apch_feasible_stepsize(y, p, k, a);
      if (k >= 0 && a < 1.0) {
        if (i.n_elem > 1) {
          add_to_working_set = true;
        }
      }

      y += a * p;
      j = find(y < 0);
      y(j).fill(0);
      y(j).fill(0);
      if (add_to_working_set) {
        t(k) = 0;
        y(k) = 0;
      }
    }
  }

  return iter;
}

inline int apch_backtracking_line_search(double f, const mat& L, const vec& w,
                                         const vec& g, const vec& x, const vec& y,
                                         double eps, double suffdecr, double beta,
                                         double amin, double& a, vec& xnew) {
  const vec p = y - x;
  int k = -1;
  double afeas = 1.0;
  apch_feasible_stepsize(x, p, k, afeas);
  int nls = 0;

  if (afeas <= amin) {
    a = afeas;
    xnew = afeas * y + (1 - afeas) * x;
  } else {
    a = std::min(1.0, afeas);
    while (true) {
      xnew = a * y + (1 - a) * x;
      const double fnew = apch_compute_objective(L, w, xnew, eps);
      ++nls;
      if ((xnew.min() >= 0) &&
          (fnew + sum(xnew) <= f + sum(x) + suffdecr * a * dot(y - x, g + 1))) {
        break;
      } else if (a * beta < amin) {
        a = amin;
        xnew = a * y + (1 - a) * x;
        break;
      }
      a *= beta;
    }
    xnew = a * y + (1 - a) * x;
  }

  return nls;
}

} // namespace

// [[Rcpp::export]]
NumericVector apch_mixsqp_dense_cpp(const arma::mat& L, const arma::vec& w,
                                    const arma::vec& x0, double eps,
                                    int numiter_em, double convtol_sqp,
                                    double convtol_activeset,
                                    double zero_threshold_solution,
                                    double zero_threshold_searchdir,
                                    double suffdecr, double stepsize_reduce,
                                    double min_step_size,
                                    double identity_contrib_increase,
                                    int maxiter_sqp, int maxiter_activeset,
                                    bool verbose) {
  const int m = L.n_cols;
  vec x = x0;
  vec xem(m);
  vec scratch(m);
  for (int i = 0; i < numiter_em; ++i) {
    apch_mixem_update(L, w, x, xem, scratch);
    if (verbose) {
      const double obj = apch_compute_objective(L, w, x, eps);
      const double dmax = 0.0;
      const int nnz = static_cast<int>(sum(x > zero_threshold_solution));
      Rprintf("%4d %+0.9e  -- EM -- %4d 1.00e+00 %0.2e  --  --\n",
              i + 1, obj, nnz, dmax);
    }
  }

  vec g(m);
  vec ghat(m);
  vec y(m);
  vec xnew(m);
  mat H(m, m);
  mat Z = L;
  uvec j0(m);
  uvec j1(m);

  for (int i = 0; i < maxiter_sqp; ++i) {
    apch_mixem_update(L, w, x, xem, scratch);

    j0 = find(x <= zero_threshold_solution);
    j1 = find(x > zero_threshold_solution);
    x(j0).fill(0);

    double obj = 0.0;
    apch_compute_objective_grad(L, w, x, eps, obj, g, H, Z);
    const double gmin = 1 + g(j1).min();

    if (verbose) {
      if (i == 0) {
        Rprintf("%4d %+0.9e %+0.3e%4d  ------   ------   --  --\n",
                i + 1, obj, -gmin, static_cast<int>(sum(x > 0)));
      }
    }

    if (gmin >= -convtol_sqp) {
      break;
    }

    checkUserInterrupt();

    ghat = g - H * x + 1;
    y = x;
    apch_activesetqp(H, ghat, y, maxiter_activeset, zero_threshold_searchdir,
                     convtol_activeset, identity_contrib_increase);

    double step = 1.0;
    apch_backtracking_line_search(obj, L, w, g, x, y, eps, suffdecr,
                                  stepsize_reduce, min_step_size, step, xnew);

    x = xnew;
  }

  return wrap(x);
}
