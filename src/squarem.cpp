#include "squarem.h"

SquaremControl sqctl;

std::vector<double> squarem1(
  const std::vector<double>& p0,
  std::function<std::vector<double>(const std::vector<double>&)> fixptfn,
  std::function<double(const std::vector<double>&)>              objfn,
  int& iter, int& fev, int& oev, double& ll,
  bool verbose) {

    std::vector<double> p = p0, p1, p2, pn;
    const int m = p.size();
    ll  = objfn(p);
    oev = 1;
    fev = 0;
    iter = 1;
    
    while (fev < sqctl.maxiter) {
      // two fixed-point steps
      p1 = fixptfn(p); ++fev;
      double sr2 = 0;
      for (int i = 0; i < m; ++i) {
        double d = p1[i] - p[i];
        sr2 += d * d;
      }
      if (std::sqrt(sr2) < sqctl.tol) break;
      
      p2 = fixptfn(p1); ++fev;
      double sq2 = 0, sv2 = 0, srv = 0;
      for (int i = 0; i < m; ++i) {
        double d1 = p1[i] - p[i];
        double d2 = p2[i] - p1[i];
        double v  = p2[i] - 2 * p1[i] + p[i];
        sq2 += d2 * d2;
        sv2 += v * v;
        srv += v * d1;
      }
      if (std::sqrt(sq2) < sqctl.tol) break;
      
      // step length alpha (method 3)
      double alpha = std::sqrt(sr2 / sv2);
      alpha = std::max(sqctl.stepmin, std::min(sqctl.stepmax, alpha));
      
      // extrapolation
      pn.resize(m);
      for (int i = 0; i < m; ++i) {
        pn[i] = p[i] + 2 * alpha * (p1[i] - p[i]) + alpha * alpha * (p2[i] - 2 * p1[i] + p[i]);
      }
      
      // stabilization
      if (std::fabs(alpha - 1.0) > 1e-2) {
        pn = fixptfn(pn); ++fev;
      }
      
      // optional monotonicity
      if (std::isfinite(sqctl.objfninc)) {
        double ll_new = objfn(pn); ++oev;
        if (ll_new > ll + sqctl.objfninc) {
          pn = p2;
          ll_new = objfn(pn); ++oev;
        }
        ll = ll_new;
      }
      
      // update
      p.swap(pn);
      if (alpha == sqctl.stepmax) sqctl.stepmax *= sqctl.mstep;
      
      // always refresh ll for trace
      ll = objfn(p); ++oev;
      if (verbose && sqctl.trace && iter % 50 == 0) {
        Rcout << "squarem iter " << iter
              << " ll=" << ll
              << " alpha=" << alpha << "\n";
      }
      ++iter;
    }
    return p;
}

