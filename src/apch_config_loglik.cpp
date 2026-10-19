// apch_config_loglik.cpp
#include <Rcpp.h>
#include <RcppParallel.h>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace Rcpp;
using namespace RcppParallel;

struct BlockCfgLoglikWorker : public Worker {
  const RMatrix<double> log_alloc;
  const RMatrix<double> alloc_logw_cond;
  const std::vector< std::vector<int> >& ids_by_code;
  RMatrix<double> out;
  const int Rm;

  BlockCfgLoglikWorker(const NumericMatrix& log_alloc_mat,
                       const NumericMatrix& alloc_logw_cond_mat,
                       const std::vector< std::vector<int> >& ids_by_code_,
                       NumericMatrix& out_mat)
    : log_alloc(log_alloc_mat),
      alloc_logw_cond(alloc_logw_cond_mat),
      ids_by_code(ids_by_code_),
      out(out_mat),
      Rm(alloc_logw_cond_mat.ncol()) {}

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t ii = begin; ii < end; ++ii) {
      const int i = static_cast<int>(ii);
      for (int c = 0; c < Rm; ++c) {
        const auto& ids = ids_by_code[c];
        if (ids.empty()) {
          out(i, c) = R_NegInf;
          continue;
        }

        double maxv = R_NegInf;
        double sumexp = 0.0;
        for (int id : ids) {
          const double lw = alloc_logw_cond(id, c);
          const double val = log_alloc(i, id) + lw;
          if (!std::isfinite(val)) continue;
          if (val > maxv) {
            sumexp = std::isfinite(maxv) ? (sumexp * std::exp(maxv - val) + 1.0) : 1.0;
            maxv = val;
          } else {
            sumexp += std::exp(val - maxv);
          }
        }
        out(i, c) = std::isfinite(maxv) ? (maxv + std::log(sumexp)) : R_NegInf;
      }
    }
  }
};

// [[Rcpp::export]]
NumericMatrix apch_config_loglik_cpp(
    const NumericMatrix& log_alloc_mat,
    const IntegerVector& alloc_code,
    const NumericMatrix& alloc_logw_cond,
    bool use_parallel = true,
    int grain_size = 512
) {
  const int n = log_alloc_mat.nrow();
  const int L = log_alloc_mat.ncol();
  const int Rm = alloc_logw_cond.ncol();

  std::vector< std::vector<int> > ids_by_code(Rm);
  for (int l = 0; l < L; ++l) {
    const int c = alloc_code[l];
    if (c >= 0 && c < Rm) ids_by_code[c].push_back(l);
  }

  NumericMatrix out(n, Rm);
  BlockCfgLoglikWorker worker(log_alloc_mat, alloc_logw_cond, ids_by_code, out);
  if (n > 1 && use_parallel) {
    parallelFor(0, (std::size_t)n, worker, (std::size_t)std::max(grain_size, 1));
  } else {
    worker(0, (std::size_t)n);
  }
  return out;
}
