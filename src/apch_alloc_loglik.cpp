// apch_alloc_loglik.cpp
#include "calc_density.h"

// [[Rcpp::depends(RcppEigen,RcppParallel)]]
// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cmath>
#include <vector>

using namespace Rcpp;
using namespace RcppParallel;

static std::vector<std::vector<double>> build_grids(const List &sgL) {
  const int p = sgL.size();
  std::vector<std::vector<double>> grids(p);
  for (int j = 0; j < p; ++j) {
    NumericVector g = sgL[j];
    grids[j].assign(g.begin(), g.end());
  }
  return grids;
}

// [[Rcpp::export]]
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> apch_alloc_loglik_cpp(
    const NumericMatrix &Bhat,
    const NumericMatrix &S,
    const List &C_list,
    const List &sgL,
    const IntegerMatrix &alloc_mat,
    bool use_parallel = true
) {
  return logmat_normal_list(Bhat, S, C_list, alloc_mat, build_grids(sgL), use_parallel);
}

// [[Rcpp::export]]
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> apch_alloc_loglik_shared_cpp(
    const NumericMatrix &Bhat,
    const NumericMatrix &S,
    const NumericMatrix &C_shared,
    const List &sgL,
    const IntegerMatrix &alloc_mat,
    bool use_parallel = true
) {
  return logmat_normal(Bhat, S, C_shared, alloc_mat, build_grids(sgL), use_parallel);
}
