#pragma once

// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <functional>
#include <limits>
#include <algorithm>
#include <cmath>

using namespace Rcpp;

// control parameters
struct SquaremControl {
  int    maxiter  = 5000;
  double stepmin  = 0.0;
  double stepmax  = 4.0;
  double mstep    = 4.0;
  double tol      = 1e-7;
  double objfninc = std::numeric_limits<double>::infinity();
  bool   trace    = true;
};

// global control object
extern SquaremControl sqctl;

// squarem
std::vector<double> squarem1(
  const std::vector<double>& p0,
  std::function<std::vector<double>(const std::vector<double>&)> fixptfn,
  std::function<double(const std::vector<double>&)>              objfn,
  int& iter, int& fev, int& oev, double& ll, bool verbose = true);
