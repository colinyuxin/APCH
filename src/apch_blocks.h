// apch_blocks.h
#pragma once

#include <Rcpp.h>
#include <vector>
#include <cmath>

static inline std::vector< std::vector<int> > apch_blocks_from_numeric_matrix(
    const Rcpp::NumericMatrix& E,
    const double tol
) {
  const int p = E.nrow();
  std::vector<int> comp(p, -1);
  std::vector< std::vector<int> > blocks;
  int cid = 0;
  for (int v = 0; v < p; ++v) {
    if (comp[v] >= 0) continue;
    std::vector<int> stack(1, v);
    comp[v] = cid;
    std::vector<int> block;
    while (!stack.empty()) {
      const int u = stack.back();
      stack.pop_back();
      block.push_back(u);
      for (int w = 0; w < p; ++w) {
        if (w == u || comp[w] >= 0) continue;
        if (std::abs(E(u, w)) >= tol) {
          comp[w] = cid;
          stack.push_back(w);
        }
      }
    }
    blocks.push_back(block);
    ++cid;
  }
  return blocks;
}

static inline std::vector< std::vector<int> > apch_blocks_from_numeric_matrix_subset(
    const Rcpp::NumericMatrix& E,
    const std::vector<int>& idx,
    const double tol
) {
  const int p = static_cast<int>(idx.size());
  std::vector<int> comp(p, -1);
  std::vector< std::vector<int> > blocks;
  int cid = 0;
  for (int v = 0; v < p; ++v) {
    if (comp[v] >= 0) continue;
    std::vector<int> stack(1, v);
    comp[v] = cid;
    std::vector<int> block;
    while (!stack.empty()) {
      const int u = stack.back();
      stack.pop_back();
      block.push_back(u);
      const int ju = idx[u];
      for (int w = 0; w < p; ++w) {
        if (w == u || comp[w] >= 0) continue;
        const int jw = idx[w];
        if (std::abs(E(ju, jw)) >= tol) {
          comp[w] = cid;
          stack.push_back(w);
        }
      }
    }
    blocks.push_back(block);
    ++cid;
  }
  return blocks;
}
