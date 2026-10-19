// apch_inference.cpp

// [[Rcpp::depends(Rcpp)]]
// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <numeric>

using namespace Rcpp;

struct SubsetMask {
  int code;
  int weight;
};

static std::vector<SubsetMask> build_nonempty_subset_masks(const int p) {
  const int M = (1 << p) - 1;
  std::vector<SubsetMask> masks;
  masks.reserve(M);
  for (int code = 1; code < (1 << p); ++code) {
    masks.push_back({code, __builtin_popcount((unsigned int)code)});
  }
  std::sort(masks.begin(), masks.end(),
            [](const SubsetMask& a, const SubsetMask& b) {
              return (a.weight == b.weight) ? (a.code > b.code) : (a.weight < b.weight);
            });
  return masks;
}

// [[Rcpp::export]]
List apch_subset_lfdr_cpp(
    const NumericMatrix& post_config_prob,
    const IntegerMatrix& config_vec_mat
) {
  const int n = post_config_prob.nrow();
  const int R = post_config_prob.ncol();
  const int p = config_vec_mat.ncol();
  const std::vector<SubsetMask> masks = build_nonempty_subset_masks(p);
  const int M = masks.size();

  IntegerMatrix subset_vec_mat(M, p);
  NumericMatrix lfdr_mat(n, M);
  IntegerVector config_code(R);
  int null_col = -1;

  for (int c = 0; c < R; ++c) {
    int code = 0;
    int sum = 0;
    for (int j = 0; j < p; ++j) {
      const int bit = config_vec_mat(c, j);
      sum += bit;
      if (bit) code |= (1 << (p - 1 - j));
    }
    config_code[c] = code;
    if (sum == 0) null_col = c;
  }

  for (int m = 0; m < M; ++m) {
    const int mask = masks[m].code;
    for (int j = 0; j < p; ++j) {
      subset_vec_mat(m, j) = (mask >> (p - 1 - j)) & 1;
    }

    for (int i = 0; i < n; ++i) {
      double ppa = 0.0;
      for (int c = 0; c < R; ++c) {
        if ((config_code[c] & mask) == mask) {
          ppa += post_config_prob(i, c);
        }
      }
      lfdr_mat(i, m) = 1.0 - ppa;
    }
  }

  NumericVector pnull(n, NA_REAL);
  if (null_col >= 0) {
    for (int i = 0; i < n; ++i) pnull[i] = post_config_prob(i, null_col);
  }

  return List::create(
    _["subset_vec_mat"] = subset_vec_mat,
    _["lfdr_mat"] = lfdr_mat,
    _["pnull"] = pnull
  );
}

// [[Rcpp::export]]
List level_infer_cpp(const NumericMatrix& lfdr_mat,          // n x n_subset
                     const IntegerMatrix& subset_vec_mat,    // n_subset x p
                     double alpha = 0.05,
                     Nullable<NumericVector> pnull = R_NilValue,
                     double null_th = 0.4,
                     double ppa_min = 0.0) {
  const int n = lfdr_mat.nrow();
  const int n_subset = lfdr_mat.ncol();
  const int p = subset_vec_mat.ncol();
  const double lfdr_hi = 0.9;
  std::vector<char> declared(n, 0);

  if (pnull.isNotNull()) {
    NumericVector p0(pnull);
    for (int i = 0; i < n; ++i) {
      if (p0[i] > null_th) declared[i] = 1;
    }
  }

  std::vector< std::vector<int> > subsets_by_level(p + 1);
  for (int s = 0; s < n_subset; ++s) {
    int L = 0;
    for (int j = 0; j < p; ++j) L += subset_vec_mat(s, j);
    if (L >= 1 && L <= p) subsets_by_level[L].push_back(s);
  }

  struct Cand {
    int eff;
    int subset_id;
    double lfdr;
  };

  std::vector<int> call_eff;
  std::vector<int> call_subset_id;
  std::vector<double> call_lfdr;
  call_eff.reserve(n);
  call_subset_id.reserve(n);
  call_lfdr.reserve(n);

  for (int L = p; L >= 1; --L) {
    const std::vector<int>& subset_ids = subsets_by_level[L];
    if (subset_ids.empty()) continue;

    std::vector<Cand> cand;
    cand.reserve(n);
    for (int i = 0; i < n; ++i) {
      if (declared[i]) continue;
      int best_subset = -1;
      double best_lfdr = std::numeric_limits<double>::infinity();
      for (int sid : subset_ids) {
        double lfdr = lfdr_mat(i, sid);
        if (lfdr < best_lfdr) {
          best_lfdr = lfdr;
          best_subset = sid;
        }
      }
      if (best_subset >= 0 && best_lfdr < lfdr_hi) {
        cand.push_back({i, best_subset, best_lfdr});
      }
    }
    if (cand.empty()) continue;

    std::sort(cand.begin(), cand.end(),
              [](const Cand& a, const Cand& b) {
                return (a.lfdr == b.lfdr) ? (a.eff < b.eff) : (a.lfdr < b.lfdr);
              });

    double layer_sum = 0.0;
    int keep_k = 0;
    for (size_t k = 0; k < cand.size(); ++k) {
      layer_sum += cand[k].lfdr;
      double mean_lfdr = layer_sum / static_cast<double>(k + 1);
      if (mean_lfdr <= alpha) keep_k = static_cast<int>(k) + 1;
      else break;
    }
    if (keep_k == 0) continue;

    for (int k = 0; k < keep_k; ++k) {
      const Cand& c = cand[k];
      if (declared[c.eff]) continue;
      declared[c.eff] = 1;
      call_eff.push_back(c.eff + 1);
      call_subset_id.push_back(c.subset_id);
      call_lfdr.push_back(c.lfdr);
    }
  }

  const int ncalls = static_cast<int>(call_eff.size());
  double est_fdr = 0.0;
  if (ncalls > 0) {
    for (double x : call_lfdr) est_fdr += x;
    est_fdr /= static_cast<double>(ncalls);
  }

  return List::create(
    _["calls_eff"] = IntegerVector(call_eff.begin(), call_eff.end()),
    _["calls_sid"] = IntegerVector(call_subset_id.begin(), call_subset_id.end()),
    _["calls_lfdr"] = NumericVector(call_lfdr.begin(), call_lfdr.end()),
    _["est_fdr"] = est_fdr
  );
}
