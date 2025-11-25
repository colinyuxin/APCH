//apch_inference.cpp

// [[Rcpp::depends(Rcpp)]]
// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <unordered_map>

using namespace Rcpp;

// ---- helpers ----
void comb_rec(int start, int L,
              std::vector<int>& cur,
              std::vector< std::vector<int> >& out,
              int p){
  if(L==0){ out.push_back(cur); return; }
  for(int j=start;j<=p-L;++j){
    cur.push_back(j);
    comb_rec(j+1,L-1,cur,out,p);
    cur.pop_back();
  }
}

std::vector<int> cfg_mask_idx(const IntegerMatrix& cfg,
                              const std::vector<int>& S){
  const int n_cfg = cfg.nrow();
  std::vector<int> idx;
  for(int r=0;r<n_cfg;++r){
    bool ok = true;
    for(int col: S) if(cfg(r,col)==0){ ok=false; break; }
    if(ok) idx.push_back(r);
  }
  return idx;
}

// [[Rcpp::export]]
List level_infer_cpp(const NumericMatrix& w_mat,   // n × n_cfg
                     const IntegerMatrix& cfg,     // n_cfg × p
                     double alpha   = 0.05,        // per-layer mean-lfdr
                     double null_th = 0.9,         // "almost all zeros"
                     double ppa_min = 0.2) {       // min PPA filter
  const int n   = w_mat.nrow();
  const int p   = cfg.ncol();
  
  NumericMatrix gamma_hat(n, p);
  std::vector<char> declared(n, 0);
  
  // pre-filter null-like rows
  for (int i = 0; i < n; ++i) if (w_mat(i, 0) > null_th) declared[i] = 1;
  
  std::vector<int>                 call_eff;
  std::vector< std::vector<int> >  call_sub;
  std::vector<double>              call_ppa;
  
  for (int L = p; L >= 1; --L) {
    std::vector<int> remain;
    for (int i = 0; i < n; ++i) if (!declared[i]) remain.push_back(i);
    if (remain.empty()) break;
    
    std::vector< std::vector<int> > allS;
    std::vector<int> buf; comb_rec(0, L, buf, allS, p);
    
    struct Cand { int eff; std::vector<int> S; double ppa; };
    std::vector<Cand> cand_raw;
    
    for (const auto& Sset : allS) {
      std::vector<int> keep_cfg = cfg_mask_idx(cfg, Sset);
      for (int eff : remain) {
        double ppa = 0.0;
        for (int r : keep_cfg) ppa += w_mat(eff, r);
        if (ppa >= ppa_min) cand_raw.push_back({eff, Sset, ppa});
      }
    }
    if (cand_raw.empty()) continue;
    
    std::unordered_map<int, Cand> best;
    for (const auto& c : cand_raw) {
      auto it = best.find(c.eff);
      if (it == best.end() || c.ppa > it->second.ppa) best[c.eff] = c;
    }
    std::vector<Cand> cand; cand.reserve(best.size());
    for (auto& kv : best) cand.push_back(std::move(kv.second));
    
    std::sort(cand.begin(), cand.end(),
              [](const Cand& a, const Cand& b) {
                double la = 1.0 - a.ppa, lb = 1.0 - b.ppa;
                return (la == lb) ? (a.eff < b.eff) : (la < lb);
              });
    
    double layer_sum = 0.0; int keep_k = 0;
    for (size_t k = 0; k < cand.size(); ++k) {
      layer_sum += 1.0 - cand[k].ppa;
      double mean_lfdr = layer_sum / (k + 1);
      if (mean_lfdr <= alpha) keep_k = (int)k + 1;
      else break;
    }
    if (keep_k == 0) continue;
    
    for (int k = 0; k < keep_k; ++k) {
      const Cand& c = cand[k];
      if (declared[c.eff]) continue;
      for (int col : c.S) gamma_hat(c.eff, col) = 1.0;
      declared[c.eff] = 1;
      call_eff.push_back(c.eff + 1);
      call_sub.push_back(c.S);
      call_ppa.push_back(c.ppa);
    }
  }
  
  int ncalls = call_eff.size();
  double est_fdr = 0.0;
  if (ncalls) {
    for (double ppa : call_ppa) est_fdr += (1.0 - ppa);
    est_fdr /= ncalls;
  }
  
  List sub_list(ncalls);
  for (int i = 0; i < ncalls; ++i) {
    const auto& S = call_sub[i];
    IntegerVector iv(S.size());
    for (size_t t = 0; t < S.size(); ++t) iv[t] = S[t] + 1;
    sub_list[i] = iv;
  }
  
  return List::create(
    _["gamma_hat"]  = gamma_hat,
    _["calls_eff"]  = IntegerVector(call_eff.begin(), call_eff.end()),
    _["calls_sub"]  = sub_list,
    _["calls_PPA"]  = NumericVector(call_ppa.begin(), call_ppa.end()),
    _["est_fdr"]    = est_fdr
  );
}
