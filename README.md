# APCH: Adaptive Partial Conjunction Hypothesis R package

APCH is an R package implementing the **Adaptive Partial Conjunction Hypothesis (APCH)** framework for identifying jointly non-null subsets of traits, tissues, or other features using only summary statistics. APCH is designed for pleiotropy and multi-feature association analysis, for example:

- cross-trait GWAS meta-analysis,
- multi-tissue / multi-omic association studies,
- other settings with many “effect units” (SNPs, genes, loci) measured across multiple features (traits, tissues, omics).

For each effect unit, APCH learns a flexible empirical-Bayes model for the effect size distribution, accounts for correlation in estimation errors across features, and then searches over feature subsets to find  one maximally informative jointly non-null subset while controlling the false discovery rate (FDR)**.

---

## Installation

```r
# install.packages("remotes")  # if not installed
remotes::install_github("colinyuxin/APCH")

library(APCH)
