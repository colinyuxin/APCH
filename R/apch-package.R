#' @section Main features:
#' - Fast log-likelihood computation with parallel processing (RcppParallel)
#' - Hierarchical posterior inference via EM-style algorithms
#' - Adaptive shrinkage and correlation modeling
#' - Integration with ashr and future for distributed computation
#'
#' @section C++ integration:
#' C++ code is used for computationally intensive matrix operations,
#' using Eigen for linear algebra and RcppParallel for multi-threading.
#'
#' @docType package
#' @name apch-package
#' @useDynLib apch, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom RcppEigen fastLm
#' @importFrom RcppParallel RcppParallelLibs
#' @importFrom stats dnorm pnorm median
#' @importFrom utils tail
"_PACKAGE"
