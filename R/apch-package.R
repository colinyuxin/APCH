#' @section Main features:
#' - Subset-level local false discovery rates and local false sign rates
#' - Adaptive shrinkage priors for each feature's effect sizes
#' - Configuration-specific sharing of effect sizes across features
#' - Correlated estimation errors across features
#'
#' @section C++ integration:
#' C++ code is used for computationally intensive likelihood evaluations,
#' with RcppParallel for multi-threading.
#'
#' @keywords internal
#' @name apch-package
#' @useDynLib apch, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom RcppParallel RcppParallelLibs
#' @importFrom stats dnorm pnorm median setNames
#' @importFrom utils tail modifyList
#' @importFrom mvtnorm pmvnorm
#' @importFrom pbivnorm pbivnorm
"_PACKAGE"
