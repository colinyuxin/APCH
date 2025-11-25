#' Level-by-Level Inference for Partial Conjunction Hypotheses
#'
#' The \code{level_by_level_inference()} function performs multi-level inference
#' on posterior mixture weights obtained from the APCH model. It determines which
#' effects (rows) are active across subsets of features at varying conjunction
#' levels.
#'
#' @param w_mat Matrix of normalized posterior weights (n × K), where each row
#'   sums to 1.
#' @param configs Binary configuration matrix (K × p), representing all possible
#'   feature activation patterns.
#' @param alpha Significance level for false discovery rate control.
#' @param null_th Posterior probability threshold for null acceptance.
#' @param ppa_min Minimum posterior probability required to report a call.
#' @param feature_names Optional vector of feature names.
#'
#' @return A list containing:
#' \describe{
#'   \item{calls}{Data frame of detected conjunctions, including effect name,
#'     level \code{L}, local FDR, and subset indicators.}
#'   \item{est_fdr}{Estimated overall false discovery rate.}
#' }
#'
#' @examples
#' \dontrun{
#' lvl_res <- level_by_level_inference(
#'   w_mat = w_mat,
#'   configs = configs,
#'   alpha = 0.05,
#'   null_th = 0.9,
#'   ppa_min = 0.5,
#'   feature_names = paste0("F", 1:p)
#' )
#'
#' cat("Calls returned:", nrow(lvl_res$calls), "\n")
#' head(lvl_res$calls)
#' }
#'
#' @export
level_by_level_inference <- function(
    w_mat, configs,
    alpha = 0.05, null_th = 0.9, ppa_min = 0.5,
    feature_names = NULL
) {
  # C++ inference
  res <- level_infer_cpp(w_mat, configs, alpha, null_th, ppa_min)
  
  # feature names
  p <- ncol(configs)
  if (is.null(feature_names)) {
    feature_names <- colnames(configs)
    if (is.null(feature_names) || length(feature_names) != p) {
      feature_names <- paste0("F", seq_len(p))
    }
  }
  
  # effect names (derived from the row names of w_mat; if missing, supplement)
  eff_names <- rownames(w_mat)
  if (is.null(eff_names) || length(eff_names) != nrow(w_mat)) {
    eff_names <- paste0("e", seq_len(nrow(w_mat)))
  }
  
  ncalls <- length(res$calls_eff)
  
  if (ncalls > 0) {
    # calls_eff is 1-based index mapped to names.
    L_vec     <- vapply(res$calls_sub, length, integer(1))
    lfdr_vec  <- 1 - res$calls_PPA
    subset_ix <- lapply(res$calls_sub, function(ix) as.integer(ix))
    subset_nm <- lapply(subset_ix, function(ix) feature_names[ix])
    
    calls <- data.frame(
      effect = eff_names[as.integer(res$calls_eff)],
      L      = as.integer(L_vec),
      lfdr   = as.numeric(lfdr_vec),
      stringsAsFactors = FALSE
    )
    
    calls$subset <- vapply(
      subset_nm,
      function(x) paste(x, collapse = "|"),
      character(1)
    )
    
    calls$subset01 <- vapply(
      subset_nm,
      function(x) paste(as.integer(feature_names %in% x), collapse = ""),
      character(1)
    )
    
    # sort by L, lfdr, effect
    ord <- order(-calls$L, calls$lfdr, calls$effect)
    calls <- calls[ord, , drop = FALSE]
    rownames(calls) <- NULL
  } else {
    calls <- data.frame(
      effect   = character(0),
      L        = integer(0),
      lfdr     = numeric(0),
      subset   = character(0),
      subset01 = character(0),
      stringsAsFactors = FALSE
    )
  }
  
  list(
    calls   = calls,
    est_fdr = res$est_fdr
  )
}
