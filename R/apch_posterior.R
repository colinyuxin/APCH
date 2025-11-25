#' Compute Posterior Mean and Covariance under Gaussian Mixture Model
#'
#' The \code{apch_posterior()} function computes posterior means, variances, and
#' covariances for each effect-feature pair under the Gaussian mixture model
#' used in APCH, given configuration-level weights and feature-level priors.
#' It serves as the final step of the analysis, producing interpretable
#' posterior summaries for downstream interpretation.
#'
#' @param X Matrix of observed effects (n × p).
#' @param S Matrix of standard errors (same dimension as \code{X}).
#' @param C Observation-noise structure among features. For
#'   \code{mode = "meta"}, a single \eqn{p \times p} matrix shared across
#'   effects (typically the global correlation matrix in meta-analysis).
#'   For \code{mode = "xwas"}, a list of length \code{n} of \eqn{p \times p}
#'   matrices, giving per-effect covariance matrices \eqn{C_i}.
#' @param w_mat Posterior weight matrix (n × K).
#' @param cfg Binary configuration matrix (K × p).
#' @param grid_list List of mixture grids (one per feature).
#' @param pi_list List of mixture weights (one per feature).
#' @param mixcompdist Character; mixture component distribution. Currently only
#'   \code{"normal"} is supported for exact posterior computation; for other
#'   slab families the function returns matrices of \code{NA} with an
#'   explanatory note.
#' @param return_cov_full Logical; whether to compute full covariance matrices
#'   for each effect.
#' @param w_thresh Numeric threshold for posterior weight truncation
#'   (to improve stability).
#' @param mode Character string, either \code{"meta"} or \code{"xwas"}, matching
#'   the interpretation of \code{C}. In \code{"meta"} mode, \code{C} is a single
#'   matrix; in \code{"xwas"} mode, \code{C} must be a list of length \code{n}.
#'
#' @return A list containing:
#' \describe{
#'   \item{mean}{Matrix of posterior means (n × p).}
#'   \item{var_diag}{Matrix of marginal posterior variances.}
#'   \item{cov_full}{(Optional) 3D array of full posterior covariance matrices
#'     (p × p × n) when \code{return_cov_full = TRUE}.}
#'   \item{note}{Character string describing any numerical approximation or
#'     skipped computation. For non-Gaussian slabs, a note explains that the
#'     posterior is not implemented and \code{NA}-filled matrices are returned.}
#' }
#'
#' @examples
#' \dontrun{
#' grid_list <- apch_res$ashr_res$signed_grids
#' pi_list   <- apch_res$ashr_res$pi_list
#' w_mat <- apch_res$em_res$w_mat
#' configs <- dat$configs
#'
#' post_res <- apch_posterior(
#'   X = X,
#'   S = S,
#'   C = dat$R,
#'   w_mat = w_mat,
#'   cfg = configs,
#'   grid_list = grid_list,
#'   pi_list = pi_list,
#'   mixcompdist = "normal",
#'   return_cov_full = FALSE,
#'   w_thresh = 1e-12,
#'   mode = "meta"
#' )
#'
#' cat("Posterior mean matrix dims:", dim(post_res$mean), "\n")
#' print(head(post_res$mean))
#' }
#'
#' @export
apch_posterior <- function(
    X, S, C, w_mat, cfg, grid_list, pi_list,
    mixcompdist = "normal", return_cov_full = TRUE, w_thresh = 1e-12,
    mode = c("meta", "xwas")
) {
  mode <- match.arg(mode)
  n <- nrow(X); p <- ncol(X)
  eff_names  <- rownames(X); feat_names <- colnames(X)
  
  # Non-Gaussian slab: return NA matrices with note
  if (!identical(tolower(mixcompdist), "normal")) {
    note_msg <- paste(
      "Posterior for non-Gaussian slabs (uniform / half-uniform / half-normal)",
      "under non-diagonal covariance requires MVN numerical integration;",
      "not implemented here."
    )
    cov_full <- if (return_cov_full) {
      array(NA_real_, dim = c(p, p, n),
            dimnames = list(feat_names, feat_names, eff_names))
    } else {
      NULL
    }
    return(list(
      mean     = matrix(NA_real_, n, p,
                        dimnames = list(eff_names, feat_names)),
      cov_full = cov_full,
      var_diag = matrix(NA_real_, n, p,
                        dimnames = list(eff_names, feat_names)),
      note     = note_msg
    ))
  }
  
  # Input checks
  stopifnot(is.matrix(X), is.matrix(S), all(dim(X) == dim(S)))
  stopifnot(is.matrix(w_mat), nrow(w_mat) == n)
  if (!is.matrix(cfg)) cfg <- as.matrix(cfg)
  storage.mode(cfg) <- "integer"
  stopifnot(nrow(cfg) == ncol(w_mat), ncol(cfg) == p)
  
  if (!is.list(grid_list) || length(grid_list) != p) {
    stop("grid_list must be a list of length p (one grid per feature).")
  }
  if (!is.list(pi_list) || length(pi_list) != p) {
    stop("pi_list must be a list of length p (one weight vector per feature).")
  }
  
  if (mode == "meta") {
    # global C: p x p
    stopifnot(is.matrix(C), nrow(C) == p, ncol(C) == p)
    
    res <- gaussian_posterior_cpp(
      X = X, S = S, R_cor = C,
      w_mat = w_mat, cfg = cfg,
      grid_list = grid_list, pi_list = pi_list,
      return_cov_full = return_cov_full,
      w_thresh = w_thresh
    )
    
  } else {
    # xwas: C is list(n), each p x p
    if (!is.list(C) || length(C) != n) {
      stop("In 'xwas' mode, C must be a list of length n (one p x p matrix per effect).")
    }
    for (i in seq_len(n)) {
      Mi <- as.matrix(C[[i]])
      if (!all(dim(Mi) == c(p, p))) {
        stop("C[[", i, "]] must be a p x p matrix.")
      }
    }
    
    res <- gaussian_posterior_xwas_cpp(
      X = X, S = S, C_list = C,
      w_mat = w_mat, cfg = cfg,
      grid_list = grid_list, pi_list = pi_list,
      return_cov_full = return_cov_full,
      w_thresh = w_thresh
    )
  }
  
  # Ensure dimnames (backend often sets them; here is just a fallback)
  if (is.null(dimnames(res$mean))) {
    dimnames(res$mean) <- list(eff_names, feat_names)
  }
  if (is.null(dimnames(res$var_diag))) {
    dimnames(res$var_diag) <- list(eff_names, feat_names)
  }
  if (return_cov_full && !is.null(res$cov_full)) {
    dnm <- dimnames(res$cov_full)
    if (is.null(dnm) || length(dnm) != 3L ||
        is.null(dnm[[1]]) || is.null(dnm[[2]]) || is.null(dnm[[3]])) {
      dimnames(res$cov_full) <- list(feat_names, feat_names, eff_names)
    }
  }
  
  res
}
