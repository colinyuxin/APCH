#' Fit Adaptive Shrinkage Models for All Features
#'
#' The \code{fit_all_feature()} function performs independent adaptive shrinkage
#' (via \code{ashr}) across multiple features, possibly in parallel. It fits
#' mixture distributions to the observed data and optionally performs grid
#' distillation to simplify posterior grids.
#'
#' @param X A numeric matrix of observed effects (n × p).
#' @param S A numeric matrix of standard errors (same dimensions as \code{X}).
#' @param mixcompdist Character; mixture component distribution
#'   (\code{"normal"} by default).
#' @param lambda0 Regularization parameter for mixture prior.
#' @param fine_s_min_mult,fine_s_max_mult Numeric multipliers controlling the
#'   range of scale parameters for fine grids.
#' @param fine_mult_distill,fine_mult_nodistill Numeric multipliers controlling
#'   the fine-grid step size for features selected for distillation and for
#'   non-distilled features, respectively. These are passed as \code{fine_mult}
#'   to \code{distill_grid_path_from_xs()}. Default: \code{1.35} and
#'   \code{sqrt(2)}.
#' @param fine_max_len Integer; maximum grid length.
#' @param mixsqp_control Control list for \code{mixsqp} optimization.
#' @param mll_drop_total_tol,ks_increase_total_tol Convergence tolerance
#'   parameters for grid refinement.
#' @param min_grid_len,max_merges Integer limits controlling grid merging
#'   behavior.
#' @param pair_frac Fraction of grid pairs allowed to merge at each step.
#' @param rel_thresh Relative tolerance for grid merging.
#' @param distill Logical or \code{"auto"}; enables grid simplification.
#' @param R_cor Optional correlation matrix among features.
#' @param cor_thresh Correlation threshold for distillation selection.
#' @param workers Number of parallel workers.
#' @param plan Parallelization plan (\code{"multicore"} or \code{"multisession"}).
#' @param future_seed Logical; whether to use reproducible random seeds in futures.
#' @param future_scheduling Chunk scheduling ratio for parallel tasks.
#' @param output Either \code{"light"} or \code{"full"}, controlling result
#'   verbosity.
#' @param verbose Logical; print diagnostic messages.
#'
#' @return A list containing:
#' \describe{
#'   \item{signed_grids}{List of signed mixture grids for each feature.}
#'   \item{pi_list}{List of mixture weights for each feature.}
#'   \item{fit_res}{(If \code{output="full"}) Full fit results for each feature.}
#' }
#'
#' @export
fit_all_feature <- function(
    X, S,
    mixcompdist = "normal", lambda0 = 10,
    fine_s_min_mult = 0.5, fine_s_max_mult = 1.0,
    fine_mult_distill = 1.35, fine_mult_nodistill = sqrt(2),
    fine_max_len = 500, mixsqp_control = NULL,
    mll_drop_total_tol = 2e-4, ks_increase_total_tol = 1e-2,
    min_grid_len = 3, max_merges = Inf, pair_frac = 0.20,
    rel_thresh = 1e-6, distill = "auto",
    R_cor = NULL, cor_thresh = sqrt(.Machine$double.eps),
    workers = max(1L, parallel::detectCores(logical = TRUE) - 1L),
    plan = NULL, future_seed = TRUE, future_scheduling = 1.0,
    output = c("light", "full"), verbose = TRUE
) {
  stopifnot(is.matrix(X), is.matrix(S), all(dim(X) == dim(S)))
  output <- match.arg(output)
  p <- ncol(X)
  
  .set_single_thread_blas()
  workers <- max(1L, min(as.integer(workers), p))
  
  if (is.null(plan)) {
    plan <- if (.Platform$OS.type == "windows") "multisession" else "multicore"
  }
  if (identical(plan, "multisession")) {
    future::plan(future::multisession, workers = workers)
  } else if (identical(plan, "multicore")) {
    future::plan(future::multicore, workers = workers)
  } else {
    stop("Unsupported plan: use 'multisession' or 'multicore'.")
  }
  
  if (verbose) {
    cat(sprintf(
      "[Parallel] features=%d | workers=%d | plan=%s | future.seed=%s | scheduling=%s\n",
      p, workers, plan, deparse(future_seed), as.character(future_scheduling)
    ))
  }
  
  fnm <- colnames(X); if (is.null(fnm)) fnm <- paste0("feat", seq_len(p))
  if (is.null(R_cor)) R_cor <- diag(p)
  
  distill_sel <- .resolve_distill(
    distill = distill, R_cor = R_cor,
    feature_names = fnm, cor_thresh = cor_thresh
  )
  fine_mult_vec <- ifelse(distill_sel, fine_mult_distill, fine_mult_nodistill)
  
  tasks <- lapply(seq_len(p), function(j) {
    list(
      x = X[, j], s = S[, j], name = fnm[j],
      do_distill = isTRUE(distill_sel[j]), fine_mult = fine_mult_vec[j]
    )
  })
  
  # Optional dependency: RhpcBLASctl (if available)
  future_pkgs <- c("apch", "ashr", "sn")
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    future_pkgs <- c(future_pkgs, "RhpcBLASctl")
  }
  
  res_list <- future.apply::future_lapply(
    X = tasks,
    FUN = function(task) {
      if (exists(".set_single_thread_blas", mode = "function")) {
        .set_single_thread_blas()
      }
      fit <- distill_grid_path_from_xs(
        x = task$x, s = task$s,
        mixcompdist = mixcompdist, lambda0 = lambda0,
        fine_mult = task$fine_mult,
        fine_s_min_mult = fine_s_min_mult, fine_s_max_mult = fine_s_max_mult,
        fine_max_len = fine_max_len, mixsqp_control = mixsqp_control,
        mll_drop_total_tol = mll_drop_total_tol,
        ks_increase_total_tol = ks_increase_total_tol,
        min_grid_len = min_grid_len, max_merges = max_merges,
        pair_frac = pair_frac,
        rel_thresh = rel_thresh, verbose = verbose,
        distill = task$do_distill
      )
      last_step <- as.character(max(fit$table$step))
      list(
        name        = task$name,
        fit         = fit,
        signed_grid = fit$steps[[last_step]]$signed_grid,
        pi          = fit$steps[[last_step]]$pi
      )
    },
    future.seed       = future_seed,
    future.packages   = future_pkgs,
    future.scheduling = future_scheduling
  )
  
  nm <- vapply(res_list, function(x) x$name, character(1))
  if (any(nm == "")) nm <- paste0("feat", seq_len(p))
  
  signed_grids <- vector("list", p)
  pi_list <- vector("list", p)
  for (j in seq_len(p)) {
    signed_grids[[j]] <- res_list[[j]]$signed_grid
    pi_list[[j]]      <- res_list[[j]]$pi
  }
  names(signed_grids) <- nm
  names(pi_list)      <- nm
  
  if (output == "light") {
    list(signed_grids = signed_grids, pi_list = pi_list)
  } else {
    fit_res <- vector("list", p)
    for (j in seq_len(p)) fit_res[[j]] <- res_list[[j]]$fit
    names(fit_res) <- nm
    list(fit_res = fit_res, signed_grids = signed_grids, pi_list = pi_list)
  }
}
