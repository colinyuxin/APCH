#' Adaptive partial conjunction hypothesis analysis
#'
#' Runs the full APCH analysis: fits the marginal layer for each feature, fits
#' the joint layer across features, and selects for each effect the largest
#' subset of features declared jointly significant.
#'
#' \code{apch()} is \code{apch_joint()} followed by \code{apch_select()}. Use
#' the stage functions directly to reuse a marginal fit, to compute
#' subset-level lfdr or lfsr for chosen subsets, or to select at another
#' \code{alpha} without refitting.
#'
#' @param Bhat Numeric matrix of effect estimates, \eqn{n \times p}
#'   (rows = effects, columns = features). Missing entries are allowed.
#' @param S Numeric matrix of standard errors with the same dimensions as
#'   \code{Bhat}.
#' @param C Correlation of estimation errors across features: \code{NULL}
#'   (independent), a \eqn{p \times p} matrix shared by all effects, a list of
#'   \eqn{n} such matrices, or a \eqn{p \times p \times n} array.
#' @param effect_id Optional effect identifiers; defaults to the row names of
#'   \code{Bhat}, or \code{e1, e2, ...}.
#' @param feature_names Optional feature names; defaults to the column names
#'   of \code{Bhat}.
#' @param alpha Target level for the level-by-level selection.
#' @param method Selection criterion: subset-level local false sign rate
#'   (\code{"lfsr"}, recommended) or local false discovery rate
#'   (\code{"lfdr"}).
#' @param joint_mode Joint model: \code{"apch_s"} (APCH-s after
#'   allocation-state pruning, the model used in the paper),
#'   \code{"apch_s_full"} (APCH-s over the full allocation space), or
#'   \code{"apch_b"} (binary configurations with feature-wise magnitudes).
#' @param dirichlet_count Strength of the weak Dirichlet prior on the joint
#'   weights, as a fraction of \code{n}; \code{0} disables it.
#' @param workers Number of threads and parallel workers.
#' @param verbose Logical; print progress messages.
#'
#' @return An object of class \code{"apch"}: the output of
#'   \code{\link{apch_joint}} with additional elements \code{calls} (the
#'   output of \code{\link{apch_select}}), \code{alpha}, and \code{method}.
#'
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 2000
#' gamma <- matrix(rbinom(n * 2, 1, 0.1), n, 2)
#' gamma[rbinom(n, 1, 0.05) == 1, ] <- 1
#' Bhat <- gamma * matrix(rnorm(n * 2, sd = 2), n, 2) + matrix(rnorm(n * 2), n, 2)
#' fit <- apch(Bhat, matrix(1, n, 2))
#' head(fit$calls)
#' }
#' @seealso \code{\link{apch_marginal}}, \code{\link{apch_joint}},
#'   \code{\link{apch_lfdr}}, \code{\link{apch_lfsr}}, \code{\link{apch_select}}
#' @export
apch <- function(
    Bhat, S,
    C = NULL,
    effect_id = NULL,
    feature_names = NULL,
    alpha = 0.05,
    method = c("lfsr", "lfdr"),
    joint_mode = c("apch_s", "apch_s_full", "apch_b"),
    dirichlet_count = 0.001,
    workers = 1L,
    verbose = TRUE
) {
  method <- match.arg(method)
  fit <- apch_joint(
    Bhat = Bhat, S = S, C = C,
    effect_id = effect_id,
    feature_names = feature_names,
    joint_mode = match.arg(joint_mode),
    dirichlet_count = dirichlet_count,
    workers = workers,
    verbose = verbose
  )
  fit$calls <- apch_select(fit, alpha = alpha, method = method, workers = workers, verbose = verbose)
  fit$alpha <- alpha
  fit$method <- method
  fit
}

#' Marginal layer of APCH
#'
#' Fits an adaptive shrinkage prior to each feature's effects on the z-score
#' scale and distills its grid of normal components to a few representative
#' magnitudes. The result can be passed to \code{\link{apch_joint}} so that
#' several joint fits share one marginal fit.
#'
#' @inheritParams apch
#'
#' @return An object of class \code{"apch_marginal"}, a list with
#'   \code{slab_list} (standard deviations of the non-null components, per
#'   feature), \code{pi_list} (their weights, summing to 1 per feature),
#'   \code{p_null} (null proportion per feature), and the effect and feature
#'   names it was fitted on.
#' @export
apch_marginal <- function(
    Bhat, S,
    effect_id = NULL,
    feature_names = NULL,
    workers = 1L,
    verbose = TRUE
) {
  dat <- .apch_prepare_data(Bhat, S, C = NULL, effect_id = effect_id, feature_names = feature_names)
  .apch_run_marginal(dat$Bhat, dat$S, workers = workers, verbose = verbose)
}

#' Joint layer of APCH
#'
#' Fits the joint prior over null/non-null configurations and effect
#' magnitudes across features. With \code{joint_mode = "apch_s"}, allocation
#' states without support in pairwise and three-way fits are pruned before the
#' final fit.
#'
#' @inheritParams apch
#' @param marginal Optional output of \code{\link{apch_marginal}} fitted on the
#'   same \code{Bhat} and \code{S}. If \code{NULL}, the marginal layer is
#'   fitted here.
#'
#' @return An object of class \code{"apch"} with elements \code{marginal} (the
#'   marginal fit), \code{joint} (the joint fit, including per-effect posterior
#'   configuration probabilities \code{post_config_prob} and the global
#'   configuration weights \code{config_prior_df}), \code{joint_mode}, and
#'   \code{data} (the inputs on the z-score scale, used by
#'   \code{\link{apch_lfdr}}, \code{\link{apch_lfsr}}, and
#'   \code{\link{apch_select}}).
#' @export
apch_joint <- function(
    Bhat, S,
    C = NULL,
    marginal = NULL,
    effect_id = NULL,
    feature_names = NULL,
    joint_mode = c("apch_s", "apch_s_full", "apch_b"),
    dirichlet_count = 0.001,
    workers = 1L,
    verbose = TRUE
) {
  joint_mode <- match.arg(joint_mode)
  workers <- max(1L, as.integer(workers))
  dat <- .apch_prepare_data(Bhat, S, C = C, effect_id = effect_id, feature_names = feature_names)
  Bhat <- dat$Bhat
  S <- dat$S
  noise <- dat$noise
  p <- ncol(Bhat)
  if (isTRUE(verbose) && p == 1L) {
    cat("[APCH] p=1: APCH reduces to the single-feature ash posterior.\n")
  }

  if (is.null(marginal)) {
    marginal <- .apch_run_marginal(Bhat, S, workers = workers, verbose = verbose)
  } else {
    .apch_check_marginal(marginal, Bhat)
  }

  C_input <- .apch_pick_C_input(
    C_mode = noise$C_mode,
    C_shared = noise$C_shared,
    C_list = noise$C_list
  )
  config_vec <- generate_all_configs(p)
  colnames(config_vec) <- colnames(Bhat)
  use_parallel <- workers > 1L

  pairwise_alloc_support <- NULL
  if (identical(joint_mode, "apch_s")) {
    pairwise_alloc_support <- .apch_build_pairwise_alloc_support_prepared(
      Bhat = Bhat,
      S = S,
      C_mode = noise$C_mode,
      C_shared = noise$C_shared,
      C_list = noise$C_list,
      marginal_res = marginal,
      dirichlet_count = dirichlet_count,
      workers = workers,
      keep_diagnostics = FALSE,
      verbose = verbose
    )
  }

  joint_fit <- .apch_fit_joint(
    Bhat = Bhat, S = S, C = C_input,
    slab_list = marginal$slab_list,
    pi_list = marginal$pi_list,
    p_null = marginal$p_null,
    joint_mode = joint_mode,
    config_vec_mat = config_vec,
    workers = workers,
    use_parallel = use_parallel,
    verbose = verbose,
    dirichlet_count = dirichlet_count,
    pairwise_alloc_support = pairwise_alloc_support,
    marginal_loglik_cache = marginal$loglik_cache
  )

  if (identical(joint_mode, "apch_b") && is.null(joint_fit$lfsr_alloc_state)) {
    joint_fit$lfsr_alloc_state <- .apch_build_apchb_lfsr_alloc_state(
      Bhat = Bhat,
      S = S,
      C_mode = noise$C_mode,
      C_shared = noise$C_shared,
      C_list = noise$C_list,
      slab_list = marginal$slab_list,
      pi_list = marginal$pi_list,
      marginal_loglik_cache = marginal$loglik_cache,
      use_parallel = use_parallel,
      verbose = verbose
    )
  }
  marginal$loglik_cache <- NULL

  joint <- joint_fit[c(
    "post_config_prob", "config_vec_mat", "config_prior_df",
    "alloc_state_df", "alloc_cond_state", "lfsr_alloc_state"
  )]
  if (!is.null(joint_fit$pairwise_alloc_support)) {
    joint$pairwise_alloc_support <- joint_fit$pairwise_alloc_support
  }

  structure(
    list(
      marginal = marginal,
      joint = joint,
      joint_mode = joint_mode,
      data = list(Bhat = Bhat, S = S, C = C_input)
    ),
    class = "apch"
  )
}

#' Subset-level local false discovery rate
#'
#' The subset-level lfdr of effect \eqn{i} and subset \eqn{U} is the posterior
#' probability that at least one feature in \eqn{U} is null.
#'
#' @param fit Output of \code{\link{apch}} or \code{\link{apch_joint}}.
#' @param subset Subsets to evaluate: a 0/1 matrix with one row per subset and
#'   one column per feature, or integer indices into all non-empty subsets.
#'   \code{NULL} evaluates all \eqn{2^p - 1} non-empty subsets.
#'
#' @return A matrix with one row per effect and one column per subset; columns
#'   are named by the subset's 0/1 pattern. Subsets that include a missing
#'   feature of an effect are \code{NA}.
#' @export
apch_lfdr <- function(fit, subset = NULL) {
  .apch_check_fit(fit)
  bundle <- .apch_lfdr_bundle_masked(fit)
  subset_vec_mat <- .apch_parse_subset(subset, p = ncol(bundle$subset_vec_mat))
  keep_idx <- .apch_subset_match_index(
    available_subset_vec_mat = bundle$subset_vec_mat,
    target_subset_vec_mat = subset_vec_mat
  )
  out <- bundle$lfdr_mat[, keep_idx, drop = FALSE]
  rownames(out) <- rownames(fit$data$Bhat)
  colnames(out) <- apply(subset_vec_mat, 1L, paste0, collapse = "")
  out
}

#' Subset-level local false sign rate
#'
#' The subset-level lfsr of effect \eqn{i} and subset \eqn{U} is one minus the
#' largest posterior probability, over joint sign patterns, that every feature
#' in \eqn{U} is non-null with that sign pattern. It is never smaller than the
#' subset-level lfdr.
#'
#' @inheritParams apch_lfdr
#' @param workers Number of threads.
#' @param verbose Logical; print progress messages.
#'
#' @return A matrix with one row per effect and one column per subset, as in
#'   \code{\link{apch_lfdr}}.
#' @export
apch_lfsr <- function(fit, subset = NULL, workers = 1L, verbose = FALSE) {
  .apch_check_fit(fit)
  workers <- max(1L, as.integer(workers))
  out <- .apch_lfsr_matrix(
    marginal_res = fit$marginal,
    joint_res = fit$joint,
    Bhat = fit$data$Bhat,
    S = fit$data$S,
    C = fit$data$C,
    subset = subset,
    workers = workers,
    use_parallel = workers > 1L,
    verbose = verbose
  )
  subset_vec_mat <- .apch_parse_subset(subset, p = ncol(fit$data$Bhat))
  rownames(out) <- rownames(fit$data$Bhat)
  colnames(out) <- apply(subset_vec_mat, 1L, paste0, collapse = "")
  out
}

#' Level-by-level subset selection
#'
#' For each effect, searches from the largest subset size down and reports
#' the largest subset declared jointly significant. At each size, candidates
#' are accepted in increasing order of lfsr (or lfdr) while the running mean
#' stays at or below \code{alpha}; accepted effects are not considered at
#' smaller sizes.
#'
#' @inheritParams apch_lfsr
#' @param alpha Target level.
#' @param method \code{"lfsr"} (recommended) or \code{"lfdr"}.
#'
#' @return A data frame with one row per selected effect: \code{effect},
#'   subset size \code{L}, the subset's \code{lfsr} or \code{lfdr},
#'   \code{subset} (feature names joined by \code{"|"}), and \code{subset01}
#'   (the 0/1 pattern joined by \code{"|"}). Rows are ordered by decreasing
#'   \code{L}, then increasing score.
#' @export
apch_select <- function(
    fit,
    alpha = 0.05,
    method = c("lfsr", "lfdr"),
    workers = 1L,
    verbose = FALSE
) {
  .apch_check_fit(fit)
  method <- match.arg(method)
  workers <- max(1L, as.integer(workers))
  bundle <- .apch_lfdr_bundle_masked(fit)
  if (identical(method, "lfdr")) {
    res <- .apch_select_lfdr(
      lfdr_mat = bundle$lfdr_mat,
      subset_vec_mat = bundle$subset_vec_mat,
      alpha = alpha,
      pnull = bundle$pnull,
      feature_names = colnames(fit$data$Bhat),
      effect_names = rownames(fit$data$Bhat)
    )
  } else {
    res <- .apch_select_lfsr(
      fit = list(marginal_res = fit$marginal, joint_res = fit$joint),
      Bhat = fit$data$Bhat,
      S = fit$data$S,
      C = fit$data$C,
      lfdr_bundle = bundle,
      alpha = alpha,
      workers = workers,
      use_parallel = workers > 1L,
      verbose = verbose
    )
  }
  res$calls
}

#' @export
print.apch <- function(x, ...) {
  cat(sprintf(
    "APCH fit (%s): %d effects x %d features\n",
    x$joint_mode, nrow(x$data$Bhat), ncol(x$data$Bhat)
  ))
  if (!is.null(x$calls)) {
    cat(sprintf(
      "%d effects selected by %s at alpha = %g\n",
      nrow(x$calls), x$method, x$alpha
    ))
    if (nrow(x$calls) > 0L) {
      tab <- table(factor(x$calls$L, levels = rev(seq_len(ncol(x$data$Bhat)))))
      cat("Subset size L:", paste0(names(tab), ":", tab, collapse = "  "), "\n")
    }
  }
  invisible(x)
}

# Names, z-score scaling, and noise normalization shared by every stage.
.apch_prepare_data <- function(Bhat, S, C, effect_id, feature_names) {
  Bhat <- as.matrix(Bhat)
  S <- as.matrix(S)
  if (!identical(dim(Bhat), dim(S))) {
    stop("Bhat and S must have the same dimensions.")
  }
  if (is.null(effect_id) && is.null(rownames(Bhat))) effect_id <- .apch_effect_names(Bhat)
  if (is.null(feature_names) && is.null(colnames(Bhat))) feature_names <- .apch_feature_names(Bhat)
  noise <- .apch_normalize_C(Bhat, C)
  dimfilled <- .apch_fill_dimnames(
    Bhat = Bhat,
    S = S,
    R_cor = noise$C_ref,
    effect_id = effect_id,
    feature_names = feature_names
  )
  noise$C_ref <- dimfilled$R_cor
  z_scaled <- .apch_scale_to_zscore(dimfilled$Bhat, dimfilled$S)
  dimnames(z_scaled$S) <- dimnames(z_scaled$Bhat)
  list(Bhat = z_scaled$Bhat, S = z_scaled$S, noise = noise)
}

# Marginal fit on prepared (z-scale, named) data, with the future plan set for
# the duration of the fit.
.apch_run_marginal <- function(Bhat, S, workers, verbose) {
  workers <- max(1L, as.integer(workers))
  old_opts <- options(future.globals.maxSize = 20 * 1024^3)
  on.exit(options(old_opts), add = TRUE)
  old_plan <- future::plan()
  on.exit(future::plan(old_plan), add = TRUE)
  .apch_set_future_plan(workers = max(1L, min(workers, ncol(Bhat))))

  res <- .apch_fit_marginal(Bhat, S, workers = workers, verbose = verbose)
  res$effect_names <- rownames(Bhat)
  res$feature_names <- colnames(Bhat)
  class(res) <- "apch_marginal"
  res
}

.apch_check_marginal <- function(marginal, Bhat) {
  if (!inherits(marginal, "apch_marginal")) {
    stop("marginal must be the output of apch_marginal().")
  }
  if (!identical(marginal$feature_names, colnames(Bhat)) ||
      !identical(marginal$effect_names, rownames(Bhat))) {
    stop("marginal was fitted on different effects or features than Bhat.")
  }
}

.apch_check_fit <- function(fit) {
  if (!inherits(fit, "apch")) {
    stop("fit must be the output of apch() or apch_joint().")
  }
}

# Subset-level lfdr for all non-empty subsets, with NA for subsets that
# include a feature missing for that effect.
.apch_lfdr_bundle_masked <- function(fit) {
  bundle <- .apch_compute_lfdr_bundle(fit$joint)
  observed_mat <- is.finite(fit$data$Bhat) & is.finite(fit$data$S)
  subset_size <- rowSums(bundle$subset_vec_mat)
  observed_subset_count <- observed_mat %*% t(bundle$subset_vec_mat)
  bundle$lfdr_mat[
    observed_subset_count != matrix(
      subset_size,
      nrow = nrow(observed_subset_count),
      ncol = length(subset_size),
      byrow = TRUE
    )
  ] <- NA_real_
  bundle
}
