# joint_mode_interface.R — configuration layer
# Provides:
#   - .apch_fit_joint()

# ------------------------------------------------------------------
# Internal helpers for the new configuration likelihood path
# ------------------------------------------------------------------

.apch_alloc_core <- function(Kj) {
  apch_alloc_core_cpp(as.integer(Kj))
}

.apch_noise_components <- function(C_mode, C_shared, C_list, p) {
  A <- .apch_noise_adj_matrix(C_mode, C_shared, C_list, p)
  if (p <= 1L) {
    return(list(1L))
  }
  g <- igraph::graph_from_adjacency_matrix(A, mode = "undirected", diag = FALSE)
  comp <- igraph::components(g)$membership
  blocks_out <- split(seq_len(p), comp)
  blocks_out[order(vapply(blocks_out, function(x) min(x), integer(1)))]
}

.apch_config_code <- function(config_vec_mat) {
  apply(config_vec_mat, 1L, function(v) {
    s <- 0L
    for (b in v) {
      s <- bitwOr(bitwShiftL(s, 1L), as.integer(b > 0L))
    }
    s
  })
}

.apch_prior_p0_vec <- function(p_null) {
  as.numeric(unlist(p_null, use.names = FALSE))
}

.apch_config_prior_center_from_marginals <- function(config_vec_mat, p_null) {
  p0 <- .apch_prior_p0_vec(p_null)
  q <- apply(config_vec_mat, 1L, function(r) {
    prod(ifelse(r > 0, 1 - p0, p0))
  })
  q <- as.numeric(q)
  q / sum(q)
}

.apch_alloc_prior_center_from_marginals <- function(alloc_mat, p_null, pi_list) {
  p0 <- .apch_prior_p0_vec(p_null)
  pi_num <- lapply(pi_list, as.numeric)
  # Important: q is normalized over the current alloc_mat only.
  # After pairwise pruning, alloc_mat is already the surviving admissible support,
  # so this conditions the allocation-level prior center on that support.
  q <- vapply(seq_len(nrow(alloc_mat)), function(l) {
    out <- 1
    for (j in seq_len(ncol(alloc_mat))) {
      aj <- alloc_mat[l, j]
      if (aj <= 0L) {
        out <- out * p0[j]
      } else {
        out <- out * (1 - p0[j]) * pi_num[[j]][aj]
      }
    }
    out
  }, numeric(1))
  q / sum(q)
}

# Validate and normalize slab-based marginal inputs.
# Fit prior weights from log-likelihood matrix using mixsqp.
.apch_fit_config_prior <- function(
    log_prob_mat,
    x0 = NULL,
    prior_q = NULL,
    prior_frac = 0.001,
    prior_alpha_min = 0,
    prior_alpha_cap = Inf,
    prior_alpha_cap_null = Inf,
    null_col = NULL,
    keep_post_prob = TRUE,
    keep_post_colsum = FALSE,
    blas_threads = 1L
) {
  storage.mode(log_prob_mat) <- "double"
  n <- nrow(log_prob_mat)
  R <- ncol(log_prob_mat)
  
  if (R == 1L) {
    post_prob <- if (isTRUE(keep_post_prob)) matrix(1, nrow = n, ncol = 1L) else NULL
    post_colsum <- if (isTRUE(keep_post_colsum)) rep(n, 1L) else NULL
    ll <- mixloglik_from_logL_cpp(log_prob_mat, row_max_cpp(log_prob_mat), 1)
    return(list(config_prior = 1, post_config_prob = post_prob, post_colsum = post_colsum, ll = ll))
  }
 
  prior_mixsqp <- rep(1, R)

  if (!is.null(prior_q) && prior_frac > 0) {
    q <- as.numeric(prior_q)
    q <- q / sum(q)
    kappa <- prior_frac * n
    alpha <- kappa * q
    if (prior_alpha_min > 0) {
      alpha <- pmax(alpha, prior_alpha_min)
    }
    null_col <- if (is.null(null_col)) {
      integer(0)
    } else {
      unique(as.integer(null_col))
    }
    null_col <- null_col[null_col >= 1L & null_col <= length(alpha)]
    nonnull_col <- if (length(null_col) > 0L) setdiff(seq_along(alpha), null_col) else seq_along(alpha)
    if (prior_alpha_cap > 0 && length(nonnull_col) > 0L) {
      alpha[nonnull_col] <- pmin(alpha[nonnull_col], prior_alpha_cap)
    }
    if (prior_alpha_cap_null > 0 && length(null_col) > 0L) {
      alpha[null_col] <- pmin(alpha[null_col], prior_alpha_cap_null)
    }
    prior_mixsqp <- prior_mixsqp + alpha
  }
  
  if (is.null(x0)) {
    x0 <- rep(1 / R, R)
  } else {
    x0 <- as.numeric(x0)
    x0 <- x0 / sum(x0)
  }

  mix_inputs <- build_mixsqp_inputs_cpp(log_prob_mat, prior_mixsqp)
  rm(prior_mixsqp)
  
  fit <- .apch_with_blas_threads(
    blas_threads,
    .apch_mixsqp(
      L = mix_inputs$A,
      w = mix_inputs$w,
      x0 = x0,
      control = list(
        verbose = FALSE,
        zero.threshold.solution = 0
      )
    )
  )
  
  prior <- pmax(as.numeric(fit$x), 1e-15)
  prior <- prior / sum(prior)
  row_max <- mix_inputs$row_max
  rm(fit, mix_inputs)
  gc(verbose = FALSE)

  if (isTRUE(keep_post_prob)) {
    norm_res <- posterior_from_logL_cpp(log_prob_mat, prior)
    post_prob <- norm_res$post_prob
    ll <- norm_res$ll
    rm(norm_res)
    return(list(config_prior = prior, post_config_prob = post_prob, post_colsum = NULL, ll = ll))
  }

  if (isTRUE(keep_post_colsum)) {
    norm_res <- posterior_colsum_from_logL_cpp(log_prob_mat, prior)
    post_colsum <- norm_res$post_colsum
    ll <- norm_res$ll
    rm(norm_res)
    return(list(config_prior = prior, post_config_prob = NULL, post_colsum = post_colsum, ll = ll))
  }

  ll <- mixloglik_from_logL_cpp(log_prob_mat, row_max, prior)
  list(config_prior = prior, post_config_prob = NULL, post_colsum = NULL, ll = ll)
}

.apch_config_loglik <- function(log_alloc_mat, alloc_code, alloc_logw_cond, use_parallel = TRUE) {
  apch_config_loglik_cpp(
    log_alloc_mat = log_alloc_mat,
    alloc_code = as.integer(alloc_code),
    alloc_logw_cond = alloc_logw_cond,
    use_parallel = use_parallel,
    grain_size = 512L
  )
}

# Global weight of each allocation state: config prior times the within-config weight.
# The columns of alloc_logw_cond are indexed by config code (.apch_config_code), while
# config_prior follows the rows of config_vec_mat; config_code maps the one onto the
# other. Adding them by position paired "10" with "01" for p >= 2.
.apch_alloc_weight_from_joint <- function(alloc_logw_cond, config_prior, config_code) {
  L <- nrow(alloc_logw_cond)
  out <- numeric(L)
  log_pi <- rep(-Inf, ncol(alloc_logw_cond))
  log_pi[as.integer(config_code) + 1L] <- log(as.numeric(config_prior))
  for (l in seq_len(L)) {
    x <- alloc_logw_cond[l, ] + log_pi
    m <- max(x)
    if (!is.finite(m)) {
      out[l] <- 0
    } else {
      out[l] <- exp(m) * sum(exp(x - m))
    }
  }
  out / sum(out)
}

# Allocation-likelihood builder for one noise component.

.apch_build_block_multivariate_log_alloc_mat <- function(
    Bhat_b,
    Sb,
    idx,
    full_grid_list,
    C_mode,
    C_shared,
    C_list,
    use_parallel
) {
  Kj <- vapply(full_grid_list[idx], length, integer(1))
  alloc_core <- .apch_alloc_core(Kj)
  alloc_mat <- alloc_core$alloc_mat
  alloc_code <- alloc_core$alloc_code

  log_alloc_mat <- .apch_alloc_loglik_mat(
    Bhat_b = Bhat_b,
    Sb = Sb,
    idx = idx,
    full_grid_list = full_grid_list,
    C_mode = C_mode,
    C_shared = C_shared,
    C_list = C_list,
    alloc_mat = alloc_mat,
    use_parallel = use_parallel
  )

  list(
    alloc_mat = alloc_mat,
    alloc_code = alloc_code,
    log_alloc_mat = log_alloc_mat
  )
}

.apch_build_apchs_alloc_logw_cond_from_log_alloc <- function(
    log_alloc_mat,
    alloc_mat,
    alloc_code,
    idx,
    pi_list,
    prior_q = NULL,
    prior_frac = 0.001,
    prior_alpha_min = 1e-4,
    prior_alpha_cap = 5,
    prior_alpha_cap_null = 10,
    keep_alloc_post_prob = FALSE,
    keep_alloc_post_colsum = FALSE,
    blas_threads = 1L
) {
  null_col <- which(rowSums(alloc_mat > 0L) == 0L)
  if (length(null_col) == 0L) {
    null_col <- NULL
  }
  fit_alloc <- .apch_fit_config_prior(
    log_prob_mat = log_alloc_mat,
    x0 = rep(1 / ncol(log_alloc_mat), ncol(log_alloc_mat)),
    prior_q = prior_q,
    prior_frac = prior_frac,
    prior_alpha_min = prior_alpha_min,
    prior_alpha_cap = prior_alpha_cap,
    prior_alpha_cap_null = prior_alpha_cap_null,
    null_col = null_col,
    keep_post_prob = keep_alloc_post_prob,
    keep_post_colsum = keep_alloc_post_colsum,
    blas_threads = blas_threads
  )
  alloc_prior <- as.numeric(fit_alloc$config_prior)
  base <- apch_build_apchs_alloc_logw_cond_cpp(
    alloc_code = as.integer(alloc_code),
    alloc_prior = alloc_prior,
    p = length(idx)
  )
  alloc_logw_cond <- base$alloc_logw_cond
  config_prior_m <- base$config_prior_m
  list(
    alloc_logw_cond = alloc_logw_cond,
    alloc_prior = alloc_prior,
    config_prior_m = config_prior_m,
    alloc_post_prob = fit_alloc$post_config_prob,
    alloc_post_colsum = fit_alloc$post_colsum,
    alloc_ll = fit_alloc$ll
  )
}

# Reuse a fixed configuration prior after allocation-level fitting.
# For APCH-s, the weak prior is applied at the allocation layer
# and then aggregated to the configuration margin on the default path.
.apch_config_posterior_from_fixed_prior <- function(log_prob_mat, prior) {
  prior <- as.numeric(prior)
  prior <- prior / sum(prior)
  norm_res <- posterior_from_logL_cpp(log_prob_mat, prior)
  list(
    config_prior = prior,
    post_config_prob = norm_res$post_prob,
    ll = norm_res$ll
  )
}

#' Fit APCH configuration-layer parameters
#'
#' Wrapper around the configuration layer using a unified noise-input interface.
#'
#' The input \code{C} may be either:
#' \itemize{
#'   \item a single \eqn{p \times p} matrix shared across all effects; or
#'   \item a list of length \code{n} of \eqn{p \times p} matrices.
#' }
#'
#' The current implementation uses Gaussian feature-layer mixtures only and
#' directly computes the full configuration likelihood matrix.
#'
#' @param Bhat Numeric matrix of observed effect estimates with dimension
#'   \eqn{n \times p}.
#' @param S Numeric matrix of standard errors with the same dimensions as \code{Bhat}.
#' @param C Noise structure, either a shared \eqn{p \times p} matrix or a list
#'   of length \code{n} of \eqn{p \times p} matrices.
#' @param slab_list List of feature-specific non-null slab SD grids.
#' @param pi_list List of feature-specific conditional non-null weights.
#' @param p_null Optional list of feature-specific null masses from the
#'   marginal fit.
#' @param joint_mode Joint model: \code{"apch_s"}, \code{"apch_s_full"}, or
#'   \code{"apch_b"}; see \code{\link{apch}}.
#' @param config_vec_mat Optional binary configuration matrix. If \code{NULL},
#'   all configurations are generated internally.
#' @param workers Integer number of threads for the configuration layer.
#'   If \code{NULL}, uses \code{workers_multicore} or \code{workers_multisession}
#'   based on plan.
#' @param workers_multicore Default thread count when using \code{multicore}.
#' @param workers_multisession Default thread count when using \code{multisession}.
#' @param plan Optional future plan name used only to choose defaults.
#' @param use_parallel Logical; whether to enable parallel likelihood evaluation.
#' @param verbose Logical; whether to print diagnostic messages.
#' @param dirichlet_count Nonnegative scalar controlling the total pseudo-count
#'   strength as a fraction of \code{n}. The total pseudo-count mass is
#'   \code{dirichlet_count * n}; set to \code{0} to disable this weak prior.
#' @param pairwise_alloc_support Optional pairwise allocation-support object
#'   returned by \code{\link{build_pairwise_alloc_support}}. This is used by
#'   \code{joint_mode="apch_s"}.
#' @param marginal_loglik_cache Optional internal cache of final 1D marginal
#'   log-likelihood matrices aligned with \code{slab_list}. This is reused by
#'   \code{joint_mode="apch_s"} and \code{"apch_s_full"} when the noise graph
#'   contains singleton components.
#' @section Performance notes:
#' For joint_mode \code{"apch_s_full"} or \code{"apch_s"}
#' with shared, block-diagonal
#' noise, the allocation likelihood is assembled over the true noise connected
#' components and combined exactly before collapsing to configurations. For large \code{n} and small \code{p}
#' (e.g., \code{n} in 1e4--1e5, \code{p} in 5--8), effect-parallel evaluation
#' inside each block is typically faster than block-parallel evaluation. This
#' is the default strategy.
#' @return A list containing configuration-layer outputs, with rows of
#'   \code{post_config_prob} normalized to sum to 1.
#' @noRd
.apch_fit_joint <- function(
    Bhat, S, C,
    slab_list, pi_list,
    p_null = NULL,
    joint_mode = c("apch_s", "apch_s_full", "apch_b"),
    config_vec_mat = NULL,
    workers         = NULL,
    workers_multicore = 16,
    workers_multisession = 8,
    plan            = NULL,
    use_parallel    = TRUE,
    verbose         = TRUE,
    dirichlet_count = 0.001,
    pairwise_alloc_support = NULL,
    marginal_loglik_cache = NULL
) {
  joint_mode <- match.arg(joint_mode)
  p <- ncol(Bhat)

  if (identical(joint_mode, "apch_s") && is.null(pairwise_alloc_support)) {
    stop("joint_mode='apch_s' requires pairwise_alloc_support.")
  }
  if (!is.null(pairwise_alloc_support) && !identical(joint_mode, "apch_s")) {
    stop("pairwise_alloc_support is only used by joint_mode='apch_s'.")
  }


  marginal_inputs <- .apch_prepare_marginal_inputs(
    slab_list = slab_list,
    pi_list = pi_list,
    p = p
  )
  slab_list <- marginal_inputs$slab_list
  pi_list <- marginal_inputs$pi_list
  full_grid_list <- marginal_inputs$full_grid_list
  
  if (is.null(config_vec_mat)) {
    config_vec <- generate_all_configs(p)
  } else {
    config_vec <- as.matrix(config_vec_mat)
  }
  
  resolved <- .apch_resolve_workers(plan, workers, workers_multicore, workers_multisession)
  workers_em <- max(1L, as.integer(resolved$workers))
  mixsqp_threads <- .apch_mixsqp_threads(n = nrow(Bhat), workers = workers_em)
  RcppParallel::setThreadOptions(
    numThreads = if (isTRUE(use_parallel)) workers_em else 1L
  )
  C_obj <- .apch_normalize_C(Bhat, C)
  C_mode <- C_obj$C_mode
  C_shared <- C_obj$C_shared
  C_list <- C_obj$C_list

  if (isTRUE(verbose)) {
    cat(sprintf(
      '[Joint] start: joint_mode=%s, n=%d, p=%d, configs=%d, C_mode=%s, threads=%d\n',
      joint_mode,
      nrow(Bhat),
      p,
      nrow(config_vec),
      C_mode,
      if (isTRUE(use_parallel)) workers_em else 1L
    ))
  }

  pairwise_alloc_support <- .apch_validate_pairwise_alloc_support(
    pairwise_alloc_support,
    feature_names = colnames(Bhat)
  )
  pairwise_alloc_support_out <- pairwise_alloc_support

  alloc_prior_m <- NULL
  lfsr_alloc_state <- NULL
  alloc_mat <- NULL
  alloc_code <- NULL
  alloc_logw_cond <- NULL
  if (identical(joint_mode, "apch_b")) {
    # APCH-b branch: configuration likelihood is built directly via
    # plugin slab weights, with no allocation-prior fitting.
    # Dedicated APCH-b pathway (p-arbitrary):
    # Build the full allocation likelihood grid, then collapse directly to
    # the selected configurations using plugin slab weights.
    if (isTRUE(verbose)) {
      cat('[Joint][APCH-b] building allocation likelihood matrix and collapsing to selected configs...\n')
    }
    apchb <- .apch_apchb_logfmat_direct(
      Bhat = Bhat,
      S = S,
      C_mode = C_mode,
      C_shared = C_shared,
      C_list = C_list,
      full_grid_list = full_grid_list,
      pi_list = pi_list,
      config_vec = config_vec,
      use_parallel = use_parallel
    )
    if (isTRUE(verbose)) {
      cat(sprintf('[Joint][APCH-b] likelihood build done: kernel_eval=%d, alloc_states=%d\n',
                  apchb$n_kernel_eval, apchb$n_alloc_state))
    }
    log_fmat <- apchb$log_fmat
  } else {
    # Joint layer for apch_s_full / apch_s:
    # compute the allocation likelihood once on the current support, reusing
    # lower-order caches on singleton and pairwise noise components when available.
    if (isTRUE(verbose)) {
      cat(sprintf(
        "[Joint][%s] allocation likelihood start (n=%d, alloc_states=%d) ...\n",
        joint_mode,
        nrow(Bhat),
        if (is.null(pairwise_alloc_support)) prod(vapply(full_grid_list, length, integer(1))) else NA_integer_
      ))
    }
    t_alloc0 <- proc.time()[["elapsed"]]
    alloc_state <- .apch_alloc_loglik_joint(
      Bhat = Bhat,
      S = S,
      full_grid_list = full_grid_list,
      slab_list = slab_list,
      pi_list = pi_list,
      C_mode = C_mode,
      C_shared = C_shared,
      C_list = C_list,
      pairwise_alloc_support = pairwise_alloc_support,
      marginal_loglik_cache = marginal_loglik_cache,
      use_parallel = use_parallel,
      verbose = verbose
    )
    t_alloc1 <- proc.time()[["elapsed"]]
    log_alloc_mat <- alloc_state$log_alloc_mat
    alloc_mat <- alloc_state$alloc_mat
    alloc_code <- alloc_state$alloc_code
    alloc_logw_cond <- alloc_state$alloc_logw_cond
    alloc_state <- NULL
    if (isTRUE(verbose)) {
      cat(sprintf(
        "[Joint][%s] allocation likelihood done: kept=%d, elapsed=%.3fs\n",
        joint_mode,
        nrow(alloc_mat),
        t_alloc1 - t_alloc0
      ))
    }
    pairwise_alloc_support_out <- .apch_strip_pairwise_alloc_support_cache(pairwise_alloc_support)
    pairwise_alloc_support <- NULL
    marginal_loglik_cache <- NULL
    gc(verbose = FALSE)
    alloc_prior_q <- NULL
    if (dirichlet_count > 0 && !is.null(p_null) && length(p_null) == p) {
      alloc_prior_q <- .apch_alloc_prior_center_from_marginals(
        alloc_mat = alloc_mat,
        p_null = p_null,
        pi_list = pi_list
      )
    }
    if (isTRUE(verbose)) {
      cat(sprintf(
        "[Joint][%s] allocation prior fit start (R=%d) ...\n",
        joint_mode,
        ncol(log_alloc_mat)
      ))
    }
    t_prior0 <- proc.time()[["elapsed"]]
    alloc_fit_state <- .apch_build_apchs_alloc_logw_cond_from_log_alloc(
      log_alloc_mat = log_alloc_mat,
      alloc_mat = alloc_mat,
      alloc_code = alloc_code,
      idx = seq_len(p),
      pi_list = pi_list,
      prior_q = alloc_prior_q,
      prior_frac = dirichlet_count,
      keep_alloc_post_prob = FALSE,
      blas_threads = mixsqp_threads
    )
    t_prior1 <- proc.time()[["elapsed"]]
    if (isTRUE(verbose)) {
      cat(sprintf(
        "[Joint][%s] allocation prior fit done: elapsed=%.3fs, mean loglik=%.6f\n",
        joint_mode,
        t_prior1 - t_prior0,
        alloc_fit_state$alloc_ll
      ))
    }
    alloc_logw_cond <- alloc_fit_state$alloc_logw_cond
    alloc_prior_m <- alloc_fit_state$config_prior_m
    gc(verbose = FALSE)
    log_cfg_full <- .apch_config_loglik(
      log_alloc_mat = log_alloc_mat,
      alloc_code = alloc_code,
      alloc_logw_cond = alloc_logw_cond,
      use_parallel = use_parallel
    )
    cfg_code <- .apch_config_code(config_vec)
    log_fmat <- log_cfg_full[, cfg_code + 1L, drop = FALSE]
    lfsr_alloc_state <- list(
      log_alloc_mat = log_alloc_mat,
      alloc_mat = alloc_mat,
      alloc_code = alloc_code,
      alloc_logw_cond = alloc_logw_cond
    )
  }

  if (!identical(joint_mode, "apch_b")) {
    cfg_code <- .apch_config_code(config_vec)
    config_prior <- alloc_prior_m[cfg_code + 1L]
    fit_config <- .apch_config_posterior_from_fixed_prior(
      log_prob_mat = log_fmat,
      prior = config_prior
    )
  } else {
    prior_q <- NULL
    if (dirichlet_count > 0 && !is.null(p_null) && length(p_null) == p) {
      prior_q <- .apch_config_prior_center_from_marginals(
        config_vec_mat = config_vec,
        p_null = p_null
      )
    }
    if (isTRUE(verbose)) {
      cat(sprintf('[Joint] configuration prior fit start (R=%d) ...\n', ncol(log_fmat)))
    }
    fit_config <- .apch_fit_config_prior(
      log_prob_mat = log_fmat,
      x0 = rep(1 / ncol(log_fmat), ncol(log_fmat)),
      prior_q = prior_q,
      prior_frac = dirichlet_count,
      prior_alpha_cap = 5,
      prior_alpha_cap_null = 10,
      null_col = which(rowSums(config_vec) == 0L),
      blas_threads = mixsqp_threads
    )
    if (isTRUE(verbose)) {
      cat(sprintf('[Joint] configuration prior fit done: mean loglik=%.6f\n', fit_config$ll))
    }
  }
  config_labels <- apply(config_vec, 1L, paste0, collapse = "")
  joint_res <- list(
    post_config_prob = fit_config$post_config_prob,
    config_prior = fit_config$config_prior,
    config_vec_mat = config_vec
  )
  if (!is.null(pairwise_alloc_support_out)) {
    joint_res$pairwise_alloc_support <- pairwise_alloc_support_out
  }
  if (!is.null(lfsr_alloc_state)) {
    joint_res$lfsr_alloc_state <- lfsr_alloc_state
  }
  config_prior_df <- data.frame(
    L = rowSums(config_vec),
    weight = as.numeric(fit_config$config_prior),
    config01 = config_labels,
    config_vec,
    row.names = NULL,
    check.names = FALSE
  )
  config_prior_df <- config_prior_df[order(-config_prior_df$L, -config_prior_df$weight), , drop = FALSE]
  rownames(config_prior_df) <- NULL
  joint_res$config_prior_df <- config_prior_df
  if (!is.null(alloc_mat) && !is.null(alloc_logw_cond)) {
    alloc_weight <- .apch_alloc_weight_from_joint(
      alloc_logw_cond = alloc_logw_cond,
      config_prior = fit_config$config_prior,
      config_code = .apch_config_code(config_vec)
    )
    alloc_state_df <- data.frame(
      alloc_state = apply(alloc_mat, 1L, paste0, collapse = ","),
      weight = as.numeric(alloc_weight),
      L = rowSums(alloc_mat > 0L),
      alloc_mat,
      row.names = NULL,
      check.names = FALSE
    )
    colnames(alloc_state_df)[4:(3 + ncol(alloc_mat))] <- colnames(config_vec)
    alloc_state_df <- alloc_state_df[order(-alloc_state_df$L, -alloc_state_df$weight), , drop = FALSE]
    rownames(alloc_state_df) <- NULL
    joint_res$alloc_state_df <- alloc_state_df
    joint_res$alloc_cond_state <- list(
      alloc_mat = alloc_mat,
      alloc_code = alloc_code,
      alloc_logw_cond = alloc_logw_cond
    )
  }
  if (isTRUE(verbose)) {
    cat('[Joint] apch_joint_fit done.\n')
  }
  
  joint_res
}
