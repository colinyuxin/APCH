.apch_compute_lfdr_bundle <- function(joint_res) {
  post_config_prob <- joint_res$post_config_prob
  if (is.null(post_config_prob)) {
    stop("joint_res$post_config_prob is required to compute LFDR.")
  }
  config_vec_mat <- joint_res$config_vec_mat
  if (is.null(config_vec_mat)) {
    stop("joint_res$config_vec_mat is required to compute LFDR.")
  }
  config_vec_mat <- as.matrix(config_vec_mat)
  storage.mode(config_vec_mat) <- "integer"
  apch_subset_lfdr_cpp(
    post_config_prob = post_config_prob,
    config_vec_mat = config_vec_mat
  )
}


.apch_subset_match_index <- function(available_subset_vec_mat, target_subset_vec_mat) {
  available_key <- apply(available_subset_vec_mat, 1L, paste0, collapse = "")
  target_key <- apply(target_subset_vec_mat, 1L, paste0, collapse = "")
  idx <- match(target_key, available_key)
  if (anyNA(idx)) {
    stop("subset contains rows outside the non-empty subset set.")
  }
  idx
}

#' Subset-level local false sign rate (LFSR)
#'
#' Computes subset-level LFSR for APCH models using allocation-vector
#' components. The computation targets the joint event
#' \eqn{H^U_{A,i} \cap A_i(U,s)}. Under the exact-null model this equals the
#' usual sign-based lfsr.
#'
#' @param marginal_res Marginal-layer result list.
#' @param joint_res Joint-layer result list.
#' @param Bhat Numeric matrix of observed effects (n x p).
#' @param S Numeric matrix of standard errors (n x p).
#' @param C Noise structure (NULL, shared p x p, or list length n).
#' @param subset Subset specification: integer indices or a binary matrix
#'   (rows = subsets).
#'   If NULL, all non-empty subsets are used.
#' @param workers Integer number of threads for likelihood evaluation.
#' @param use_parallel Logical; whether to enable parallel likelihood evaluation.
#' @param verbose Logical; print diagnostic messages.
#' @param pmvnorm_maxpts Maximum number of points for multivariate normal
#'   probability calculation (non-diagonal noise only).
#' @param pmvnorm_abseps Absolute tolerance for multivariate normal probability
#'   calculation (non-diagonal noise only).
#' @param pmvnorm_releps Relative tolerance for multivariate normal probability
#'   calculation (non-diagonal noise only).
#' @param pmvnorm_jitter Diagonal jitter added to posterior covariance blocks
#'   before calling \code{mvtnorm::pmvnorm} (non-diagonal noise only).
#'
#' @return A numeric matrix of LFSR values (n x M), with rows = effects and
#'   columns = subsets.
#'
#' \code{Bhat} and \code{S} must contain exactly the same effect rows as
#' \code{joint_res$lfsr_alloc_state}.
#'
#' @noRd
.apch_lfsr_matrix <- function(
    marginal_res,
    joint_res,
    Bhat, S,
    C = NULL,
    subset = NULL,
    workers = 1,
    use_parallel = TRUE,
    verbose = TRUE,
    pmvnorm_maxpts = NULL,
    pmvnorm_abseps = NULL,
    pmvnorm_releps = NULL,
    pmvnorm_jitter = NULL
) {
  z_scaled <- .apch_scale_to_zscore(Bhat, S)

  pmvnorm <- .apch_lfsr_pmvnorm_control(
    maxpts = pmvnorm_maxpts,
    abseps = pmvnorm_abseps,
    releps = pmvnorm_releps,
    jitter = pmvnorm_jitter
  )
  t0 <- proc.time()[["elapsed"]]
  state <- .apch_prepare_lfsr_state(
    marginal_res = marginal_res,
    joint_res = joint_res,
    Bhat = z_scaled$Bhat,
    S = z_scaled$S,
    C = C,
    workers = workers,
    use_parallel = use_parallel,
    verbose = verbose
  )
  t1 <- proc.time()[["elapsed"]]
  if (isTRUE(verbose)) {
    cat(sprintf(
      "[LFSR posterior] state prepared: n=%d p=%d elapsed=%.3fs\n",
      state$n,
      state$p,
      t1 - t0
    ))
  }
  subset_vec_mat <- .apch_parse_subset(subset, p = state$p)
  out <- .apch_eval_lfsr_state(
    state = state,
    subset_vec_mat = subset_vec_mat,
    effect_idx = NULL,
    pmvnorm = pmvnorm
  )
  t2 <- proc.time()[["elapsed"]]
  if (isTRUE(verbose)) {
    cat(sprintf(
      "[LFSR posterior] evaluated: effects=%d subsets=%d elapsed=%.3fs\n",
      nrow(out),
      ncol(out),
      t2 - t1
    ))
  }
  out
}


.apch_prepare_lfsr_state <- function(
    marginal_res,
    joint_res,
    Bhat, S,
    C = NULL,
    workers = 1,
    use_parallel = TRUE,
    verbose = TRUE,
    alloc_state_override = NULL
) {
  n <- nrow(Bhat)
  p <- ncol(Bhat)

  marginal <- .apch_prepare_marginal_inputs(
    slab_list = marginal_res$slab_list,
    pi_list = marginal_res$pi_list,
    p = p
  )
  full_grid_list <- marginal$full_grid_list

  cfg_df <- joint_res$config_prior_df
  config_vec_mat <- .apch_config_vec_from_df(cfg_df, p = p)
  config_prior <- as.numeric(cfg_df$weight)

  Bhat_z <- Bhat
  S_z <- S

  noise <- .apch_normalize_C(Bhat_z, C)
  diag_ok <- .apch_is_diag_noise(
    C_mode = noise$C_mode,
    C_shared = noise$C_shared,
    C_list = noise$C_list
  )

  workers_em <- max(1L, as.integer(workers))
  alloc_state <- if (is.null(alloc_state_override)) joint_res$lfsr_alloc_state else alloc_state_override
  if (is.null(alloc_state)) {
    stop("joint_res$lfsr_alloc_state is required for LFSR.")
  }
  if (is.null(alloc_state$alloc_logw_cond)) {
    stop("joint_res$lfsr_alloc_state$alloc_logw_cond is required for LFSR.")
  }
  if (nrow(alloc_state$log_alloc_mat) != n) {
    stop("Bhat/S must include exactly the same effects as joint_res$lfsr_alloc_state.")
  }
  alloc_mat <- alloc_state$alloc_mat
  alloc_code <- alloc_state$alloc_code
  log_alloc_mat <- alloc_state$log_alloc_mat
  alloc_logw_cond <- alloc_state$alloc_logw_cond

  alpha_mat <- .apch_alloc_posterior_from_config_prior(
    log_alloc_mat = log_alloc_mat,
    alloc_code = alloc_code,
    alloc_logw_cond = alloc_logw_cond,
    config_vec_mat = config_vec_mat,
    config_prior = config_prior
  )

  state <- list(
    z = Bhat_z,
    S = S_z,
    n = n,
    p = p,
    feature_names = .apch_feature_names(Bhat_z, p),
    effect_names = .apch_effect_names(Bhat_z, n),
    full_grid_list = full_grid_list,
    alloc_mat = alloc_mat,
    alpha_mat = alpha_mat,
    diag_ok = diag_ok,
    use_parallel_lfsr = isTRUE(use_parallel) && (workers_em > 1L),
    workers_lfsr = workers_em
  )

  if (isTRUE(diag_ok)) {
    state$e_diag <- .apch_noise_diag_mat(
      C_mode = noise$C_mode,
      C_shared = noise$C_shared,
      C_list = noise$C_list,
      n = n
    )
  } else {
    if (!requireNamespace("mvtnorm", quietly = TRUE)) {
      stop("Non-diagonal noise requires the mvtnorm package.")
    }
    shared_has_missing <- identical(noise$C_mode, "shared") &&
      any(!is.finite(Bhat_z) | !is.finite(S_z))
    if (identical(noise$C_mode, "shared") && !shared_has_missing) {
      state$C_shared <- noise$C_shared
      state$C_list <- list()
      state$use_shared <- TRUE
      state$use_shared_by_mask <- FALSE
      state$shared_lfsr_cache <- lfsr_nondiag_shared_build_cache_cpp(
        C_shared = noise$C_shared,
        alloc_mat = alloc_mat,
        grid_list = full_grid_list,
        block_tol = 1e-12
      )
    } else if (identical(noise$C_mode, "shared")) {
      state$C_shared <- noise$C_shared
      state$C_list <- list()
      state$use_shared <- FALSE
      state$use_shared_by_mask <- TRUE
      state$shared_mask_cache <- .apch_build_shared_mask_lfsr_cache(
        z = Bhat_z,
        C_shared = noise$C_shared,
        alloc_mat = alloc_mat,
        grid_list = full_grid_list
      )
    } else {
      state$C_shared <- matrix(0, 0, 0)
      state$C_list <- noise$C_list
      state$use_shared <- FALSE
      state$use_shared_by_mask <- FALSE
      state$list_lfsr_cache <- lfsr_nondiag_list_build_cache_cpp(
        z = Bhat_z,
        C_list = noise$C_list,
        block_tol = 1e-4
      )
    }
  }

  state
}

.apch_build_shared_mask_lfsr_cache <- function(z, C_shared, alloc_mat, grid_list) {
  obs_mat <- is.finite(z)
  mask_key <- apply(obs_mat, 1L, paste0, collapse = "")
  groups <- split(seq_len(nrow(z)), mask_key)
  lapply(groups, function(effect_idx) {
    obs <- obs_mat[effect_idx[1L], ]
    obs_idx <- which(obs)
    if (length(obs_idx) == 0L) {
      return(list(
        effect_idx = effect_idx,
        obs = obs,
        obs_idx = obs_idx,
        alloc_mat = alloc_mat[, FALSE, drop = FALSE],
        cache = NULL
      ))
    }
    alloc_obs <- alloc_mat[, obs_idx, drop = FALSE]
    list(
      effect_idx = effect_idx,
      obs = obs,
      obs_idx = obs_idx,
      alloc_mat = alloc_obs,
      cache = lfsr_nondiag_shared_build_cache_cpp(
        C_shared = C_shared[obs_idx, obs_idx, drop = FALSE],
        alloc_mat = alloc_obs,
        grid_list = grid_list[obs_idx],
        block_tol = 1e-12
      )
    )
  })
}

.apch_subset_lfsr_alloc_state <- function(alloc_state, effect_idx) {
  if (is.null(alloc_state)) {
    return(NULL)
  }

  out <- alloc_state
  if (!is.null(out$log_alloc_mat)) {
    out$log_alloc_mat <- out$log_alloc_mat[effect_idx, , drop = FALSE]
  }

  out
}

.apch_eval_lfsr_state <- function(
    state,
    subset_vec_mat,
    effect_idx = NULL,
    pmvnorm = .apch_lfsr_pmvnorm_control(),
    attach_dimnames = TRUE
) {
  subset_vec_mat <- as.matrix(subset_vec_mat)
  storage.mode(subset_vec_mat) <- "integer"

  if (is.null(effect_idx)) {
    effect_idx <- seq_len(state$n)
  } else {
    effect_idx <- as.integer(effect_idx)
  }

  if (isTRUE(state$diag_ok)) {
    RcppParallel::setThreadOptions(
      numThreads = if (isTRUE(state$use_parallel_lfsr)) state$workers_lfsr else 1L
    )
    out <- lfsr_diag_indexed_cpp(
      z = state$z,
      e_diag = state$e_diag,
      effect_idx = effect_idx,
      subset_vec_mat = subset_vec_mat,
      alloc_mat = state$alloc_mat,
      grid_list = state$full_grid_list,
      alpha_mat = state$alpha_mat,
      use_parallel = isTRUE(state$use_parallel_lfsr) && (length(effect_idx) > 1L),
      grain_size = 64L
    )
  } else {
    if (isTRUE(state$use_parallel_lfsr) &&
        length(effect_idx) > 1L &&
        identical(.Platform$OS.type, "unix")) {
      chunks <- .apch_split_effect_chunks(effect_idx, state$workers_lfsr)
      mats <- parallel::mclapply(
        X = chunks,
        FUN = function(idx) {
          .apch_eval_lfsr_nondiag_chunk(
            state = state,
            subset_vec_mat = subset_vec_mat,
            effect_idx = idx,
            pmvnorm = pmvnorm
          )
        },
        mc.cores = state$workers_lfsr
      )
      err_idx <- which(vapply(mats, inherits, logical(1), what = "try-error"))
      if (length(err_idx) > 0L) {
        err_msg <- vapply(mats[err_idx], function(x) {
          conditionMessage(attr(x, "condition"))
        }, character(1))
        stop(sprintf(
          "non-diag LFSR chunk failed: %s; %s",
          paste(err_idx, collapse = ","),
          paste(err_msg, collapse = " | ")
        ))
      }
      out <- do.call(rbind, mats)
    } else {
      out <- .apch_eval_lfsr_nondiag_chunk(
        state = state,
        subset_vec_mat = subset_vec_mat,
        effect_idx = effect_idx,
        pmvnorm = pmvnorm
      )
    }
  }

  if (isTRUE(attach_dimnames)) {
    rownames(out) <- state$effect_names[effect_idx]
    colnames(out) <- apply(subset_vec_mat, 1L, paste0, collapse = "")
  }
  out
}

# One row per configuration, one column per feature. Built with matrix() rather than
# t(vapply()): for p = 1, vapply returns a vector and t() turned it into a single
# row, so every configuration was read as one row and the lfsr state became NaN.
.apch_config_vec_from_df <- function(config_prior_df, p) {
  cfg <- as.character(config_prior_df$config01)
  bits <- strsplit(cfg, "", fixed = TRUE)
  if (any(lengths(bits) != p)) {
    stop("config01 labels must have one digit per feature.")
  }
  matrix(as.integer(unlist(bits, use.names = FALSE)), nrow = length(cfg), ncol = p, byrow = TRUE)
}

.apch_eval_lfsr_nondiag_chunk <- function(
    state,
    subset_vec_mat,
    effect_idx,
    pmvnorm
) {
  z_sub <- state$z[effect_idx, , drop = FALSE]
  alpha_sub <- state$alpha_mat[effect_idx, , drop = FALSE]
  if (isTRUE(state$use_shared_by_mask)) {
    .apch_eval_lfsr_shared_mask_chunk(
      state = state,
      subset_vec_mat = subset_vec_mat,
      effect_idx = effect_idx,
      pmvnorm = pmvnorm
    )
  } else if (isTRUE(state$use_shared)) {
    lfsr_nondiag_shared_cached_cpp(
      z = z_sub,
      subset_vec_mat = subset_vec_mat,
      alloc_mat = state$alloc_mat,
      alpha_mat = alpha_sub,
      shared_cache = state$shared_lfsr_cache,
      maxpts = pmvnorm$maxpts,
      abseps = pmvnorm$abseps,
      releps = pmvnorm$releps,
      jitter = pmvnorm$jitter
    )
  } else {
    C_list <- state$C_list[effect_idx]
    list_lfsr_cache <- list(
      obs = state$list_lfsr_cache$obs[effect_idx],
      blocks = state$list_lfsr_cache$blocks[effect_idx]
    )
    lfsr_nondiag_list_cached_cpp(
      z = z_sub,
      C_list = C_list,
      list_cache = list_lfsr_cache,
      subset_vec_mat = subset_vec_mat,
      alloc_mat = state$alloc_mat,
      grid_list = state$full_grid_list,
      alpha_mat = alpha_sub,
      maxpts = pmvnorm$maxpts,
      abseps = pmvnorm$abseps,
      releps = pmvnorm$releps,
      jitter = pmvnorm$jitter
    )
  }
}

.apch_eval_lfsr_shared_mask_chunk <- function(
    state,
    subset_vec_mat,
    effect_idx,
    pmvnorm
) {
  out <- matrix(NA_real_, nrow = length(effect_idx), ncol = nrow(subset_vec_mat))
  for (mask_cache in state$shared_mask_cache) {
    eff <- intersect(effect_idx, mask_cache$effect_idx)
    if (length(eff) == 0L || length(mask_cache$obs_idx) == 0L || is.null(mask_cache$cache)) {
      next
    }
    row_pos <- match(eff, effect_idx)
    obs <- mask_cache$obs
    keep_subset <- rowSums(subset_vec_mat[, !obs, drop = FALSE]) == 0L &
      rowSums(subset_vec_mat[, obs, drop = FALSE]) > 0L
    if (!any(keep_subset)) {
      next
    }
    val <- lfsr_nondiag_shared_cached_cpp(
      z = state$z[eff, mask_cache$obs_idx, drop = FALSE],
      subset_vec_mat = subset_vec_mat[keep_subset, obs, drop = FALSE],
      alloc_mat = mask_cache$alloc_mat,
      alpha_mat = state$alpha_mat[eff, , drop = FALSE],
      shared_cache = mask_cache$cache,
      maxpts = pmvnorm$maxpts,
      abseps = pmvnorm$abseps,
      releps = pmvnorm$releps,
      jitter = pmvnorm$jitter
    )
    out[row_pos, keep_subset] <- val
  }
  out
}

.apch_split_effect_chunks <- function(effect_idx, workers) {
  nchunk <- min(length(effect_idx), max(1L, as.integer(workers)))
  split(effect_idx, cut(seq_along(effect_idx), breaks = nchunk, labels = FALSE))
}

.apch_parse_subset <- function(subset, p) {
  if (is.null(subset)) {
    subset_vec_mat <- generate_all_subsets(p)
  } else if (is.matrix(subset)) {
    subset_vec_mat <- as.matrix(subset)
  } else {
    idx <- as.integer(subset)
    if (length(idx) > 1L) {
      stop(
        "Ambiguous `subset` vector. Use a binary matrix:\n",
        " - one multi-trait subset: matrix(c(...), nrow = 1)\n",
        " - multiple subsets: rbind(...)"
      )
    }
    v <- integer(p); v[idx] <- 1L
    subset_vec_mat <- matrix(v, nrow = 1L)
  }
  subset_vec_mat <- as.matrix(subset_vec_mat)
  storage.mode(subset_vec_mat) <- "integer"
  subset_vec_mat
}

.apch_is_diag_noise <- function(C_mode, C_shared, C_list, tol = 1e-12) {
  !.apch_noise_offdiag_any(C_mode, C_shared, C_list, tol = tol)
}
