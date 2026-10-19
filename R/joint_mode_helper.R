## joint_mode_helper.R
## Joint-mode helpers for APCH-b and APCH-s.

.apch_build_apchb_alloc_logw_cond <- function(
    alloc_mat,
    alloc_code,
    idx,
    pi_list
) {
  apch_build_apchb_alloc_logw_cond_cpp(
    alloc_mat = alloc_mat,
    alloc_code = as.integer(alloc_code),
    pi_block = pi_list[idx]
  )
}

.apch_apchb_logfmat_direct_shared_components <- function(
    Bhat,
    S,
    C_shared,
    pi_list,
    full_grid_list,
    config_vec,
    use_parallel
) {
  p <- ncol(Bhat)

  comps <- .apch_noise_components(
    C_mode = "shared",
    C_shared = C_shared,
    C_list = NULL,
    p = p
  )

  slab_list <- lapply(full_grid_list, function(g) {
    g <- as.numeric(g)
    if (length(g) <= 1L) numeric(0) else g[-1L]
  })

  block_cfg_loglik <- vector("list", length(comps))
  n_kernel_eval <- 0L

  for (m in seq_along(comps)) {
    idx <- as.integer(comps[[m]])
    bm <- length(idx)
    block_res <- if (bm == 1L) {
      .apch_build_block_univariate(
        Bhat = Bhat,
        idx = idx,
        slab_list = slab_list,
        pi_list = pi_list,
        C_mode = "shared",
        C_shared = C_shared,
        C_list = NULL
      )
    } else {
      core <- .apch_build_block_multivariate_log_alloc_mat(
        Bhat_b = Bhat[, idx, drop = FALSE],
        Sb = S[, idx, drop = FALSE],
        idx = idx,
        full_grid_list = full_grid_list,
        C_mode = "shared",
        C_shared = C_shared,
        C_list = NULL,
        use_parallel = use_parallel
      )
      list(
        alloc_mat = core$alloc_mat,
        alloc_code = core$alloc_code,
        log_alloc_mat = core$log_alloc_mat,
        alloc_logw_cond = .apch_build_apchb_alloc_logw_cond(
          alloc_mat = core$alloc_mat,
          alloc_code = core$alloc_code,
          idx = idx,
          pi_list = pi_list
        )
      )
    }

    n_kernel_eval <- n_kernel_eval + ncol(block_res$log_alloc_mat)
    block_cfg_loglik[[m]] <- .apch_config_loglik(
      log_alloc_mat = block_res$log_alloc_mat,
      alloc_code = block_res$alloc_code,
      alloc_logw_cond = block_res$alloc_logw_cond,
      use_parallel = use_parallel
    )
  }

  n <- nrow(block_cfg_loglik[[1L]])
  R <- nrow(config_vec)
  log_fmat <- matrix(0, nrow = n, ncol = R)
  for (m in seq_along(comps)) {
    idx <- as.integer(comps[[m]])
    code <- .apch_config_code(config_vec[, idx, drop = FALSE])
    log_fmat <- log_fmat + block_cfg_loglik[[m]][, code + 1L, drop = FALSE]
  }

  Kj <- vapply(full_grid_list, length, integer(1))
  n_alloc_state <- as.integer(prod(Kj))

  list(
    log_fmat = log_fmat,
    n_alloc = as.integer(n_kernel_eval),
    n_alloc_state = n_alloc_state,
    n_kernel_eval = as.integer(n_kernel_eval),
    fast_path = (length(comps) > 1L),
    components = vapply(comps, length, integer(1)),
    n_components = length(comps),
    n_config_full = as.integer(2^p),
    n_config_selected = nrow(config_vec)
  )
}

.apch_apchb_logfmat_direct <- function(
    Bhat,
    S,
    C_mode,
    C_shared,
    C_list,
    full_grid_list,
    pi_list,
    config_vec,
    use_parallel
) {
  p <- ncol(Bhat)
  Kj <- vapply(full_grid_list, length, integer(1))

  if (.apch_is_shared(C_mode)) {
    return(.apch_apchb_logfmat_direct_shared_components(
      Bhat = Bhat,
      S = S,
      C_shared = C_shared,
      pi_list = pi_list,
      full_grid_list = full_grid_list,
      config_vec = config_vec,
      use_parallel = use_parallel
    ))
  }

  core <- .apch_build_block_multivariate_log_alloc_mat(
    Bhat_b = Bhat,
    Sb = S,
    idx = seq_len(p),
    full_grid_list = full_grid_list,
    C_mode = C_mode,
    C_shared = C_shared,
    C_list = C_list,
    use_parallel = use_parallel
  )
  alloc_mat <- core$alloc_mat
  alloc_code <- core$alloc_code
  log_alloc_mat <- core$log_alloc_mat

  alloc_logw_cond <- .apch_build_apchb_alloc_logw_cond(
    alloc_mat = alloc_mat,
    alloc_code = alloc_code,
    idx = seq_len(p),
    pi_list = pi_list
  )
  log_cfg_full <- .apch_config_loglik(
    log_alloc_mat = log_alloc_mat,
    alloc_code = alloc_code,
    alloc_logw_cond = alloc_logw_cond,
    use_parallel = use_parallel
  )

  cfg_code <- .apch_config_code(config_vec)
  log_fmat <- log_cfg_full[, cfg_code + 1L, drop = FALSE]

  list(
    log_fmat = log_fmat,
    n_alloc = nrow(alloc_mat),
    n_alloc_state = as.integer(prod(Kj)),
    n_kernel_eval = nrow(alloc_mat),
    n_config_full = 2^p,
    n_config_selected = nrow(config_vec)
  )
}

.apch_build_block_univariate <- function(
    Bhat,
    idx,
    slab_list,
    pi_list,
    C_mode,
    C_shared,
    C_list
) {
  j <- idx[1]
  cjj <- .apch_noise_diag_vec(C_mode, C_shared, C_list, j, nrow(Bhat))

  slab_j <- as.numeric(slab_list[[j]])
  K_j <- length(slab_j)
  alloc_code <- c(0L, rep(1L, K_j))
  z_j <- as.numeric(Bhat[, j])
  s_j <- sqrt(cjj)

  log_alloc_mat <- loglik_normalmix_cpp(z_j, s_j, c(0, slab_j))
  # A missing estimate carries no information about this feature.
  log_alloc_mat[!is.finite(z_j), ] <- 0

  alloc_logw_cond <- matrix(-Inf, nrow = K_j + 1L, ncol = 2L)
  alloc_logw_cond[1L, 1L] <- 0
  if (K_j > 0L) {
    alloc_logw_cond[2:(K_j + 1L), 2L] <- log(as.numeric(pi_list[[j]]))
  }

  list(
    alloc_code = alloc_code,
    log_alloc_mat = log_alloc_mat,
    alloc_logw_cond = alloc_logw_cond
  )
}

.apch_alloc_loglik_joint <- function(
    Bhat,
    S,
    full_grid_list,
    slab_list,
    pi_list,
    C_mode,
    C_shared,
    C_list,
    pairwise_alloc_support,
    marginal_loglik_cache,
    use_parallel,
    verbose
) {
  p <- ncol(Bhat)
  Kj_full <- vapply(full_grid_list, length, integer(1))
  has_support <- !is.null(pairwise_alloc_support)
  support_loglik_cache <- if (has_support) {
    pairwise_alloc_support$loglik_cache
  } else {
    .apch_make_component_loglik_cache()
  }
  component_loglik_cache <- .apch_make_component_loglik_cache(
    order1 = marginal_loglik_cache,
    order2 = support_loglik_cache$order2,
    order3 = support_loglik_cache$order3,
    order4 = support_loglik_cache$order4
  )
  if (has_support) {
    alloc_core <- .apch_alloc_core(Kj_full)
    prune_res <- .apch_prune_alloc_mat_by_pair_support(
      alloc_mat = alloc_core$alloc_mat,
      pairwise_alloc_support = pairwise_alloc_support
    )
    alloc_mat <- prune_res$alloc_mat
    if (nrow(alloc_mat) == 0L) {
      stop("pairwise allocation pruning removed all allocation states.")
    }
    alloc_code <- .apch_alloc_code_from_mat(alloc_mat)
    if (isTRUE(verbose) && !is.null(prune_res)) {
      cat(sprintf("[Joint] pairwise prune done: kept=%d states\n", prune_res$n_after_pair))
      if (length(pairwise_alloc_support$triples) > 0L) {
        cat(sprintf("[Joint] triple prune done: kept=%d states\n", prune_res$n_after_triple))
      }
    }
  } else {
    alloc_core <- .apch_alloc_core(Kj_full)
    alloc_mat <- alloc_core$alloc_mat
    alloc_code <- alloc_core$alloc_code
  }

  # Allocation-likelihood core for apch_s_full / apch_s:
  # - if noise is block-diagonal, we compute blockwise log-likelihoods and
  #   combine exactly (logliks add across independent blocks).
  # - otherwise we compute the full log_alloc_mat directly.
  if (p == 1L) {
    block1 <- .apch_build_block_univariate(
      Bhat = Bhat,
      idx = 1L,
      slab_list = slab_list,
      pi_list = pi_list,
      C_mode = C_mode,
      C_shared = C_shared,
      C_list = C_list
    )
    return(list(
      log_alloc_mat = block1$log_alloc_mat,
      alloc_mat = alloc_mat,
      alloc_code = alloc_code,
      alloc_logw_cond = NULL
    ))
  }

  if (!.apch_is_shared(C_mode)) {
    log_alloc_mat <- .apch_alloc_loglik_mat(
      Bhat_b = Bhat,
      Sb = S,
      idx = seq_len(p),
      full_grid_list = full_grid_list,
      C_mode = C_mode,
      C_shared = C_shared,
      C_list = C_list,
      alloc_mat = alloc_mat,
      use_parallel = use_parallel
    )
    return(list(
      log_alloc_mat = log_alloc_mat,
      alloc_mat = alloc_mat,
      alloc_code = alloc_code,
      alloc_logw_cond = NULL
    ))
  }

  noise_blocks <- .apch_noise_components(
    C_mode = C_mode,
    C_shared = C_shared,
    C_list = C_list,
    p = p
  )
  assembly_blocks <- .apch_joint_assembly_blocks(noise_blocks, component_loglik_cache)
  block_logs <- lapply(seq_along(assembly_blocks), function(b) {
    idx <- as.integer(assembly_blocks[[b]])
    bm <- length(idx)
    cache_block <- if (bm <= 4L) {
      .apch_get_component_loglik_cache(
        component_loglik_cache = component_loglik_cache,
        idx = idx
      )
    } else {
      NULL
    }
    if (length(cache_block) > 0L) {
      return(cache_block)
    }
    if (bm == 1L) {
      return(.apch_build_block_univariate(
        Bhat = Bhat,
        idx = idx,
        slab_list = slab_list,
        pi_list = pi_list,
        C_mode = C_mode,
        C_shared = C_shared,
        C_list = C_list
      )$log_alloc_mat)
    }
    .apch_build_block_multivariate_log_alloc_mat(
      Bhat_b = Bhat[, idx, drop = FALSE],
      Sb = S[, idx, drop = FALSE],
      idx = idx,
      full_grid_list = full_grid_list,
      C_mode = C_mode,
      C_shared = C_shared,
      C_list = C_list,
      use_parallel = use_parallel
    )$log_alloc_mat
  })
  log_alloc_mat <- apch_combine_log_alloc_blocks_cpp(
    log_alloc_blocks = block_logs,
    blocks = assembly_blocks,
    Kvec = Kj_full,
    alloc_mat = alloc_mat
  )

  list(
    log_alloc_mat = log_alloc_mat,
    alloc_mat = alloc_mat,
    alloc_code = alloc_code,
    alloc_logw_cond = NULL
  )
}

.apch_alloc_posterior_from_config_prior <- function(
    log_alloc_mat,
    alloc_code,
    alloc_logw_cond,
    config_vec_mat,
    config_prior
) {
  apch_alloc_posterior_from_config_prior_cpp(
    log_alloc_mat = log_alloc_mat,
    alloc_code = as.integer(alloc_code),
    alloc_logw_cond = alloc_logw_cond,
    config_code = .apch_config_code(config_vec_mat),
    config_prior = as.numeric(config_prior)
  )
}

.apch_build_apchb_lfsr_alloc_state <- function(
    Bhat,
    S,
    C_mode,
    C_shared,
    C_list,
    slab_list,
    pi_list,
    marginal_loglik_cache = NULL,
    use_parallel = TRUE,
    verbose = TRUE
) {
  p <- ncol(Bhat)
  marginal_inputs <- .apch_prepare_marginal_inputs(
    slab_list = slab_list,
    pi_list = pi_list,
    p = p
  )
  full_grid_list <- marginal_inputs$full_grid_list

  if (isTRUE(verbose)) {
    cat("[Joint][APCH-b] building lfsr allocation state...\n")
  }
  t0 <- proc.time()[["elapsed"]]
  alloc_state <- .apch_alloc_loglik_joint(
    Bhat = Bhat,
    S = S,
    full_grid_list = full_grid_list,
    slab_list = slab_list,
    pi_list = pi_list,
    C_mode = C_mode,
    C_shared = C_shared,
    C_list = C_list,
    pairwise_alloc_support = NULL,
    marginal_loglik_cache = marginal_loglik_cache,
    use_parallel = use_parallel,
    verbose = FALSE
  )
  alloc_logw_cond <- .apch_build_apchb_alloc_logw_cond(
    alloc_mat = alloc_state$alloc_mat,
    alloc_code = alloc_state$alloc_code,
    idx = seq_len(p),
    pi_list = pi_list
  )
  t1 <- proc.time()[["elapsed"]]
  if (isTRUE(verbose)) {
    cat(sprintf(
      "[Joint][APCH-b] lfsr allocation state done: alloc=%d, elapsed=%.3fs\n",
      nrow(alloc_state$alloc_mat),
      t1 - t0
    ))
  }
  list(
    log_alloc_mat = alloc_state$log_alloc_mat,
    alloc_mat = alloc_state$alloc_mat,
    alloc_code = alloc_state$alloc_code,
    alloc_logw_cond = alloc_logw_cond
  )
}
