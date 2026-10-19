# Marginal layer: one ash fit per feature on the z-score scale, followed by
# grid distillation. Features are distilled in parallel through future.
# Hyperparameters are fixed in .apch_marginal_hyperparams().
.apch_fit_marginal <- function(Bhat, S, workers = 1L, verbose = TRUE) {
  p <- ncol(Bhat)
  n_total <- nrow(Bhat)
  hp <- .apch_marginal_hyperparams()

  .set_single_thread_blas()
  workers_distill <- max(1L, min(as.integer(workers), p))
  sched <- if (p > workers_distill) 4.0 else 1.0

  if (verbose) {
    cat(sprintf("[Marginal] distill workers=%d\n", workers_distill))
  }

  fnm <- colnames(Bhat)
  fnm_display <- .apch_feature_names(Bhat, p)

  fit_cache <- vector("list", p)
  for (j in seq_len(p)) {
    z <- as.numeric(Bhat[, j])
    s <- as.numeric(S[, j])
    keep <- is.finite(z) & is.finite(s) & (s > 0)
    fit_cache[[j]] <- list(z = z[keep], s = s[keep])
  }

  distill_one <- function(j) {
    .set_single_thread_blas()
    cj <- fit_cache[[j]]
    grid0 <- make_mixsd_grid(
      x = cj$z,
      s = cj$s,
      mult = hp$fine_mult,
      s_min_mult = hp$fine_s_min_mult,
      s_max_mult = hp$fine_s_max_mult,
      max_len = hp$fine_max_len
    )
    fit <- distill_grid_path_fixed_null_from_bhat_s(
      bhat = cj$z,
      s = cj$s,
      logL_null = as.numeric(loglik_normalmix_cpp(cj$z, cj$s, 0)),
      init_slab_grid = grid0,
      lambda0 = hp$lambda0,
      mixsqp_control = NULL,
      verbose = FALSE
    )
    last_step <- as.character(max(fit$table$step))
    list(
      grid_all = fit$steps[[last_step]]$slab_grid,
      pi = fit$steps[[last_step]]$pi,
      pi0 = fit$steps[[last_step]]$pi0,
      loglik_cache = if (length(cj$z) == n_total) {
        loglik_normalmix_cpp(cj$z, cj$s, fit$steps[[last_step]]$slab_grid)
      } else {
        NULL
      }
    )
  }

  res_list <- if (workers_distill > 1L) {
    future.apply::future_lapply(
      X = seq_len(p),
      FUN = distill_one,
      future.seed = TRUE,
      future.packages = "apch",
      future.scheduling = sched
    )
  } else {
    lapply(seq_len(p), distill_one)
  }

  slab_list <- vector("list", p)
  pi_list <- vector("list", p)
  p_null <- vector("list", p)
  loglik_cache <- vector("list", p)
  for (j in seq_len(p)) {
    grid_raw <- as.numeric(res_list[[j]]$grid_all)
    pi_raw <- as.numeric(res_list[[j]]$pi)
    p0 <- as.numeric(res_list[[j]]$pi0)

    if (any(grid_raw == 0)) {
      p0 <- sum(pi_raw[grid_raw == 0])
      slab_j <- grid_raw[grid_raw > 0]
      pi_pos <- pi_raw[grid_raw > 0]
    } else {
      slab_j <- grid_raw
      pi_pos <- pi_raw
    }

    if (!length(slab_j)) {
      pi_list[[j]] <- numeric(0)
    } else if (sum(pi_pos) <= 0) {
      pi_list[[j]] <- rep(1 / length(pi_pos), length(pi_pos))
    } else {
      pi_list[[j]] <- pi_pos / sum(pi_pos)
    }

    slab_list[[j]] <- slab_j
    p_null[[j]] <- p0
    loglik_cache[j] <- list(res_list[[j]]$loglik_cache)
  }
  if (!is.null(fnm)) {
    names(slab_list) <- fnm
    names(pi_list) <- fnm
    names(p_null) <- fnm
    names(loglik_cache) <- fnm
  }

  if (isTRUE(verbose)) {
    slab_n <- vapply(slab_list, length, integer(1))
    p0_vec <- vapply(p_null, function(x) as.numeric(x), numeric(1))
    cat("[Marginal] done. Distilled slabs per feature:\n")
    for (j in seq_len(p)) {
      cat(sprintf("  - %s: slabs=%d, p0=%.4f\n", fnm_display[j], slab_n[j], p0_vec[j]))
    }
  }

  list(
    slab_list = slab_list,
    p_null = p_null,
    pi_list = pi_list,
    loglik_cache = loglik_cache
  )
}
