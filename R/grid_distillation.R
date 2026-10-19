## grid_distillation

# utility function: set the number of BLAS/OMP threads to 1
.set_single_thread_blas <- function() {
  Sys.setenv(
    "OPENBLAS_NUM_THREADS"    = "1",
    "MKL_NUM_THREADS"         = "1",
    "OMP_NUM_THREADS"         = "1",
    "VECLIB_MAXIMUM_THREADS"  = "1",
    "BLIS_NUM_THREADS"        = "1"
  )
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    try(RhpcBLASctl::blas_set_num_threads(1), silent = TRUE)
    try(RhpcBLASctl::omp_set_num_threads(1),  silent = TRUE)
  }
}

# generate geometric slab grid; first slab is exactly sigmaamin.
make_mixsd_grid <- function(x, s, mult = 1.35, s_min_mult = 1, s_max_mult = 1.0, max_len = 500) {
  s <- s[is.finite(s) & s > 0]
  x <- x[is.finite(x)]
  sigmaamin <- stats::median(s) * s_min_mult

  if (all(x^2 <= s^2)) {
    sigmaamax <- 8 * stats::median(s)
  } else {
    sigmaamax <- 2 * sqrt(max(x^2 - s^2))
  }
  sigmaamax <- sigmaamax * s_max_mult

  if (mult == 0) {
    g <- c(0, sigmaamax / 2)
  } else {
    ratio <- sigmaamax / sigmaamin
    npoint <- if (!is.finite(ratio) || ratio <= 1) 0L else ceiling(log(ratio) / log(mult))
    sigmaamax <- sigmaamin * mult^npoint
    g <- sigmaamin * mult^(0:npoint)
  }

  g <- unique(sort(g))
  if (length(g) > max_len) {
    keep <- round(seq(1, length(g), length.out = max_len))
    g <- g[unique(keep)]
  }
  g
}

.sym_kl_prior <- function(v1, v2, m1 = 0, m2 = 0) {
  kl12 <- (v1/v2) + (m2-m1)^2/v2 - 1 + log(v2/v1)
  kl21 <- (v2/v1) + (m1-m2)^2/v1 - 1 + log(v1/v2)
  0.5 * (kl12 + kl21)
}

merge_two_grids <- function(s1, s2, w1, w2) {
  if (w1 + w2 > 1e-12) {
    v_new <- (w1 * s1^2 + w2 * s2^2) / (w1 + w2)
    return(sign(s1) * sqrt(max(v_new, .Machine$double.eps)))
  }
  sign(s1) * sqrt(max(s1 * s2, .Machine$double.eps))
}


.fit_with_logL_fast <- function(
    logL,
    sd_full,
    lambda0 = 10,
    mixsqp_control = NULL,
    pi_init = NULL
) {
  sd_full <- as.numeric(sd_full)
  k <- ncol(logL)

  null_comp <- which.min(sd_full)
  prior <- rep(1, k)
  prior[null_comp] <- lambda0

  ctl_allowed <- c("eps", "maxiter.sqp", "verbose", "numiter.em")
  ctl <- if (is.list(mixsqp_control)) mixsqp_control[names(mixsqp_control) %in% ctl_allowed] else NULL
  control0 <- list(verbose = FALSE, eps = 1e-5, numiter.em = 8)
  control <- if (length(ctl)) utils::modifyList(control0, ctl) else control0

  if (is.null(pi_init)) {
    pi_init <- rep(1 / k, k)
  } else {
    pi_init <- as.numeric(pi_init)
    pi_init <- pi_init / sum(pi_init)
  }

  mix_inputs <- build_mixsqp_inputs_cpp(logL, prior)
  out <- .apch_mixsqp(
    L = mix_inputs$A,
    w = mix_inputs$w,
    x0 = pi_init,
    control = control
  )
  pi <- as.numeric(out$x)
  pi <- pi / sum(pi)
  pi0 <- sum(pi[sd_full == sd_full[null_comp]])
  mean_loglik <- mixloglik_from_logL_cpp(logL, mix_inputs$row_max, pi)

  list(
    fit = list(fitted_g = list(pi = pi, sd = sd_full)),
    pi0 = pi0,
    mean_loglik = mean_loglik,
    logL = logL
  )
}

# Unified distillation core:
# - null component is fixed via logL_null
# - only non-null slabs are merged/distilled.
distill_grid_path_fixed_null_from_bhat_s <- function(
    bhat, s,
    logL_null,
    init_slab_grid,
    lambda0 = 10,
    mixsqp_control = NULL,
    verbose = FALSE
) {
  x <- as.numeric(bhat)
  s <- as.numeric(s)
  keep <- is.finite(x) & is.finite(s) & (s > 0) & is.finite(logL_null)
  x <- x[keep]
  s <- s[keep]
  logL_null <- as.numeric(logL_null[keep])

  mll_drop_total_tol <- 2e-4
  min_grid_len <- 2L
  rel_thresh <- 1e-6
  tol <- 1e-12

  is_close <- function(a, b) abs(a - b) <= tol * max(1, abs(a), abs(b))

  cur_grid <- sort(unique(as.numeric(init_slab_grid)))
  cur_grid <- cur_grid[is.finite(cur_grid) & cur_grid > 0]
  if (!length(cur_grid)) stop("init_slab_grid has no positive slab.")

  cur_logL_pos <- loglik_normalmix_cpp(x, s, cur_grid)
  cur_fr <- .fit_with_logL_fast(
    logL = cbind(logL_null, cur_logL_pos),
    sd_full = c(0, cur_grid),
    lambda0 = lambda0,
    mixsqp_control = mixsqp_control,
    pi_init = NULL
  )

  results <- list()
  step_id <- 1L

  record_step <- function(fr, grid_now, sid) {
    pi_all <- as.numeric(fr$fit$fitted_g$pi)
    sd_all <- as.numeric(fr$fit$fitted_g$sd)
    pi0 <- fr$pi0

    idx_pos <- which(sd_all > 0)
    keep_pos <- integer(0)
    if (length(idx_pos) > 0L) {
      rel_pos <- rep(0, length(idx_pos))
      if ((1 - pi0) > 0) {
        rel_pos <- pi_all[idx_pos] / (1 - pi0)
      }
      keep_pos <- idx_pos[which(rel_pos > rel_thresh)]

      # Slabs with negligible weight are dropped even if fewer than min_grid_len remain;
      # padding with a zero-weight slab only added dead allocation states. Keep the
      # heaviest slab when none passes the threshold.
      if (!length(keep_pos)) {
        keep_pos <- idx_pos[which.max(pi_all[idx_pos])]
      }
    }

    keep_idx <- c(which(sd_all == 0), keep_pos)

    list(
      step = sid,
      grid_size = length(grid_now),
      remain_ratio = 1,
      mean_loglik = fr$mean_loglik,
      pi0 = pi0,
      grid = grid_now,
      slab_grid = sd_all[keep_idx],
      pi = pi_all[keep_idx]
    )
  }

  results[[as.character(step_id)]] <- record_step(cur_fr, cur_grid, step_id)

  repeat {
    pi_all_cur <- as.numeric(cur_fr$fit$fitted_g$pi)
    sd_all_cur <- as.numeric(cur_fr$fit$fitted_g$sd)
    idx_pos <- which(sd_all_cur > 0)
    if (length(idx_pos) <= min_grid_len) break

    sd_pos <- sd_all_cur[idx_pos]
    pi_pos <- pi_all_cur[idx_pos]

    # align cached logL_pos columns to current sd order
    idx_align <- vapply(sd_pos, function(v) {
      jj <- which(vapply(cur_grid, function(g) is_close(g, v), logical(1)))
      if (length(jj)) jj[[1]] else NA_integer_
    }, integer(1))
    if (anyNA(idx_align)) {
      cur_logL_pos <- loglik_normalmix_cpp(x, s, sd_pos)
      cur_grid <- sd_pos
    } else {
      cur_logL_pos <- cur_logL_pos[, idx_align, drop = FALSE]
      cur_grid <- sd_pos
    }

    v <- sd_pos^2
    approx_cost <- vapply(
      seq_len(length(sd_pos) - 1L),
      function(k) {
        cst <- .sym_kl_prior(v[k], v[k + 1L])
        cst * (pi_pos[k] + pi_pos[k + 1L] + 1e-16)
      },
      numeric(1)
    )
    ord <- order(approx_cost)

    accepted <- FALSE
    for (ii in ord) {
      merged_sd <- merge_two_grids(sd_pos[ii], sd_pos[ii + 1L], pi_pos[ii], pi_pos[ii + 1L])

      # construct new positive grid
      new_sd <- numeric(0)
      i <- 1L
      while (i <= length(sd_pos)) {
        if (i == ii && i < length(sd_pos)) {
          new_sd <- c(new_sd, merged_sd)
          i <- i + 2L
        } else {
          new_sd <- c(new_sd, sd_pos[i])
          i <- i + 1L
        }
      }
      new_grid <- sort(unique(abs(new_sd)))
      new_grid <- new_grid[new_grid > 0]
      if (length(new_grid) < min_grid_len) next

      # reuse unchanged columns; only recompute merged/new columns
      n <- length(x)
      new_logL_pos <- matrix(NA_real_, nrow = n, ncol = length(new_grid))
      col_merged <- as.numeric(loglik_normalmix_cpp(x, s, merged_sd))

      for (kk in seq_along(new_grid)) {
        g <- new_grid[kk]
        if (is_close(g, merged_sd)) {
          new_logL_pos[, kk] <- col_merged
          next
        }

        old_match <- which(vapply(sd_pos, function(vv) is_close(vv, g), logical(1)))
        old_match <- setdiff(old_match, c(ii, ii + 1L))
        if (length(old_match) >= 1L) {
          new_logL_pos[, kk] <- cur_logL_pos[, old_match[[1]], drop = TRUE]
        } else {
          new_logL_pos[, kk] <- as.numeric(loglik_normalmix_cpp(x, s, g))
        }
      }

      # warm-start pi on new grid
      pi_pos_init <- numeric(length(new_grid))
      for (jj in seq_along(sd_pos)) {
        g_from <- sd_pos[jj]
        g_to <- if (jj %in% c(ii, ii + 1L)) merged_sd else g_from
        kk <- which(vapply(new_grid, function(vv) is_close(vv, g_to), logical(1)))
        if (length(kk)) pi_pos_init[kk[[1]]] <- pi_pos_init[kk[[1]]] + pi_pos[jj]
      }
      pi_init <- c(cur_fr$pi0, pi_pos_init)
      spi <- sum(pi_init)
      pi_init <- if (is.finite(spi) && spi > 0) pi_init / spi else rep(1 / length(pi_init), length(pi_init))

      new_fr <- .fit_with_logL_fast(
        logL = cbind(logL_null, new_logL_pos),
        sd_full = c(0, new_grid),
        lambda0 = lambda0,
        mixsqp_control = mixsqp_control,
        pi_init = pi_init
      )

      delta_mll <- new_fr$mean_loglik - cur_fr$mean_loglik
      if (delta_mll >= -abs(mll_drop_total_tol)) {
        cur_grid <- new_grid
        cur_logL_pos <- new_logL_pos
        cur_fr <- new_fr
        accepted <- TRUE
        break
      }
    }

    if (!accepted) break

    step_id <- step_id + 1L
    results[[as.character(step_id)]] <- record_step(cur_fr, cur_grid, step_id)
    if (isTRUE(verbose)) {
      cat(sprintf("[fixed-null-distill] step=%d grid=%d mll=%.6f\n", step_id, length(cur_grid), cur_fr$mean_loglik))
    }
  }

  tab <- do.call(
    rbind,
    lapply(results, function(r) {
      data.frame(
        step = r$step,
        grid_size = r$grid_size,
        remain = sprintf("%.1f%%", 100 * r$remain_ratio),
        mean_loglik = r$mean_loglik,
        pi0 = r$pi0,
        stringsAsFactors = FALSE
      )
    })
  )

  list(
    table = tab,
    steps = results,
    fine_grid = init_slab_grid,
    x = x,
    s = s
  )
}

# ASHR uses the same fixed-null distillation core with exact null.
distill_grid_path_from_bhat_s <- function(
    bhat, s,
    lambda0 = NULL,
    fine_mult = NULL,
    fine_s_min_mult = NULL,
    fine_s_max_mult = NULL,
    fine_max_len = NULL,
    mixsqp_control = NULL,
    verbose = TRUE
) {
  hp <- .apch_marginal_hyperparams()
  if (is.null(lambda0)) lambda0 <- hp$lambda0
  if (is.null(fine_mult)) fine_mult <- hp$fine_mult
  if (is.null(fine_s_min_mult)) fine_s_min_mult <- hp$fine_s_min_mult
  if (is.null(fine_s_max_mult)) fine_s_max_mult <- hp$fine_s_max_mult
  if (is.null(fine_max_len)) fine_max_len <- hp$fine_max_len
  x <- as.numeric(bhat)
  keep <- is.finite(x) & is.finite(s) & s > 0
  x <- x[keep]
  s <- as.numeric(s[keep])

  grid0 <- make_mixsd_grid(
    x = x,
    s = s,
    mult = fine_mult,
    s_min_mult = fine_s_min_mult,
    s_max_mult = fine_s_max_mult,
    max_len = fine_max_len
  )

  logL_null <- as.numeric(loglik_normalmix_cpp(x, s, 0))
  distill_grid_path_fixed_null_from_bhat_s(
    bhat = x,
    s = s,
    logL_null = logL_null,
    init_slab_grid = grid0,
    lambda0 = lambda0,
    mixsqp_control = mixsqp_control,
    verbose = verbose
  )
}
