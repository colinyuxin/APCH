##ashr_grid_distillation
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

# generate grid for ashr fitting
make_mixsd_grid <- function(x, s, mult = 1.35, s_min_mult = 2, s_max_mult = 1.0, max_len = 500) {
  s <- s[is.finite(s) & s > 0]; x <- x[is.finite(x)]
  if (!length(x) || !length(s)) stop("no valid x/s")
  smin <- median(s) * s_min_mult
  base <- sqrt(max(pmax(0, x^2 - s^2)))
  smax0 <- if (is.finite(base) && base > 0) base else 8 * min(s)
  smax <- max(smin, smax0 * s_max_mult)
  g <- c(); cur <- smin
  while (cur <= smax && length(g) < max_len) { g <- c(g, cur); cur <- cur * mult }
  unique(sort(g))
}

cdf_normal_vec <- function(x, s, pi, sd) {
  stopifnot(is.numeric(x), is.numeric(s), is.numeric(pi), is.numeric(sd))
  denom <- sqrt(outer(s^2, sd^2, "+"))
  z <- sweep(denom, 1, x, "/")
  as.vector(pnorm(z) %*% pi)
}

cdf_uniform_vec <- function(x, s, pi, a, b) {
  out <- numeric(length(x))
  for (k in seq_along(pi)) {
    if (pi[k] == 0) next
    if (a[k] == 0 && b[k] == 0) {
      out <- out + pi[k] * pnorm(x / s)
    } else {
      ta <- (x - a[k]) / s
      tb <- (x - b[k]) / s
      term <- ta * pnorm(ta) + dnorm(ta) - (tb * pnorm(tb) + dnorm(tb))
      out <- out + pi[k] * (s / (b[k] - a[k])) * term
    }
  }
  out
}

cdf_halfnormal_vec <- function(x, s, pi, sd, a, b) {
  out <- numeric(length(x))
  for (k in seq_along(pi)) {
    if (pi[k] == 0) next
    denom <- sqrt(s^2 + sd[k]^2)
    z <- x / denom
    alpha <- sd[k] / s
    Tval <- vapply(seq_along(z), function(i) sn::T.Owen(z[i], alpha[i]), numeric(1))
    Phi_z <- pnorm(z) - 2 * Tval
    if (identical(a[k], 0) && is.infinite(b[k]) && b[k] > 0) {
      out <- out + pi[k] * Phi_z
    } else if (is.infinite(a[k]) && a[k] < 0 && identical(b[k], 0)) {
      out <- out + pi[k] * (1 - Phi_z)
    }
  }
  out
}

# ashr fitting with grid
.fit_with_grid <- function(x, s, grid, mixcompdist = "normal", pointmass = TRUE, lambda0 = 10, mixsqp_control = NULL) {
  keep <- is.finite(x) & is.finite(s) & s > 0
  x <- x[keep]; s <- s[keep]
  if (!length(x)) stop("no valid observations")
  grid <- as.numeric(grid); grid <- grid[is.finite(grid) & grid > 0]; grid <- sort(unique(grid))
  if (!length(grid)) stop("grid has no valid positive sd")
  sd_floor <- max(min(s) * 1e-8, .Machine$double.eps)
  grid[grid < sd_floor] <- sd_floor

  ctl_allowed <- c("eps", "maxiter", "verbose")
  ctl <- if (is.list(mixsqp_control)) mixsqp_control[names(mixsqp_control) %in% ctl_allowed] else NULL

  args <- list(
    betahat = x,
    sebetahat = s,
    mixcompdist = mixcompdist,
    mode = 0,
    mixsd = grid,
    pointmass = pointmass,
    prior = "nullbiased",
    nullweight = lambda0
  )
  if (length(ctl)) args$control <- ctl
  fit <- do.call(ashr::ash, args)

  g <- fit$fitted_g; pi <- as.numeric(g$pi); sd <- as.numeric(g$sd)
  a <- as.numeric(g$a); b <- as.numeric(g$b)

  if (any(!is.finite(pi)) || any(!is.finite(sd))) stop("ashr returned non-finite mixture parameters")
  if (mixcompdist == "normal") {
    pi0 <- if (any(sd == 0)) sum(pi[sd == 0]) else 0
    u <- cdf_normal_vec(x, s, pi, sd)
  } else if (mixcompdist == "halfnormal") {
    pi0 <- if (any(sd == 0)) sum(pi[sd == 0]) else 0
    u <- cdf_halfnormal_vec(x, s, pi, sd, a, b)
  } else if (mixcompdist %in% c("uniform", "halfuniform", "+uniform", "-uniform")) {
    pi0 <- if (any(a == 0 & b == 0)) sum(pi[a == 0 & b == 0]) else 0
    u <- cdf_uniform_vec(x, s, pi, a, b)
  } else {
    stop("Unsupported dist")
  }

  ks <- suppressWarnings(stats::ks.test(u, "punif"))
  list(
    fit = fit,
    grid = grid,
    x = x,
    s = s,
    pit_ks = as.numeric(ks$statistic),
    pit_ks_p = as.numeric(ks$p.value),
    mean_loglik = ashr::get_loglik(fit) / length(x),
    pi0 = pi0
  )
}

.sym_kl_prior <- function(v1, v2, m1 = 0, m2 = 0) {
  kl12 <- (v1/v2) + (m2-m1)^2/v2 - 1 + log(v2/v1)
  kl21 <- (v2/v1) + (m1-m2)^2/v1 - 1 + log(v1/v2)
  0.5 * (kl12 + kl21)
}

merge_two_grids <- function(mixcompdist = "normal", s1, s2, w1, w2) {
  if (w1 + w2 > 1e-12) {
    if (mixcompdist %in% c("normal", "halfnormal")) {
      v_new <- (w1*s1^2 + w2*s2^2)/(w1+w2)
      return(sign(s1) * sqrt(max(v_new, .Machine$double.eps)))
    } else if (mixcompdist == "uniform") {
      v1 <- s1^2/3; v2 <- s2^2/3
      v_new <- (w1*v1 + w2*v2)/(w1+w2)
      return(sqrt(max(3*v_new, .Machine$double.eps)))
    } else if (mixcompdist %in% c("halfuniform", "+uniform", "-uniform")) {
      m1 <- s1/2; v1 <- s1^2/12
      m2 <- s2/2; v2 <- s2^2/12
      m_new <- (w1*m1 + w2*m2)/(w1+w2)
      q_new <- (w1*(v1+m1^2) + w2*(v2+m2^2))/(w1+w2)
      v_new <- q_new - m_new^2
      return(sign(s1) * sqrt(max(12*v_new, .Machine$double.eps)))
    } else {
      stop("Unsupported dist")
    }
  } else {
    return(sign(s1) * sqrt(max(s1*s2, .Machine$double.eps)))
  }
}

get_sd_from_fit <- function(fitted_g, mixcompdist = "normal") {
  if (mixcompdist == "normal") {
    sd <- as.numeric(fitted_g$sd)
  } else if (mixcompdist == "halfnormal") {
    a <- as.numeric(fitted_g$a); b <- as.numeric(fitted_g$b)
    sd <- as.numeric(fitted_g$sd)
    sd[b == 0] <- -sd[b]
  } else if (mixcompdist == "uniform") {
    sd <- as.numeric(fitted_g$b)
  } else if (mixcompdist %in% c("halfuniform", "+uniform", "-uniform")) {
    a <- as.numeric(fitted_g$a); b <- as.numeric(fitted_g$b)
    sd <- b
    sd[a != 0] <- a[a != 0]
  }
  sd
}

distill_grid_path_from_xs <- function(
    x, s, mixcompdist = "normal", lambda0 = 10,
    fine_mult = 1.35, fine_s_min_mult = 0.5, fine_s_max_mult = 1.0,
    fine_max_len = 500, mixsqp_control = NULL,
    mll_drop_total_tol = 2e-4, ks_increase_total_tol = 1e-2,
    min_grid_len = 8, max_merges = Inf, pair_frac = 0.20,
    rel_thresh = 1e-6, verbose = TRUE, distill = TRUE
) {
  grid0 <- make_mixsd_grid(
    x, s,
    mult = fine_mult,
    s_min_mult = fine_s_min_mult,
    s_max_mult = fine_s_max_mult,
    max_len = fine_max_len
  )
  K_fine <- length(grid0)
  if (K_fine < 2L) stop("fine grid too short to merge")

  fr0 <- .fit_with_grid(
    x, s, grid0,
    mixcompdist = mixcompdist,
    pointmass = TRUE,
    lambda0 = lambda0,
    mixsqp_control = mixsqp_control
  )

  g_init <- fr0$fit$fitted_g
  pi_init_all <- as.numeric(g_init$pi)
  sd_init_all <- get_sd_from_fit(g_init, mixcompdist)

  pi0_init <- if (any(sd_init_all == 0)) sum(pi_init_all[sd_init_all == 0]) else 0
  denom_init <- (1 - pi0_init)
  rel_init <- rep(Inf, length(pi_init_all))
  if (denom_init > 0) rel_init[sd_init_all != 0] <- pi_init_all[sd_init_all != 0] / denom_init

  to_remain_idx <- which(rel_init > rel_thresh)
  sd_to_remain <- numeric(0)
  pi <- pi_init_all
  if (length(to_remain_idx) > 0) {
    sd_to_remain <- sd_init_all[to_remain_idx]
    pi <- pi_init_all[to_remain_idx]
    remain_pos <- sapply(abs(sd_to_remain), function(v) which.min(abs(grid0 - v)))
    remain_pos <- unique(as.integer(remain_pos))
    grid_pruned <- grid0[remain_pos]
  } else {
    grid_pruned <- grid0
  }
  K_pruned <- length(grid_pruned)

  cur_fr <- .fit_with_grid(
    x, s, grid_pruned,
    mixcompdist = mixcompdist,
    pointmass = TRUE,
    lambda0 = lambda0,
    mixsqp_control = mixsqp_control
  )

  if (!isTRUE(distill)) {
    if (verbose) cat(sprintf("[Grid] prune: %d -> %d\n", K_fine, K_pruned))
    tab <- data.frame(
      step = 1L,
      grid_size = length(cur_fr$grid),
      remain = sprintf("%.1f%%", 100 * length(cur_fr$grid) / K_fine),
      pit_ks = cur_fr$pit_ks,
      pit_ks_p = cur_fr$pit_ks_p,
      mean_loglik = cur_fr$mean_loglik,
      pi0 = cur_fr$pi0,
      delta_ks = 0,
      delta_mll = 0
    )
    out <- list(
      table = tab,
      steps = list(`1` = list(
        step = 1L,
        grid_size = length(cur_fr$grid),
        remain_ratio = length(cur_fr$grid)/K_fine,
        pit_ks = cur_fr$pit_ks,
        pit_ks_p = cur_fr$pit_ks_p,
        mean_loglik = cur_fr$mean_loglik,
        pi0 = cur_fr$pi0,
        grid = cur_fr$grid,
        signed_grid = sd_to_remain,
        pi = pi
      )),
      fine_grid = grid_pruned,
      x = cur_fr$x,
      s = cur_fr$s
    )
    return(invisible(out))
  }

  results <- list()
  results[["1"]] <- list(
    step = 1L,
    grid_size = length(cur_fr$grid),
    remain_ratio = length(cur_fr$grid)/K_fine,
    pit_ks = cur_fr$pit_ks,
    pit_ks_p = cur_fr$pit_ks_p,
    mean_loglik = cur_fr$mean_loglik,
    pi0 = cur_fr$pi0,
    grid = cur_fr$grid,
    signed_grid = sd_to_remain,
    pi = pi
  )

  ref_mll <- cur_fr$mean_loglik
  ref_ks <- cur_fr$pit_ks
  cur_fit <- cur_fr$fit
  cur_grid <- cur_fr$grid

  g_init_pos_count <- sum(sd_init_all != 0)
  K0 <- g_init_pos_count
  base_pairs_target <- max(1L, floor(pair_frac * K0))

  step_id <- 2L
  batches_done <- 0L
  repeat {
    g_cur <- cur_fit$fitted_g
    pi_all_cur <- as.numeric(g_cur$pi)
    sd_all_cur <- get_sd_from_fit(g_cur, mixcompdist)

    idx_pos_cur <- which(sd_all_cur != 0)
    if (length(idx_pos_cur) == 0L) break
    sd_pos <- sd_all_cur[idx_pos_cur]
    pi_pos <- pi_all_cur[idx_pos_cur]

    sd_init_pos_idx <- which(sd_init_all != 0)
    map_to_init <- vapply(sd_pos, function(v) {
      diffs <- abs(sd_init_all[sd_init_pos_idx] - v)
      which.min(diffs)
    }, integer(1))
    mapped_init_idx <- sd_init_pos_idx[map_to_init]
    rel_for_sd_pos <- rel_init[mapped_init_idx]
    keep_mask <- (rel_for_sd_pos >= rel_thresh)

    K_decision <- sum(keep_mask)
    min_allowed <- max(1L, as.integer(min_grid_len))
    if (K_decision <= min_allowed) break
    if (batches_done >= max_merges) break
    if (K_decision < 2) break

    sd_for_decision <- sd_pos[keep_mask]
    pi_for_decision <- pi_pos[keep_mask]

    v <- sd_for_decision^2
    if (mixcompdist == "normal") {
      approx_cost <- vapply(
        1:(length(sd_for_decision) - 1L),
        function(k) {
          cost <- .sym_kl_prior(v[k], v[k+1L])
          cost * (pi_for_decision[k] + pi_for_decision[k+1L] + 1e-16)
        },
        numeric(1)
      )

    } else if (mixcompdist == "uniform") {
      approx_cost <- vapply(
        1:(length(sd_for_decision) - 1L),
        function(k) {
          cost <- .sym_kl_prior(v[k] / 3, v[k+1L] / 3)
          cost * (pi_for_decision[k] + pi_for_decision[k+1L] + 1e-16)
        },
        numeric(1)
      )

    } else if (mixcompdist == "halfnormal") {
      sign_vec <- ifelse(sd_for_decision >= 0, 1, -1)
      Kp <- length(sd_for_decision)
      approx_cost <- rep(Inf, max(0, Kp - 1L))
      same_sign <- (sign_vec[-Kp] == sign_vec[-1L])
      idx <- which(same_sign)
      if (length(idx)) {
        approx_cost[idx] <- vapply(
          idx,
          function(k) {
            cost <- .sym_kl_prior(v[k], v[k+1L])
            cost * (pi_for_decision[k] + pi_for_decision[k+1L] + 1e-16)
          },
          numeric(1)
        )
      }

    } else if (mixcompdist %in% c("halfuniform", "+uniform", "-uniform")) {
      sign_vec <- ifelse(sd_for_decision >= 0, 1, -1)
      Kp <- length(sd_for_decision)
      approx_cost <- rep(Inf, max(0, Kp - 1L))
      m <- sd_for_decision / 2
      var <- v / 12
      same_sign <- (sign_vec[-Kp] == sign_vec[-1L])
      idx <- which(same_sign)
      if (length(idx)) {
        approx_cost[idx] <- vapply(
          idx,
          function(k) {
            cost <- .sym_kl_prior(var[k], var[k+1L], m[k], m[k+1L])
            cost * (pi_for_decision[k] + pi_for_decision[k+1L] + 1e-16)
          },
          numeric(1)
        )
      }

    } else {
      stop("Unsupported dist")
    }

    ord <- order(approx_cost)
    max_pairs_this_round <- min(
      base_pairs_this_round <- base_pairs_target,
      (length(sd_for_decision) - min_allowed)
    )
    if (max_pairs_this_round < 1L) break

    select_pairs <- function(n_pairs) {
      used <- rep(FALSE, length(sd_for_decision))
      picks <- integer(0)
      for (idx in ord) {
        k <- idx
        if (!used[k] && !used[k+1L]) {
          picks <- c(picks, k)
          used[k] <- used[k+1L] <- TRUE
          if (length(picks) >= n_pairs) break
        }
      }
      sort(picks)
    }

    decision_positions <- which(keep_mask)
    build_merged_grid <- function(pairs_in_decision_idx) {
      pair_flags_pos <- rep(FALSE, length(sd_pos) - 1L)
      for (pp in pairs_in_decision_idx) {
        pos_in_sdpos <- decision_positions[pp]
        if (pos_in_sdpos <= length(pair_flags_pos)) pair_flags_pos[pos_in_sdpos] <- TRUE
      }
      new_sd <- c()
      i <- 1L
      details <- list()
      Kpos <- length(sd_pos)
      while (i <= Kpos) {
        if (i <= Kpos - 1L && pair_flags_pos[i]) {
          wk1 <- pi_pos[i]; wk2 <- pi_pos[i+1L]
          sd_new <- merge_two_grids(mixcompdist, sd_pos[i], sd_pos[i+1L], wk1, wk2)
          new_sd <- c(new_sd, sd_new)
          details[[length(details) + 1]] <- list(
            pair = c(i, i+1L),
            sd_old = c(sd_pos[i], sd_pos[i+1L]),
            sd_new = sd_new,
            w_old = c(wk1, wk2)
          )
          i <- i + 2L
        } else {
          new_sd <- c(new_sd, sd_pos[i])
          i <- i + 1L
        }
      }
      list(sd = new_sd, details = details)
    }

    n_try <- max_pairs_this_round
    accepted <- FALSE
    chosen_details <- NULL
    fr_prop <- NULL
    while (n_try >= 1L) {
      chosen_pairs <- select_pairs(n_try)
      if (length(chosen_pairs) == 0L) {
        n_try <- floor(n_try / 2)
        next
      }
      merged <- build_merged_grid(chosen_pairs)
      merged_grid <- sort(abs(merged$sd))
      keepu <- c(TRUE, abs(diff(merged_grid)) > 1e-6)
      merged_grid <- merged_grid[keepu]
      fr_prop <- tryCatch(
        .fit_with_grid(
          x, s, merged_grid,
          mixcompdist = mixcompdist,
          pointmass = TRUE,
          lambda0 = lambda0,
          mixsqp_control = mixsqp_control
        ),
        error = function(e) NULL
      )
      if (is.null(fr_prop)) {
        n_try <- floor(n_try / 2)
        next
      }

      delta_mll_total <- fr_prop$mean_loglik - ref_mll
      delta_ks_total <- fr_prop$pit_ks - ref_ks
      pass_total <- (delta_mll_total >= -abs(mll_drop_total_tol)) &&
        (delta_ks_total <= ks_increase_total_tol)
      if (pass_total) {
        accepted <- TRUE
        chosen_details <- merged$details
        break
      } else {
        n_try <- if (n_try == 1L) 0L else floor(n_try / 2)
      }
    }

    if (!accepted) break

    cur_fit <- fr_prop$fit
    cur_grid <- fr_prop$grid

    pi0 <- fr_prop$pi0
    pi_all <- as.numeric(cur_fit$fitted_g$pi)
    sd_all <- get_sd_from_fit(cur_fit$fitted_g, mixcompdist)
    rel <- rep(Inf, length(pi_all))
    rel[sd_all != 0] <- pi_all[sd_all != 0] / (1 - pi0)
    to_remain_idx <- which(rel > rel_thresh)
    signed_grid <- sd_all[to_remain_idx]
    pi <- pi_all[to_remain_idx]

    batches_done <- batches_done + 1L
    results[[as.character(step_id)]] <- list(
      step = as.integer(step_id),
      grid_size = length(cur_grid),
      remain_ratio = length(cur_grid) / K_fine,
      pit_ks = fr_prop$pit_ks,
      pit_ks_p = fr_prop$pit_ks_p,
      mean_loglik = fr_prop$mean_loglik,
      pi0 = fr_prop$pi0,
      grid = cur_grid,
      signed_grid = signed_grid,
      pi = pi
    )
    step_id <- step_id + 1L
  }

  tab <- do.call(
    rbind,
    lapply(
      results,
      function(r) {
        data.frame(
          step = r$step,
          grid_size = r$grid_size,
          remain = sprintf("%.1f%%", 100 * r$remain_ratio),
          pit_ks = r$pit_ks,
          pit_ks_p = r$pit_ks_p,
          mean_loglik = r$mean_loglik,
          pi0 = r$pi0
        )
      }
    )
  )
  rownames(tab) <- NULL
  tab <- tab[order(tab$step), ]
  ref <- results[["1"]]
  if (!is.null(ref)) {
    tab$delta_ks <- tab$pit_ks - ref$pit_ks
    tab$delta_mll <- tab$mean_loglik - ref$mean_loglik
  }

  if (verbose) {
    K_final <- tail(tab$grid_size, 1)
    cat(sprintf("[Grid] prune: %d -> %d\n", K_fine, K_pruned))
    cat(sprintf("[Grid] distill: %d -> %d\n", K_pruned, K_final))
  }

  invisible(list(
    table = tab,
    steps = results,
    fine_grid = grid_pruned,
    x = x,
    s = s
  ))
}

#' Distill an ashr mixture grid for a single feature
#'
#' This helper function runs the internal grid-construction and distillation
#' pipeline for a single feature. It is mainly intended for inspecting and
#' tuning the ashr prior used in the APCH feature layer (e.g., in meta-analysis
#' mode), without running the full APCH pipeline.
#'
#' Given effect estimates \code{x} and standard errors \code{s}, the function
#' builds a fine-scale grid of mixture standard deviations, fits \pkg{ashr}
#' on that grid, optionally prunes and distills the grid, and returns the
#' final signed grid and mixture weights, along with diagnostic information
#' about the distillation path.
#'
#' @param x Numeric vector of effect estimates for a single feature.
#' @param s Numeric vector of standard errors corresponding to \code{x}.
#' @param mixcompdist Character specifying the slab distribution passed to
#'   \code{\link[ashr]{ash}}. Typical values include \code{"normal"},
#'   \code{"uniform"}, \code{"halfuniform"}, and \code{"halfnormal"}.
#'   Default: \code{"normal"}.
#' @param lambda0 Numeric null-biasing parameter passed as \code{nullweight}
#'   to \code{\link[ashr]{ash}}. Larger values encourage more mass on the
#'   point mass at zero. Default: \code{10}.
#' @param distill Logical; if \code{TRUE} (default), perform the full
#'   distillation path (successive grid merges). If \code{FALSE}, only run
#'   the initial pruning step (fine grid \eqn{\to} pruned grid).
#' @param fine_mult,fine_s_min_mult,fine_s_max_mult,fine_max_len Numeric
#'   parameters controlling the initial fine grid (see internal
#'   \code{distill_grid_path_from_xs()} and \code{make_mixsd_grid()}).
#' @param mixsqp_control Optional list of control parameters passed to
#'   \code{ashr::ash()} (through its \code{control} argument for \code{mixsqp}),
#'   typically including \code{"eps"}, \code{"maxiter"}, and \code{"verbose"}.
#' @param mll_drop_total_tol,ks_increase_total_tol Numeric tolerances used
#'   to decide whether a proposed grid merge is acceptable, based on the
#'   change in mean log-likelihood and PIT KS statistic.
#' @param min_grid_len Integer minimum allowed positive grid size during
#'   distillation.
#' @param max_merges Maximum number of merge batches.
#' @param pair_frac Fraction of candidate adjacent grid pairs allowed to merge
#'   at each batch.
#' @param rel_thresh Relative weight threshold used to prune very low-mass
#'   grid components.
#' @param verbose Logical; if \code{TRUE}, print progress messages from
#'   the grid construction and distillation.
#'
#' @return A list with components:
#' \describe{
#'   \item{grid_final}{Numeric vector of the final signed grid values
#'         (including sign for half-normal / half-uniform families).}
#'   \item{pi_final}{Numeric vector of mixture weights aligned with
#'         \code{grid_final}. The weights are renormalized to sum to 1
#'         over the components returned here.}
#'   \item{path_table}{Data frame summarizing the distillation path
#'         (step index, grid size, PIT KS, mean log-likelihood, etc.).}
#'   \item{fine_grid}{Numeric vector of the initial pruned fine grid.}
#'   \item{steps}{Full list of steps returned by
#'         \code{distill_grid_path_from_xs()}, each containing the grid,
#'         \code{signed_grid}, and \code{pi} at that step.}
#' }
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 2000
#' x <- rnorm(n)
#' s <- rep(1, n)
#'
#' # Distill a normal-slab ashr grid
#' gd <- ashr_grid_distill(x, s, mixcompdist = "normal", lambda0 = 10)
#' gd$grid_final
#' gd$pi_final
#'
#' # Inspect the distillation path
#' gd$path_table
#' }
#'
#' @export
ashr_grid_distill <- function(
    x, s,
    mixcompdist = "normal",
    lambda0 = 10,
    distill = TRUE,
    fine_mult = 1.35,
    fine_s_min_mult = 0.5,
    fine_s_max_mult = 1.0,
    fine_max_len = 500,
    mixsqp_control = NULL,
    mll_drop_total_tol = 2e-4,
    ks_increase_total_tol = 1e-2,
    min_grid_len = 8,
    max_merges = Inf,
    pair_frac = 0.20,
    rel_thresh = 1e-6,
    verbose = TRUE
) {
  stopifnot(length(x) == length(s))

  res <- distill_grid_path_from_xs(
    x = x,
    s = s,
    mixcompdist = mixcompdist,
    lambda0 = lambda0,
    fine_mult = fine_mult,
    fine_s_min_mult = fine_s_min_mult,
    fine_s_max_mult = fine_s_max_mult,
    fine_max_len = fine_max_len,
    mixsqp_control = mixsqp_control,
    mll_drop_total_tol = mll_drop_total_tol,
    ks_increase_total_tol = ks_increase_total_tol,
    min_grid_len = min_grid_len,
    max_merges = max_merges,
    pair_frac = pair_frac,
    rel_thresh = rel_thresh,
    verbose = verbose,
    distill = distill
  )

  last_step <- as.character(max(res$table$step))
  step_res  <- res$steps[[last_step]]

  grid_final <- step_res$signed_grid
  pi_final   <- step_res$pi

  if (!is.null(pi_final)) {
    ssum <- sum(pi_final)
    if (is.finite(ssum) && ssum > 0 && abs(ssum - 1) > 1e-8) {
      pi_final <- pi_final / ssum
    }
  }

  list(
    grid_final = grid_final,
    pi_final   = pi_final,
    path_table = res$table,
    fine_grid  = res$fine_grid,
    steps      = res$steps
  )
}

# multi-feature interface (supports parallelization + distill selection)
.resolve_distill <- function(distill, R_cor, feature_names, cor_thresh) {
  p <- length(feature_names)
  if (is.character(distill) && length(distill) == 1L && identical(tolower(distill), "auto")) {
    sel <- logical(p)
    for (j in seq_len(p)) {
      others <- setdiff(seq_len(p), j)
      sel[j] <- any(abs(R_cor[j, others]) > cor_thresh, na.rm = TRUE)
    }
    return(sel)
  }
  if (is.logical(distill) && length(distill) == 1L) return(rep(distill, p))
  if (is.logical(distill) && length(distill) == p)  return(distill)
  if (is.numeric(distill)) {
    ix <- as.integer(distill); stopifnot(all(ix >= 1 & ix <= p))
    sel <- rep(FALSE, p); sel[ix] <- TRUE; return(sel)
  }
  if (is.character(distill)) {
    m <- match(distill, feature_names); stopifnot(all(!is.na(m)))
    sel <- rep(FALSE, p); sel[m] <- TRUE; return(sel)
  }
  stop("distill: must be 'auto'/logical/index/name vector")
}

