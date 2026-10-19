## helper.R

# enumerate all 2^p configurations and return a 2^p x p matrix, where each row corresponds to one configuration
generate_all_configs <- function(p) {
  configs_df <- expand.grid(replicate(p, c(0,1), simplify = FALSE))
  configs_mat <- as.matrix(configs_df)
  weight_vec <- rowSums(configs_mat)
  nums <- apply(configs_mat, 1, function(r) sum(r * 2^((p-1):0)))
  # sorting: first sort by weight in ascending order, then sort by value in descending order within the same weight
  configs_sorted <- configs_mat[order(weight_vec, -nums), , drop = FALSE]
  return(configs_sorted)
}

generate_all_subsets <- function(p) {
  cfg <- generate_all_configs(p)
  cfg[rowSums(cfg) > 0, , drop = FALSE]
}

.apch_effect_names <- function(x, n = nrow(x)) {
  nm <- rownames(x)
  if (is.null(nm)) paste0("e", seq_len(n)) else nm
}

.apch_feature_names <- function(x, p = ncol(x)) {
  nm <- colnames(x)
  if (is.null(nm)) paste0("F", seq_len(p)) else nm
}

# helper: apply user-facing ids/names and propagate them.
.apch_fill_dimnames <- function(Bhat, S, R_cor, effect_id = NULL, feature_names = NULL) {
  if (!is.null(effect_id)) rownames(Bhat) <- as.character(effect_id)
  if (!is.null(feature_names)) colnames(Bhat) <- as.character(feature_names)

  dimnames(S) <- dimnames(Bhat)

  if (!is.null(colnames(Bhat)) &&
      (is.null(rownames(R_cor)) || is.null(colnames(R_cor)))) {
    dimnames(R_cor) <- list(colnames(Bhat), colnames(Bhat))
  }

  list(Bhat = Bhat, S = S, R_cor = R_cor)
}


# helper: internal scale unification to z-score scale.
# All joint fitting assumes z-scale; S is set to 1 after scaling.
.apch_scale_to_zscore <- function(Bhat, S) {
  bad_se <- is.finite(Bhat) & (!is.finite(S) | S <= 0)
  if (any(bad_se)) {
    stop("S must be finite and > 0 wherever Bhat is finite.")
  }
  Bhat_z <- Bhat / S
  S_z <- S
  S_z[] <- 1
  list(Bhat = Bhat_z, S = S_z)
}

.apch_active_effects_from_pnull <- function(
    pnull = NULL,
    n = NULL,
    pnull_screen_thresh = 0.4
) {
  if (is.null(pnull)) {
    seq_len(n)
  } else {
    which(is.na(pnull) | (pnull < pnull_screen_thresh))
  }
}

.apch_prepare_marginal_inputs <- function(slab_list, pi_list, p) {
  slab_out <- vector("list", p)
  pi_out <- vector("list", p)
  for (j in seq_len(p)) {
    slab_j <- as.numeric(slab_list[[j]])
    pi_j <- as.numeric(pi_list[[j]])

    if (length(slab_j) == 0L) {
      slab_out[[j]] <- numeric(0)
      pi_out[[j]] <- numeric(0)
      next
    }
    pi_j <- pi_j / sum(pi_j)

    slab_out[[j]] <- slab_j
    pi_out[[j]] <- pi_j
  }
  full_grid_list <- lapply(slab_out, function(s) c(0, as.numeric(s)))
  list(
    slab_list = slab_out,
    pi_list = pi_out,
    full_grid_list = full_grid_list
  )
}

# Internal helper: normalize input C while preserving shared-vs-list structure.
# Accepts: NULL (identity), shared p x p, list of length n, or 3D array.
.apch_normalize_C <- function(Bhat, C) {
  p <- ncol(Bhat)

  if (is.null(C)) C <- diag(p)
  if (is.array(C) && length(dim(C)) == 3L) {
    C <- lapply(seq_len(dim(C)[3L]), function(i) C[, , i])
  }

  if (is.list(C)) {
    C <- .apch_regularize_C_list(Bhat, C)
    list(C_mode = "list", C_shared = NULL, C_list = C, C_ref = C[[1L]])
  } else {
    C <- as.matrix(C)
    C <- .apch_regularize_C_block(C)
    list(C_mode = "shared", C_shared = C, C_list = NULL, C_ref = C)
  }
}

.apch_regularize_C_list <- function(Bhat, C_list, min_eigen = 5e-2) {
  lapply(seq_along(C_list), function(i) {
    Ci <- as.matrix(C_list[[i]])
    obs <- which(is.finite(Bhat[i, ]))
    if (length(obs) > 0L) {
      Ci[obs, obs] <- .apch_regularize_C_block(
        Ci[obs, obs, drop = FALSE],
        min_eigen = min_eigen
      )
    }
    Ci
  })
}

.apch_regularize_C_block <- function(C, min_eigen = 5e-2) {
  C <- as.matrix(C)
  if (nrow(C) <= 1L) {
    return(C)
  }
  if (any(!is.finite(C))) {
    stop("Observed C block contains non-finite entries.")
  }

  C <- (C + t(C)) / 2
  d <- diag(C)
  if (any(d <= 0)) {
    stop("Observed C block must have positive diagonal entries.")
  }

  lambda_min <- min(eigen(C, symmetric = TRUE, only.values = TRUE)$values)
  alpha <- max(0, (min_eigen - lambda_min) / (1 - lambda_min))
  (1 - alpha) * C + alpha * diag(nrow(C))
}

# helper: subset C along retained features while preserving representation.
.apch_subset_C <- function(C_mode, C_shared, C_list, keep) {
  if (identical(C_mode, "shared")) {
    return(list(C_mode = "shared", C_shared = C_shared[keep, keep, drop = FALSE], C_list = NULL))
  }
  list(
    C_mode = "list",
    C_shared = NULL,
    C_list = lapply(C_list, function(Ci) Ci[keep, keep, drop = FALSE])
  )
}

.apch_noise_diag_vec <- function(C_mode, C_shared, C_list, j, n) {
  if (.apch_is_shared(C_mode)) {
    return(rep(C_shared[j, j], n))
  }
  vapply(C_list, function(Ci) Ci[j, j], numeric(1))
}

.apch_noise_diag_mat <- function(C_mode, C_shared, C_list, n) {
  if (.apch_is_shared(C_mode)) {
    e_diag_vec <- diag(C_shared)
    return(matrix(rep(e_diag_vec, n), nrow = n, byrow = TRUE))
  }
  do.call(rbind, lapply(C_list, diag))
}

.apch_pick_C_input <- function(C_mode, C_shared, C_list) {
  if (.apch_is_shared(C_mode)) C_shared else C_list
}

.apch_subset_effect_C_input <- function(C, effect_idx) {
  if (is.null(C)) return(NULL)
  if (is.array(C) && length(dim(C)) == 3L) return(C[, , effect_idx, drop = FALSE])
  if (is.list(C)) return(C[effect_idx])
  C
}

.apch_is_shared <- function(C_mode) {
  identical(C_mode, "shared")
}

.apch_lfsr_pmvnorm_control <- function(
    maxpts = NULL,
    abseps = NULL,
    releps = NULL,
    jitter = NULL
) {
  defaults <- list(
    maxpts = 5000L,
    abseps = 1e-4,
    releps = 1e-4,
    jitter = 1e-10
  )
  list(
    maxpts = as.integer(if (is.null(maxpts)) defaults$maxpts else maxpts),
    abseps = if (is.null(abseps)) defaults$abseps else abseps,
    releps = if (is.null(releps)) defaults$releps else releps,
    jitter = if (is.null(jitter)) defaults$jitter else jitter
  )
}

.apch_mixsqp_control <- function() {
  # Global mixsqp hyperparameters tuned for speed on large problems.
  list(
    normalize.rows = TRUE,
    tol.svd = 1e-6,
    convtol.sqp = 1e-6,
    convtol.activeset = 1e-10,
    zero.threshold.solution = 1e-8,
    zero.threshold.searchdir = 1e-14,
    suffdecr.linesearch = 0.01,
    stepsizereduce = 0.75,
    minstepsize = 1e-8,
    identity.contrib.increase = 10,
    eps = 1e-8,
    maxiter.sqp = 500,
    maxiter.activeset = NULL,
    numiter.em = 8,
    verbose = TRUE
  )
}

.apch_mixsqp_threads <- function(n, workers) {
  workers <- max(1L, as.integer(workers))
  cap <- if (n >= 50000L) 8L else 4L
  min(workers, cap)
}

.apch_with_blas_threads <- function(threads, expr) {
  if (!requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    return(force(expr))
  }

  threads <- max(1L, as.integer(threads))
  old_blas <- tryCatch(RhpcBLASctl::blas_get_num_procs(), error = function(e) 1L)

  on.exit({
    try(RhpcBLASctl::blas_set_num_threads(old_blas), silent = TRUE)
  }, add = TRUE)

  try(RhpcBLASctl::blas_set_num_threads(threads), silent = TRUE)
  force(expr)
}

.apch_mixsqp <- function(L, w = NULL, x0 = NULL, control = NULL) {
  control <- if (is.null(control)) {
    .apch_mixsqp_control()
  } else {
    modifyList(.apch_mixsqp_control(), control, keep.null = TRUE)
  }

  storage.mode(L) <- "double"
  n <- nrow(L)
  m <- ncol(L)

  w <- if (is.null(w)) rep(1, n) else as.numeric(w)
  storage.mode(w) <- "double"
  w <- w / sum(w)

  x0 <- if (is.null(x0)) rep(1 / m, m) else as.numeric(x0)
  storage.mode(x0) <- "double"
  x0 <- x0 / sum(x0)

  if (is.null(control$maxiter.activeset)) {
    control$maxiter.activeset <- min(20L, m + 1L)
  } else {
    control$maxiter.activeset <- as.integer(control$maxiter.activeset)
  }
  control$maxiter.sqp <- as.integer(control$maxiter.sqp)
  control$numiter.em <- as.integer(control$numiter.em)

  eps0 <- control$eps - min(0, min(L))
  x <- pmax(drop(apch_mixsqp_dense_cpp(
    L = L,
    w = w,
    x0 = x0,
    eps = eps0,
    numiter_em = control$numiter.em,
    convtol_sqp = control$convtol.sqp,
    convtol_activeset = control$convtol.activeset,
    zero_threshold_solution = control$zero.threshold.solution,
    zero_threshold_searchdir = control$zero.threshold.searchdir,
    suffdecr = control$suffdecr.linesearch,
    stepsize_reduce = control$stepsizereduce,
    min_step_size = control$minstepsize,
    identity_contrib_increase = control$identity.contrib.increase,
    maxiter_sqp = control$maxiter.sqp,
    maxiter_activeset = control$maxiter.activeset,
    verbose = control$verbose
  )), 0)
  x <- x / sum(x)
  list(x = x)
}

.apch_noise_adj_matrix <- function(C_mode, C_shared, C_list, p) {
  A <- matrix(0L, p, p)
  if (p <= 1L) return(A)
  if (.apch_is_shared(C_mode)) {
    nz <- abs(C_shared) > 0
    nz[row(nz) == col(nz)] <- FALSE
    A[nz] <- 1L
    return(A)
  }
  for (a in seq_len(p - 1L)) {
    for (b in (a + 1L):p) {
      hit <- FALSE
      for (i in seq_along(C_list)) {
        cab <- C_list[[i]][a, b]
        if (is.finite(cab) && abs(cab) > 0) {
          hit <- TRUE
          break
        }
      }
      if (hit) {
        A[a, b] <- 1L
        A[b, a] <- 1L
      }
    }
  }
  A
}

.apch_noise_offdiag_any <- function(C_mode, C_shared, C_list, tol = 1e-12) {
  if (.apch_is_shared(C_mode)) {
    off <- C_shared
    off[row(off) == col(off)] <- 0
    return(any(abs(off) > tol, na.rm = TRUE))
  }
  for (i in seq_along(C_list)) {
    off <- C_list[[i]]
    off[row(off) == col(off)] <- 0
    if (any(abs(off) > tol, na.rm = TRUE)) return(TRUE)
  }
  FALSE
}

.apch_alloc_loglik_mat <- function(
    Bhat_b,
    Sb,
    idx,
    full_grid_list,
    C_mode,
    C_shared,
    C_list,
    alloc_mat,
    use_parallel
) {
  noise_sub <- .apch_subset_C(C_mode, C_shared, C_list, idx)
  if (.apch_is_shared(noise_sub$C_mode)) {
    apch_alloc_loglik_shared_cpp(
      Bhat = Bhat_b,
      S = Sb,
      C_shared = noise_sub$C_shared,
      sgL = full_grid_list[idx],
      alloc_mat = alloc_mat,
      use_parallel = use_parallel
    )
  } else {
    apch_alloc_loglik_cpp(
      Bhat = Bhat_b,
      S = Sb,
      C_list = noise_sub$C_list,
      sgL = full_grid_list[idx],
      alloc_mat = alloc_mat,
      use_parallel = use_parallel
    )
  }
}


# shared internal hyperparameters for marginal fitting (ashr).
.apch_marginal_hyperparams <- function() {
  list(
    lambda0 = 10,
    fine_mult = 1.35,
    fine_s_min_mult = 0.1,
    fine_s_max_mult = 1.5,
    fine_max_len = 500
  )
}

# helper: default future plan for this package
.apch_default_future_plan <- function() {
  in_rstudio <- isTRUE(as.logical(Sys.getenv("RSTUDIO", "0") == "1"))
  if (.Platform$OS.type == "windows" || in_rstudio) "multisession" else "multicore"
}

# helper: resolve workers by plan (if workers is NULL)
.apch_resolve_workers <- function(plan, workers, workers_multicore = 16L, workers_multisession = 8L) {
  plan <- if (is.null(plan)) .apch_default_future_plan() else as.character(plan)[1L]
  if (is.null(workers)) workers <- if (identical(plan, "multisession")) workers_multisession else workers_multicore
  list(plan = plan, workers = max(1L, as.integer(workers)))
}

# helper: set future::plan once and return normalized plan string
.apch_set_future_plan <- function(plan = NULL, workers = 1L) {
  plan <- if (is.null(plan)) .apch_default_future_plan() else as.character(plan)[1L]
  strategy <- switch(
    plan,
    multisession = future::multisession,
    multicore = future::multicore
  )
  future::plan(strategy, workers = workers)
  plan
}
