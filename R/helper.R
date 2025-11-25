## helper.R
# enumerate all 2^p configurations and return a 2^p x p matrix, where each row corresponds to one configuration
generate_all_configs <- function(p) {
  configs_df <- expand.grid(replicate(p, c(0,1), simplify = FALSE))
  configs_mat <- as.matrix(configs_df)[, p:1, drop = FALSE]
  weight_vec <- rowSums(configs_mat)
  nums <- apply(configs_mat, 1, function(r) sum(r * 2^((p-1):0)))
  # sorting: first sort by weight in ascending order, then sort by value in descending order within the same weight
  configs_sorted <- configs_mat[order(weight_vec, -nums), , drop = FALSE]
  return(configs_sorted)
}

# helper: fill and propagate dimnames (no extra checks)
.apch_fill_dimnames <- function(X, S, R_cor) {
  stopifnot(is.matrix(X), is.matrix(S))
  n <- nrow(X); p <- ncol(X)
  
  # only fill when missing; never overwrite existing names
  if (is.null(rownames(X))) rownames(X) <- paste0("e", seq_len(n))
  if (is.null(colnames(X))) colnames(X) <- paste0("f", seq_len(p))
  
  # propagate to S (same shape as X)
  dimnames(S) <- dimnames(X)
  
  # fill R_cor names if missing
  if (is.null(rownames(R_cor)) || is.null(colnames(R_cor))) {
    dimnames(R_cor) <- list(colnames(X), colnames(X))
  }
  
  list(X = X, S = S, R_cor = R_cor)
}

# helper: E[sigma^2 | nonzero] per feature from ashr grids
.vbar_from_grids <- function(grid_list, pi_list) {
  p <- length(grid_list)
  vbar <- numeric(p)
  for (j in seq_len(p)) {
    sig <- grid_list[[j]]; pik <- pi_list[[j]]
    nz <- (sig > 0)
    mass_nz <- sum(pik[nz])
    vbar[j] <- if (mass_nz > 0) sum(pik[nz] * sig[nz]^2) / mass_nz else 0
    if (!is.finite(vbar[j]) || vbar[j] <= 0) vbar[j] <- 1e-12
  }
  vbar
}

# helper: posterior mean (Gaussian slab) for selected effects only
.post_mean_gaussian_selected <- function(X, S, R_cor, w_mat, cfg, vbar, eff_idx, w_thresh = 1e-12) {
  p <- ncol(X)
  fnm <- colnames(X)
  R_inv <- chol2inv(chol(R_cor))
  
  lapply(eff_idx, function(i) {
    xi <- as.numeric(X[i, ]); si <- as.numeric(S[i, ])
    # guard
    pos <- si[is.finite(si) & si > 0]
    if (!all(is.finite(si)) || any(si <= 0)) {
      si[!is.finite(si) | si <= 0] <- if (length(pos)) min(pos) else 1.0
    }
    u <- xi / si
    Rinv_u <- as.numeric(R_inv %*% u)
    
    mu_sum <- numeric(p)
    for (r in seq_len(nrow(cfg))) {
      w <- w_mat[i, r]
      if (w <= w_thresh) next
      Sidx <- which(cfg[r, ] != 0L)
      if (!length(Sidx)) next
      
      sS   <- si[Sidx]
      Q_SS <- R_inv[Sidx, Sidx, drop = FALSE] / (sS %o% sS)
      Q_Sx <- Rinv_u[Sidx] / sS
      Dinv <- diag(1.0 / vbar[Sidx], nrow = length(Sidx))
      V_S  <- chol2inv(chol(Q_SS + Dinv))
      M_S  <- V_S %*% Q_Sx
      
      mu_r <- numeric(p); mu_r[Sidx] <- as.numeric(M_S)
      mu_sum <- mu_sum + w * mu_r
    }
    stats::setNames(mu_sum, fnm)
  })
}
