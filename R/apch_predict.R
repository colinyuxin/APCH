#' Predict configuration posterior weights for new data
#'
#' Given a fitted APCH model (the full output from \code{apch_workhorse()})
#' and new summary statistics \code{B}, \code{S} with the same set and order
#' of features, reuse the configuration-layer parameters (Stage-A block weights
#' and Stage-B global configuration probabilities \code{delta}) to compute
#' posterior configuration weights for the new data.
#'
#' This function is intended for the meta-analysis setting with
#' \code{force_one_block = FALSE}, i.e. the configuration layer used
#' block-diagonal decomposition in Stage A and then fused blocks in
#' Stage B via MCMC or EM/mixsqp. It assumes that the new noise correlation
#' matrix \code{C} is the same as (or compatible with) the one used at fit
#' time, so that the block structure is unchanged.
#'
#' @param res A fitted APCH object returned by \code{apch_workhorse()} with
#'   \code{mode = "meta"} and \code{force_one_block = FALSE}. It must contain
#'   \code{res$ashr_res}, \code{res$config_res$stageA}, and
#'   \code{res$config_res$stageB$delta}.
#' @param B Numeric matrix of new effect estimates (\eqn{n_{\mathrm{new}} \times p}).
#'   Columns (features) must match those used in the original fit (same order).
#' @param S Numeric matrix of new standard errors (same dimensions as \code{B}).
#' @param C Numeric \eqn{p \times p} correlation (or covariance) matrix for the
#'   new data. This should match the matrix used in the original fit, at least
#'   up to block structure (same thresholding \code{tau_block}).
#' @param mixcompdist Slab family used in the original fit (e.g. \code{"normal"},
#'   \code{"uniform"}). This is passed to the Stage-A prediction C++ routine.
#'   Default: \code{"normal"}.
#' @param tau_block Numeric threshold used for block decomposition of \code{C}.
#'   Must be the same value as used in the original fit (default \code{1e-4}).
#' @param progress_chunk Integer chunk size for Stage-A likelihood computation
#'   in the C++ predictor. Default: \code{128}.
#' @param use_parallel Logical; whether to allow C++ parallelisation via
#'   \pkg{RcppParallel}. Default: \code{TRUE}.
#' @param verbose Logical; print progress and diagnostics. Default: \code{TRUE}.
#'
#' @return A numeric matrix \code{w_mat} of size
#'   \eqn{n_{\mathrm{new}} \times 2^p}, where each row contains posterior
#'   configuration weights (PPA over all binary feature-activation patterns)
#'   for a new effect. Row names are copied from \code{B} (if present); column
#'   names are binary strings representing configurations in the original
#'   feature order.
#'
#' @export
apch_predict_wmat <- function(
    res,
    B, S,
    C,
    mixcompdist   = "normal",
    tau_block     = 1e-4,
    progress_chunk = 128L,
    use_parallel   = TRUE,
    verbose        = TRUE
) {
  # --- Basic checks and setup ---
  if (!is.list(res))
    stop("'res' must be a list returned by apch_workhorse().")
  if (!is.matrix(B) || !is.matrix(S))
    stop("'B' and 'S' must be numeric matrices.")
  B <- as.matrix(B)
  S <- as.matrix(S)
  if (!all(dim(B) == dim(S)))
    stop("Dimensions of 'B' and 'S' must match.")
  
  n_new <- nrow(B)
  p     <- ncol(B)
  
  if (missing(C) || is.null(C))
    stop("Argument 'C' must be supplied and should match the matrix used in the original fit.")
  C <- as.matrix(C)
  if (!isSymmetric(C))
    stop("'C' must be a symmetric p x p matrix.")
  if (!all(dim(C) == c(p, p)))
    stop("Dimension mismatch: ncol(B) must equal nrow(C) == ncol(C).")
  
  # --- Extract ashr grids and (optionally) mixture weights ---
  ashr_res <- res$ashr_res
  if (is.null(ashr_res))
    stop("res$ashr_res not found; please pass the full apch_workhorse() object.")
  
  sgL <- if (!is.null(ashr_res$signed_grids)) ashr_res$signed_grids else ashr_res$sgL
  if (is.null(sgL) || length(sgL) != p)
    stop("Could not extract signed grids from res$ashr_res, or length != ncol(B).")
  
  piL <- NULL
  if (!is.null(ashr_res$pi_list)) {
    piL <- ashr_res$pi_list
  } else if (!is.null(ashr_res$piL)) {
    piL <- ashr_res$piL
  }
  
  # --- Extract configuration-layer parameters from training ---
  config_res <- res$config_res
  if (is.null(config_res))
    stop("res$config_res not found; apch_workhorse() must have been run with configuration layer enabled.")
  
  stageA_train <- config_res$stageA
  stageB_train <- config_res$stageB
  if (is.null(stageA_train) || !length(stageA_train))
    stop("res$config_res$stageA is missing; prediction requires Stage-A block information.")
  if (is.null(stageB_train) || is.null(stageB_train$delta))
    stop("res$config_res$stageB$delta not found; prediction requires Stage-B configuration probabilities.")
  
  delta <- as.numeric(stageB_train$delta)
  
  # --- Block decomposition of C (must match training) ---
  decomp <- block_decompose(C, tau = tau_block)
  num_blocks  <- decomp$num_blocks
  indices     <- decomp$indices
  block_sizes <- vapply(indices, length, integer(1L))
  
  if (length(stageA_train) != num_blocks)
    stop("Number of blocks inferred from 'C' does not match res$config_res$stageA.")
  
  if (verbose) {
    cat(sprintf(
      "[Predict] Using %d blocks; sizes: [%s]\n",
      num_blocks, paste(block_sizes, collapse = ", ")
    ))
  }
  
  # --- Prepare C++ parallel settings ---
  n_threads <- parallel::detectCores(logical = TRUE)
  n_threads <- max(1L, min(8L, as.integer(n_threads)))
  
  # --- Stage A prediction: build block_logf for new data ---
  block_logf_new <- vector("list", num_blocks)
  
  for (b in seq_len(num_blocks)) {
    idx <- as.integer(indices[[b]])
    Bb  <- B[, idx, drop = FALSE]
    Sb  <- S[, idx, drop = FALSE]
    Cb  <- C[idx, idx, drop = FALSE]
    sgB <- sgL[idx]
    
    resA_b <- stageA_train[[b]]
    if (is.null(resA_b$omega))
      stop(sprintf("Stage-A object for block %d does not contain 'omega'.", b))
    omega_b <- as.numeric(resA_b$omega)
    
    # Detect the singleton fast path used at training time
    is_fast <- identical(resA_b$solver, "ashr-fast") &&
      length(idx) == 1L &&
      length(sgB) == 1L &&
      !is.null(piL)
    
    if (is_fast) {
      # ---- Singleton block: reproduce the analytic ashr-based formula ----
      j <- idx[1]
      x <- as.numeric(Bb[, 1])
      s <- as.numeric(Sb[, 1])
      g <- as.numeric(sgB[[1]])
      
      pj <- as.numeric(piL[[j]])
      if (!length(pj) || length(pj) != length(g)) {
        stop(sprintf(
          "Inconsistent pi_list / signed_grids for feature %d in singleton block %d.",
          j, b
        ))
      }
      
      pi0    <- pj[1L]
      w_act  <- pj[-1L]
      sd_act <- g[-1L]
      denom  <- sum(w_act)
      
      logf0 <- stats::dnorm(x, mean = 0, sd = s, log = TRUE)
      if (denom > 0 && length(sd_act) >= 1L) {
        dens_mat <- vapply(sd_act, function(sd_k) {
          stats::dnorm(x, mean = 0, sd = sqrt(s^2 + sd_k^2), log = FALSE)
        }, numeric(length(x)))
        mix_dens <- as.numeric(dens_mat %*% w_act)
        logf1bar <- log(pmax(mix_dens, .Machine$double.xmin)) - log(denom)
      } else {
        logf1bar <- rep(-Inf, length(x))
      }
      block_logf_new[[b]] <- cbind(logf0, logf1bar)
      colnames(block_logf_new[[b]]) <- c("r0", "r1")
      
      if (verbose) {
        message(sprintf("[Predict] Block %d (size = 1) using ashr-fast formula.", b))
      }
      
    } else {
      # ---- General block: call C++ Stage-A predictor with fixed omega ----
      Kj <- vapply(sgB, length, integer(1L))
      eta_df  <- do.call(expand.grid, lapply(Kj, function(K) 0:(K - 1L)))
      eta_mat <- as.matrix(eta_df)
      L       <- nrow(eta_mat)
      if (L != length(omega_b)) {
        stop(sprintf(
          "Block %d: product length(sgL) = %d, but length(omega) = %d.",
          b, L, length(omega_b)
        ))
      }
      eta_list <- lapply(seq_len(L), function(i) as.integer(eta_mat[i, ]))
      
      res_pred <- block_stageA_predict_cpp(
        X            = Bb,
        S            = Sb,
        Rcor         = Cb,
        sgL          = sgB,
        eta_list     = eta_list,
        omega        = omega_b,
        prior        = mixcompdist,
        progress     = verbose && interactive(),
        progress_chunk = as.integer(progress_chunk),
        use_parallel   = use_parallel,
        n_threads      = n_threads,
        verbose        = verbose
      )
      
      # We assume the C++ function returns a list with element "block_logf"
      if (is.null(res_pred$block_logf))
        stop(sprintf("block_stageA_predict_cpp did not return 'block_logf' for block %d.", b))
      
      block_logf_new[[b]] <- res_pred$block_logf
      
      if (verbose) {
        message(sprintf(
          "[Predict] Block %d (size = %d) via block_stageA_predict_cpp.",
          b, length(idx)
        ))
      }
    }
  }
  
  # --- Build cfg in the same row order as training ---
  # generate_all_configs(p) was used at fit time; only the *columns* were
  # reordered by block indices. Rows (config indices) are unchanged.
  cfg_global <- generate_all_configs(p)
  indices_flat <- unlist(indices, use.names = FALSE)
  cfg_ordered  <- cfg_global[, indices_flat, drop = FALSE]
  
  # --- Stage B prediction with fixed delta ---
  w_new <- block_stageB_predict_cpp(
    block_logf = block_logf_new,
    cfg        = cfg_ordered,
    delta      = delta,
    p          = p,
    verbose    = verbose
  )
  
  # Normalise rows (defensive; C++ should already do this)
  w_new <- as.matrix(w_new)
  rs <- rowSums(w_new)
  rs[rs == 0] <- 1
  w_new <- w_new / rs
  
  # Attach row / column names
  rownames(w_new) <- rownames(B)
  colnames(w_new) <- apply(cfg_global, 1L, function(z) paste0(z, collapse = ""))
  
  w_new
}
