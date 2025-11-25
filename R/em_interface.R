# em_interface.R — configuration layer (meta/xwas), NO EB penalty
# Provides:
#   - block_decompose()
#   - fit_config_meta()
#   - fit_config_xwas()
#   - fit_config_meta_stageA_only()     # Stage-A-only meta single block
#   - fit_config_xwas_stageA_only()     # Stage-A-only xwas single block
#   - apch_em_fit()  (wrapper selecting meta/xwas & StageA-only/StageB)

#' Block-diagonal decomposition of a symmetric matrix
#'
#' Given a symmetric matrix \code{A}, threshold small entries to zero and treat
#' the result as the adjacency matrix of an undirected graph. Connected
#' components of this graph define (approximate) diagonal blocks of \code{A}.
#'
#' This helper is used to exploit near-block-diagonal structure in the global
#' noise matrix \code{R_cor} for Stage A, then fuse blocks in Stage B.
#'
#' @param A Symmetric numeric matrix.
#' @param tau Numeric threshold; entries with \eqn{|A_{ij}| \le \tau} are set to
#'   zero before forming the adjacency matrix. Default: \code{1e-4}.
#'
#' @return A list with components:
#' \describe{
#'   \item{num_blocks}{Number of blocks (connected components).}
#'   \item{blocks}{List of block submatrices of \code{A}.}
#'   \item{indices}{List of integer index vectors, one per block.}
#' }
#'
#' @keywords internal
#' @importFrom igraph graph_from_adjacency_matrix components
block_decompose <- function(A, tau = 1e-4) {
  if (!is.matrix(A) && !inherits(A, "Matrix")) A <- as.matrix(A)
  if (!isSymmetric(A)) stop("Matrix must be symmetric.")
  
  G <- A
  G[abs(G) <= tau] <- 0
  G[abs(G) > 0]  <- 1
  
  g <- igraph::graph_from_adjacency_matrix(G, mode = "undirected", diag = FALSE)
  comps <- igraph::components(g)
  
  num_blocks <- comps$no
  membership <- comps$membership
  
  blocks  <- vector("list", num_blocks)
  indices <- vector("list", num_blocks)
  for (i in seq_len(num_blocks)) {
    idx <- which(membership == i)
    blocks[[i]]  <- A[idx, idx, drop = FALSE]
    indices[[i]] <- idx
  }
  
  list(num_blocks = num_blocks, blocks = blocks, indices = indices)
}

# ---------- Internal helper: StageA -> (delta, w_mat) given MSB configs ----------
# block_logf: n x R (R = 2^p), columns ordered by integer r=0..R-1 (C++ side)
# omega: length L, weights on eta_list
# eta_mat: L x p matrix of eta indices (0 = null, >0 = active component)
# configs: K x p binary matrix of configurations (MSB in column 1)
.apch_stageA_to_config <- function(block_logf, omega, eta_mat, configs) {
  block_logf <- as.matrix(block_logf)
  n  <- nrow(block_logf)
  R0 <- ncol(block_logf)
  
  if (!is.matrix(eta_mat)) {
    eta_mat <- as.matrix(eta_mat)
  }
  L <- nrow(eta_mat)
  p <- ncol(eta_mat)
  stopifnot(length(omega) == L)
  stopifnot(ncol(configs) == p)
  
  # map each eta (row) to a configuration index r in {0,...,2^p-1}, MSB convention
  pow2 <- 2^((p - 1):0)
  bits_eta <- (eta_mat != 0L)
  r_of_l <- as.integer(bits_eta %*% pow2)  # length L
  
  # aggregate omega to configuration-level priors pi_r in numeric r-order
  R <- 2^p
  if (R0 != R) {
    stop(sprintf(".apch_stageA_to_config: expected block_logf with %d columns, got %d.", R, R0))
  }
  pi_r <- numeric(R)
  for (l in seq_len(L)) {
    r <- r_of_l[l] + 1L
    pi_r[r] <- pi_r[r] + omega[l]
  }
  sum_pi <- sum(pi_r)
  if (sum_pi > 0) {
    pi_r <- pi_r / sum_pi
  } else {
    pi_r[] <- 1 / R
  }
  
  # reorder pi_r and likelihood columns to match the MSB-sorted configs from R
  cfg <- as.matrix(configs)
  stopifnot(ncol(cfg) == p)
  cfg_nums <- as.numeric(cfg %*% pow2)  # numeric r for each config row
  idx_cols <- as.integer(cfg_nums) + 1L
  if (any(idx_cols < 1L | idx_cols > R)) {
    stop(".apch_stageA_to_config: invalid config indices derived from configs.")
  }
  
  delta    <- pi_r[idx_cols]
  log_fmat <- block_logf[, idx_cols, drop = FALSE]
  
  # posterior w_mat(i, k) ∝ delta[k] * exp( log_fmat[i, k] )
  K <- length(delta)
  w_mat <- matrix(0, nrow = n, ncol = K)
  log_delta <- log(pmax(delta, .Machine$double.xmin))
  
  for (i in seq_len(n)) {
    log_post <- log_delta + log_fmat[i, ]
    m <- max(log_post)
    if (!is.finite(m)) next
    tmp <- exp(log_post - m)
    s   <- sum(tmp)
    if (!is.finite(s) || s <= 0) next
    w_mat[i, ] <- tmp / s
  }
  
  list(
    w_mat    = w_mat,
    delta    = delta,
    log_fmat = log_fmat
  )
}

#' Configuration layer: block-diagonal solver (meta-analysis mode, no penalty)
#'
#' Runs Stage A per block (using near block-diagonal structure in \code{R_cor}),
#' then fuses blocks in Stage B to obtain global configuration weights.
#'
#' Stage A uses \code{"mixsqp"}, \code{"em"}, or \code{"loglinear"}; Stage B
#' can use \code{"mcmc"}, \code{"mixsqp"}, \code{"em"}, or the new
#' \code{"loglinear"} (Ising + ridge) solver. There is **no** EB/size penalty
#' in Stage B. The Stage-A \code{"loglinear"} solver reparameterizes the
#' block-level mixture weights over \eqn{\eta}-patterns via a softmax and
#' adds an L2 penalty on the unconstrained parameters, using \code{stageA_ll_tol}
#' as its convergence tolerance (independent of \code{tol_A}).
#'
#' @param X Numeric matrix of observed effects (\eqn{n \times p}).
#' @param S Numeric matrix of standard errors (same shape as \code{X}).
#' @param R_cor Global \eqn{p \times p} correlation (or covariance) matrix.
#' @param sgL List of length \code{p}; \code{sgL[[j]]} is the signed mixture
#'   grid for feature \code{j}.
#' @param piL Optional list of length \code{p}; \code{piL[[j]]} are the
#'   corresponding mixture weights. When non-\code{NULL} and
#'   \code{mixcompdist = "normal"}, single-feature blocks are fast-tracked by
#'   reusing ashr results.
#' @param configs Binary configuration matrix (\eqn{K \times p}) of all global
#'   activation patterns (strictly MSB-ordered columns).
#' @param mixcompdist Slab family (e.g. \code{"normal"}, \code{"uniform"}).
#' @param tau Threshold for block decomposition of \code{R_cor}. Default \code{1e-4}.
#' @param max_iter_A,tol_A Stage-A iteration limit and tolerance for EM/mixsqp.
#'   For \code{"loglinear"} Stage A, \code{tol_A} is ignored and
#'   \code{stageA_ll_tol} is used instead.
#' @param max_iter_B,tol_B Stage-B iteration limit and tolerance for EM/mixsqp.
#'   For Stage-B \code{"loglinear"}, \code{tol_B} is ignored and
#'   \code{stageB_ll_tol} is used instead.
#' @param progress Logical; whether to print Stage-A progress.
#' @param progress_chunk Integer chunk size for Stage-A likelihood precomputation.
#' @param use_parallel Logical; enable multi-threading in C++ backends.
#' @param n_threads Max threads for \pkg{RcppParallel} (capped to 8).
#' @param verbose Logical; print diagnostics. MixSQP output is silenced; we only
#'   print a one-line convergence summary for Stage B.
#' @param output Either \code{"full"} (return Stage A/B details) or
#'   \code{"light"} (only \code{w_mat} and \code{delta}).
#' @param force_single_block Logical; treat all features as one block in Stage A
#'   (but still perform Stage B on the fused global configurations).
#' @param solver_stageA \code{"mixsqp"} (default), \code{"em"}, or
#'   \code{"loglinear"} (softmax + L2 ridge on block-level pattern weights)
#'   for Stage A.
#' @param solver_stageB \code{"mcmc"} (default), \code{"mixsqp"}, \code{"em"},
#'   or \code{"loglinear"} for Stage B.
#' @param stageA_ll_tol Tolerance for Stage-A \code{"loglinear"} solver
#'   (default \code{0.01}); ignored by other Stage-A solvers.
#' @param stageB_ll_tol Tolerance for Stage-B \code{"loglinear"} solver
#'   (default \code{0.01}); ignored by other Stage-B solvers.
#'
#' @return A list:
#' \describe{
#'   \item{w_mat}{Posterior configuration weights (\eqn{n \times K}).}
#'   \item{delta}{Global configuration probabilities (length \eqn{K}).}
#'   \item{stageA, stageB, block_logf}{(When \code{output="full"}) internals.}
#' }
#' @keywords internal
fit_config_meta <- function(
    X, S, R_cor, sgL, piL = NULL, configs,
    mixcompdist = "normal",
    tau = 1e-4,
    max_iter_A = 1e6, tol_A = 1e-6,
    max_iter_B = 1e6, tol_B = 1e-6,
    progress = interactive(),
    progress_chunk = 128,
    use_parallel = TRUE,
    n_threads = parallel::detectCores(logical = TRUE),
    verbose = TRUE,
    output = c("full", "light"),
    force_single_block = FALSE,
    solver_stageA  = c("mixsqp","em","loglinear"),
    solver_stageB  = c("mcmc","mixsqp","em","loglinear"),
    stageA_ll_tol  = 1e-2,
    stageB_ll_tol  = 1e-2
) {
  output        <- match.arg(output)
  solver_stageA <- match.arg(solver_stageA)
  solver_stageB <- match.arg(solver_stageB)
  
  X     <- as.matrix(X)
  S     <- as.matrix(S)
  R_cor <- as.matrix(R_cor)
  stopifnot(ncol(X) == ncol(S), nrow(X) == nrow(S))
  p <- ncol(X)
  stopifnot(all(dim(R_cor) == c(p, p)))
  if (!is.list(sgL) || length(sgL) != p) stop("sgL should be a list of length p.")
  if (!is.null(piL) && (!is.list(piL) || length(piL) != p)) stop("piL (if provided) should be a list of length p.")
  
  # threads for RcppParallel
  n_threads <- max(1L, min(8L, as.integer(n_threads)))
  old_env <- Sys.getenv("RCPP_PARALLEL_NUM_THREADS", unset = NA)
  if (use_parallel) Sys.setenv(RCPP_PARALLEL_NUM_THREADS = as.character(n_threads)) else Sys.setenv(RCPP_PARALLEL_NUM_THREADS = "1")
  on.exit({
    if (is.na(old_env)) Sys.unsetenv("RCPP_PARALLEL_NUM_THREADS")
    else Sys.setenv(RCPP_PARALLEL_NUM_THREADS = old_env)
  }, add = TRUE)
  
  # -------- block decomposition --------
  if (isTRUE(force_single_block)) {
    num_blocks <- 1L; indices <- list(seq_len(p))
  } else {
    decomp <- block_decompose(R_cor, tau = tau)
    num_blocks <- decomp$num_blocks
    indices    <- decomp$indices
    if (length(indices) != num_blocks) {
      stop("block_decompose returned inconsistent result.")
    }
  }
  
  block_sizes <- vapply(indices, length, integer(1))
  fast_flags  <- (block_sizes == 1L) & identical(mixcompdist, "normal") & !is.null(piL)
  
  if (isTRUE(verbose)) {
    cat(sprintf("[Config] Stage-A will process %d blocks; sizes: [%s]\n",
                num_blocks, paste(block_sizes, collapse = ", ")))
    if (any(block_sizes == 1L)) {
      if (any(fast_flags)) {
        cat(sprintf("[Config] singletons reusing ashr: %s\n", paste(which(fast_flags), collapse = ", ")))
      } else {
        cat("[Config] singleton blocks found but fast-track conditions unmet.\n")
      }
    }
  }
  
  block_logf_list <- vector("list", num_blocks)
  stageA_list     <- vector("list", num_blocks)
  
  # ---------------- Stage A ----------------
  for (b in seq_len(num_blocks)) {
    idx1 <- as.integer(indices[[b]])
    Xb   <- X[, idx1, drop = FALSE]
    Sb   <- S[, idx1, drop = FALSE]
    Rb   <- R_cor[idx1, idx1, drop = FALSE]
    sgB  <- sgL[idx1]
    
    # fast path: |B|=1 with ashr reuse
    if (length(idx1) == 1L && identical(mixcompdist, "normal") && !is.null(piL)) {
      j  <- idx1[1]
      x  <- as.numeric(X[, j]); s <- as.numeric(S[, j])
      sg <- as.numeric(sgL[[j]]); pj <- as.numeric(piL[[j]])
      if (!length(sg) || !length(pj) || length(sg) != length(pj) || length(sg) < 2L)
        stop(sprintf("Column %d has invalid sgL/piL.", j))
      pi0    <- pj[1]; w_act <- pj[-1]; sd_act <- sg[-1]; denom <- sum(w_act)
      
      logf0 <- dnorm(x, mean = 0, sd = s, log = TRUE)
      if (denom > 0 && length(sd_act) >= 1L) {
        dens_mat <- vapply(sd_act, function(sd_k) {
          dnorm(x, mean = 0, sd = sqrt(s^2 + sd_k^2), log = FALSE)
        }, numeric(length(x)))
        mix_dens <- as.numeric(dens_mat %*% w_act)
        logf1bar <- log(pmax(mix_dens, .Machine$double.xmin)) - log(denom)
      } else {
        logf1bar <- rep(-Inf, length(x))
      }
      block_logf <- cbind(logf0, logf1bar)
      colnames(block_logf) <- c("r0", "r1")
      resA <- list(omega = c(pi0, denom), block_logf = block_logf,
                   converged = TRUE, niter = 0L, solver = "ashr-fast")
      stageA_list[[b]]     <- resA
      block_logf_list[[b]] <- block_logf
      if (isTRUE(verbose)) message(sprintf("[StageA-fast] block %d reuse ashr.", b))
      next
    }
    
    Kj <- vapply(sgB, length, integer(1))
    eta_df  <- do.call(expand.grid, lapply(Kj, function(K) 0:(K - 1L)))
    eta_mat <- as.matrix(eta_df)
    eta_list <- lapply(seq_len(nrow(eta_mat)), function(i) as.integer(eta_mat[i, ]))
    L <- length(eta_list); if (L < 2L) stop(sprintf("[StageA] block %d has L=%d (<2).", b, L))
    
    zero_idx <- 1L
    eta_init <- rep(0.5 / (L - 1L), L); eta_init[zero_idx] <- 0.5
    
    if (identical(solver_stageA, "loglinear")) {
      ## softmax + L2 log-linear solver on eta-patterns (auto-λ, own tol)
      resA <- block_stageA_loglinear_cpp(
        Xb, Sb, Rb,
        sgL      = sgB,
        eta_list = eta_list,
        eta_init = eta_init,
        prior    = mixcompdist,
        tol      = stageA_ll_tol,
        progress = progress,
        progress_chunk = progress_chunk,
        use_parallel   = use_parallel,
        n_threads      = n_threads,
        verbose        = verbose
      )
    } else {
      ## 原有 EM / mixsqp MLE 分支
      resA <- block_stageA_cpp(
        Xb, Sb, Rb,
        sgL      = sgB,
        eta_list = eta_list,
        eta_init = eta_init,
        prior    = mixcompdist,
        solver   = solver_stageA,
        max_iter = max_iter_A,
        tol      = tol_A,
        progress = progress,
        progress_chunk = progress_chunk,
        use_parallel   = use_parallel,
        n_threads      = n_threads,
        verbose        = verbose
      )
    }
    
    stageA_list[[b]]     <- resA
    block_logf_list[[b]] <- resA$block_logf
  }
  
  # ---------------- Stage B (solver switch) ----------------
  cfg_sizes <- vapply(block_logf_list, ncol, integer(1))
  Rtot <- prod(cfg_sizes)
  if (Rtot < 2L) stop(sprintf("[StageB] Rtot = %d (<2).", Rtot))
  
  phi_init <- rep(0.5 / (Rtot - 1L), Rtot); phi_init[1L] <- 0.5
  indices_flat   <- unlist(indices, use.names = FALSE)
  ordered_config <- configs[, indices_flat, drop = FALSE]
  
  if (identical(solver_stageB, "loglinear")) {
    resB <- block_stageB_loglinear_cpp(
      block_logf   = block_logf_list,
      cfg          = ordered_config,
      tol          = stageB_ll_tol,
      verbose      = verbose
    )
    resB$solver <- "loglinear"
  } else if (identical(solver_stageB, "mcmc")) {
    resB <- block_stageB_mcmc_cpp(
      block_logf = block_logf_list,
      cfg        = ordered_config,
      p          = p,
      verbose    = verbose
    )
  } else {
    resB <- block_stageB_cpp(
      block_logf = block_logf_list,
      cfg        = ordered_config,
      phi_init   = phi_init,
      p          = p,
      max_iter   = max_iter_B,
      tol        = tol_B,
      verbose    = FALSE,
      solver     = solver_stageB
    )
  }
  
  if (isTRUE(verbose)) {
    if (isTRUE(resB$converged)) {
      cat(sprintf("[StageB] %s converged (niter=%s)\n", resB$solver, as.integer(resB$niter)))
    } else {
      cat(sprintf("[StageB] %s stopped without convergence (niter=%s)\n", resB$solver, as.integer(resB$niter)))
    }
  }
  
  if (output == "light") {
    return(list(w_mat = resB$w_mat, delta = resB$delta))
  } else {
    return(list(stageA = stageA_list, stageB = resB, w_mat = resB$w_mat, block_logf = block_logf_list))
  }
}

#' Configuration layer: xwas / per-effect covariance (no penalty)
#'
#' Stage A is performed on all features as a single block but allowing
#' per-effect covariance matrices \code{C_i}. Stage B then fuses
#' sub-configurations exactly as in the meta-analysis case.
#'
#' The solvers for Stage A and Stage B are controlled by \code{solver_stageA}
#' and \code{solver_stageB}, respectively. No EB/size penalty is used in Stage B.
#' \code{solver_stageB} also supports \code{"loglinear"} (Ising + ridge) and
#' Stage A supports a \code{"loglinear"} softmax+L2 solver on the block-level
#' mixture over \eqn{\eta}-patterns, controlled by \code{stageA_ll_tol}.
#'
#' @param X Numeric matrix of observed effects (\eqn{n \times p}).
#' @param S Numeric matrix of standard errors (same as \code{X}).
#' @param C_list List of length \code{n}; per-effect \eqn{p \times p} matrices.
#' @param sgL List of length \code{p}; signed mixture grid per feature.
#' @param piL Optional list of length \code{p}; weights per feature (kept for API symmetry).
#' @param configs Binary configuration matrix (\eqn{K \times p}) of all patterns.
#' @param mixcompdist Slab family (e.g. \code{"normal"}).
#' @param max_iter_A,tol_A Stage-A iteration limit and tolerance for EM/mixsqp.
#'   For \code{"loglinear"} Stage A, \code{tol_A} is ignored and
#'   \code{stageA_ll_tol} is used instead.
#' @param max_iter_B,tol_B Stage-B iteration limit and tolerance for EM/mixsqp.
#'   For Stage-B \code{"loglinear"}, \code{tol_B} is ignored and
#'   \code{stageB_ll_tol} is used instead.
#' @param progress Logical; Stage-A progress flag.
#' @param progress_chunk Integer chunk size for Stage-A likelihood precomputation.
#' @param use_parallel Logical; enable multi-threading in C++ backends.
#' @param n_threads Max threads for \pkg{RcppParallel} (capped to 8).
#' @param verbose Logical; print diagnostics.
#' @param output Either \code{"full"} (return Stage A/B details) or
#'   \code{"light"} (only \code{w_mat} and \code{delta}).
#' @param solver_stageA \code{"mixsqp"} (default), \code{"em"}, or
#'   \code{"loglinear"} for Stage A.
#' @param solver_stageB \code{"mcmc"} (default), \code{"mixsqp"}, \code{"em"},
#'   or \code{"loglinear"}.
#' @param stageA_ll_tol Tolerance for Stage-A \code{"loglinear"} solver
#'   (default \code{0.01}); ignored by other Stage-A solvers.
#' @param stageB_ll_tol Tolerance for Stage-B \code{"loglinear"} solver
#'   (default \code{0.01}); ignored by other Stage-B solvers.
#'
#' @return A list:
#' \describe{
#'   \item{w_mat}{Posterior configuration weights (\eqn{n \times K}).}
#'   \item{delta}{Global configuration probabilities (length \eqn{K}).}
#'   \item{stageA, stageB, block_logf}{(When \code{output="full"}) internals.}
#' }
#' @keywords internal
fit_config_xwas <- function(
    X, S, C_list, sgL, piL = NULL, configs,
    mixcompdist = "normal",
    max_iter_A = 1e6, tol_A = 1e-6,
    max_iter_B = 1e6, tol_B = 1e-6,
    progress = interactive(),
    progress_chunk = 128,
    use_parallel = TRUE,
    n_threads = parallel::detectCores(logical = TRUE),
    verbose = TRUE,
    output = c("full","light"),
    solver_stageA  = c("mixsqp","em","loglinear"),
    solver_stageB  = c("mcmc","mixsqp","em","loglinear"),
    stageA_ll_tol  = 1e-2,
    stageB_ll_tol  = 1e-2
) {
  output        <- match.arg(output)
  solver_stageA <- match.arg(solver_stageA)
  solver_stageB <- match.arg(solver_stageB)
  
  X <- as.matrix(X); S <- as.matrix(S)
  stopifnot(ncol(X) == ncol(S), nrow(X) == nrow(S))
  n <- nrow(X); p <- ncol(X)
  
  if (!is.list(C_list) || length(C_list) != n)
    stop("C_list must be a list of length n (each a p x p matrix).")
  if (!is.list(sgL) || length(sgL) != p)
    stop("sgL should be a list of length p.")
  if (!is.null(piL) && (!is.list(piL) || length(piL) != p))
    stop("piL (if provided) should be a list of length p.")
  
  # threads for RcppParallel
  n_threads <- max(1L, min(8L, as.integer(n_threads)))
  old_env <- Sys.getenv("RCPP_PARALLEL_NUM_THREADS", unset = NA)
  if (use_parallel) Sys.setenv(RCPP_PARALLEL_NUM_THREADS = as.character(n_threads)) else Sys.setenv(RCPP_PARALLEL_NUM_THREADS = "1")
  on.exit({
    if (is.na(old_env)) Sys.unsetenv("RCPP_PARALLEL_NUM_THREADS")
    else Sys.setenv(RCPP_PARALLEL_NUM_THREADS = old_env)
  }, add = TRUE)
  
  # -------- Stage A: single block over all features --------
  Kj <- vapply(sgL, length, integer(1L))
  eta_df  <- do.call(expand.grid, lapply(Kj, function(K) 0:(K - 1L)))
  eta_mat <- as.matrix(eta_df)
  eta_list <- lapply(seq_len(nrow(eta_mat)), function(i) as.integer(eta_mat[i, ]))
  L <- length(eta_list)
  if (L < 2L) stop(sprintf("[StageA-xwas] L = %d (< 2).", L))
  
  zero_idx <- 1L
  eta_init <- rep(0.5 / (L - 1L), L); eta_init[zero_idx] <- 0.5
  
  if (identical(solver_stageA, "loglinear")) {
    resA <- block_stageA_xwas_loglinear_cpp(
      X, S, C_list,
      sgL      = sgL,
      eta_list = eta_list,
      eta_init = eta_init,
      prior    = mixcompdist,
      tol      = stageA_ll_tol,
      progress = progress,
      progress_chunk = progress_chunk,
      use_parallel   = use_parallel,
      n_threads      = n_threads,
      verbose        = verbose
    )
  } else {
    resA <- block_stageA_xwas_cpp(
      X, S, C_list,
      sgL       = sgL,
      eta_list  = eta_list,
      eta_init  = eta_init,
      prior     = mixcompdist,
      solver    = solver_stageA,
      max_iter  = max_iter_A,
      tol       = tol_A,
      progress  = progress,
      progress_chunk = progress_chunk,
      use_parallel   = use_parallel,
      n_threads      = n_threads,
      verbose        = verbose
    )
  }
  block_logf_list <- list(resA$block_logf)
  
  # -------- Stage B: fusion (solver switch) --------
  cfg_sizes <- vapply(block_logf_list, ncol, integer(1L))
  Rtot <- prod(cfg_sizes)
  if (Rtot < 2L) stop(sprintf("[StageB-xwas] Rtot = %d (< 2).", Rtot))
  
  phi_init <- rep(0.5 / (Rtot - 1L), Rtot); phi_init[1L] <- 0.5
  
  if (identical(solver_stageB, "loglinear")) {
    resB <- block_stageB_loglinear_cpp(
      block_logf   = block_logf_list,
      cfg          = configs,
      tol          = stageB_ll_tol,
      verbose      = verbose
    )
    resB$solver <- "loglinear"
  } else if (identical(solver_stageB, "mcmc")) {
    resB <- block_stageB_mcmc_cpp(
      block_logf = block_logf_list,
      cfg        = configs,
      p          = p,
      verbose    = verbose
    )
  } else {
    resB <- block_stageB_cpp(
      block_logf = block_logf_list,
      cfg        = configs,
      phi_init   = phi_init,
      p          = p,
      max_iter   = max_iter_B,
      tol        = tol_B,
      verbose    = FALSE,
      solver     = solver_stageB
    )
  }
  
  if (isTRUE(verbose)) {
    if (isTRUE(resB$converged)) {
      cat(sprintf("[StageB] %s converged (niter=%s)\n", resB$solver, as.integer(resB$niter)))
    } else {
      cat(sprintf("[StageB] %s stopped without convergence (niter=%s)\n", resB$solver, as.integer(resB$niter)))
    }
  }
  
  if (output == "light") {
    return(list(w_mat = resB$w_mat, delta = resB$delta))
  } else {
    return(list(stageA = list(resA), stageB = resB, w_mat = resB$w_mat, block_logf = list(resA$block_logf)))
  }
}

# ---------- NEW: meta, Stage-A-only on a single global block ----------
#' Stage-A-only configuration solver (meta, single global block)
#'
#' Runs Stage A on a single block over all features, then directly maps the
#' block-level \eqn{\eta}-pattern weights to global configuration weights
#' using \code{configs}. No Stage B is performed.
#'
#' Stage A supports \code{"mixsqp"}, \code{"em"}, or \code{"loglinear"}; the
#' \code{"loglinear"} solver is a softmax + L2 ridge on the \eqn{\eta}-pattern
#' weights, controlled by \code{stageA_ll_tol}.
#'
#' @keywords internal
fit_config_meta_stageA_only <- function(
    X, S, R_cor, sgL, configs,
    mixcompdist    = "normal",
    max_iter_A     = 1e6,
    tol_A          = 1e-6,
    progress       = interactive(),
    progress_chunk = 128,
    use_parallel   = TRUE,
    n_threads      = parallel::detectCores(logical = TRUE),
    verbose        = TRUE,
    output         = c("full","light"),
    solver         = c("mixsqp","em","loglinear"),
    stageA_ll_tol  = 1e-2
) {
  output <- match.arg(output)
  solver <- match.arg(solver)
  
  X     <- as.matrix(X)
  S     <- as.matrix(S)
  R_cor <- as.matrix(R_cor)
  stopifnot(ncol(X) == ncol(S), nrow(X) == nrow(S))
  p <- ncol(X)
  stopifnot(all(dim(R_cor) == c(p, p)))
  if (!is.list(sgL) || length(sgL) != p) stop("sgL should be a list of length p.")
  
  # threads for RcppParallel
  n_threads <- max(1L, min(8L, as.integer(n_threads)))
  old_env <- Sys.getenv("RCPP_PARALLEL_NUM_THREADS", unset = NA)
  if (use_parallel) Sys.setenv(RCPP_PARALLEL_NUM_THREADS = as.character(n_threads)) else Sys.setenv(RCPP_PARALLEL_NUM_THREADS = "1")
  on.exit({
    if (is.na(old_env)) Sys.unsetenv("RCPP_PARALLEL_NUM_THREADS")
    else Sys.setenv(RCPP_PARALLEL_NUM_THREADS = old_env)
  }, add = TRUE)
  
  # single block over all p features
  Kj <- vapply(sgL, length, integer(1L))
  eta_df  <- do.call(expand.grid, lapply(Kj, function(K) 0:(K - 1L)))
  eta_mat <- as.matrix(eta_df)
  L <- nrow(eta_mat)
  if (L < 2L) stop(sprintf("[StageA-only-meta] L = %d (< 2).", L))
  eta_list <- lapply(seq_len(L), function(i) as.integer(eta_mat[i, ]))
  
  zero_idx <- 1L
  eta_init <- rep(0.5 / (L - 1L), L); eta_init[zero_idx] <- 0.5
  
  if (identical(solver, "loglinear")) {
    resA <- block_stageA_loglinear_cpp(
      X, S, R_cor,
      sgL      = sgL,
      eta_list = eta_list,
      eta_init = eta_init,
      prior    = mixcompdist,
      tol      = stageA_ll_tol,
      progress = progress,
      progress_chunk = progress_chunk,
      use_parallel   = use_parallel,
      n_threads      = n_threads,
      verbose        = verbose
    )
  } else {
    resA <- block_stageA_cpp(
      X, S, R_cor,
      sgL      = sgL,
      eta_list = eta_list,
      eta_init = eta_init,
      prior    = mixcompdist,
      solver   = solver,
      max_iter = max_iter_A,
      tol      = tol_A,
      progress = progress,
      progress_chunk = progress_chunk,
      use_parallel   = use_parallel,
      n_threads      = n_threads,
      verbose        = verbose
    )
  }
  
  omega      <- as.numeric(resA$omega)
  block_logf <- resA$block_logf
  
  cfg_res <- .apch_stageA_to_config(
    block_logf = block_logf,
    omega      = omega,
    eta_mat    = eta_mat,
    configs    = configs
  )
  
  if (output == "light") {
    list(
      w_mat = cfg_res$w_mat,
      delta = cfg_res$delta
    )
  } else {
    list(
      stageA     = list(resA),
      w_mat      = cfg_res$w_mat,
      delta      = cfg_res$delta,
      block_logf = list(cfg_res$log_fmat)
    )
  }
}

# ---------- NEW: xwas, Stage-A-only on a single block ----------
#' Stage-A-only configuration solver (xwas, single global block)
#'
#' Runs Stage A on a single block over all features with per-effect covariance
#' matrices \code{C_list}, then directly maps the \eqn{\eta}-pattern weights to
#' global configuration weights using \code{configs}. No Stage B is performed.
#'
#' Stage A supports \code{"mixsqp"}, \code{"em"}, or \code{"loglinear"}; the
#' \code{"loglinear"} solver is a softmax + L2 ridge on the \eqn{\eta}-pattern
#' weights, controlled by \code{stageA_ll_tol}.
#'
#' @keywords internal
fit_config_xwas_stageA_only <- function(
    X, S, C_list, sgL, configs,
    mixcompdist    = "normal",
    max_iter_A     = 1e6,
    tol_A          = 1e-6,
    progress       = interactive(),
    progress_chunk = 128,
    use_parallel   = TRUE,
    n_threads      = parallel::detectCores(logical = TRUE),
    verbose        = TRUE,
    output         = c("full", "light"),
    solver         = c("mixsqp","em","loglinear"),
    stageA_ll_tol  = 1e-2
) {
  output <- match.arg(output)
  solver <- match.arg(solver)
  
  X <- as.matrix(X); S <- as.matrix(S)
  stopifnot(ncol(X) == ncol(S), nrow(X) == nrow(S))
  n <- nrow(X); p <- ncol(X)
  
  if (!is.list(C_list) || length(C_list) != n)
    stop("C_list must be a list of length n (each a p x p matrix).")
  if (!is.list(sgL) || length(sgL) != p)
    stop("sgL should be a list of length p.")
  
  # threads for RcppParallel
  n_threads <- max(1L, min(8L, as.integer(n_threads)))
  old_env <- Sys.getenv("RCPP_PARALLEL_NUM_THREADS", unset = NA)
  if (use_parallel) Sys.setenv(RCPP_PARALLEL_NUM_THREADS = as.character(n_threads)) else Sys.setenv(RCPP_PARALLEL_NUM_THREADS = "1")
  on.exit({
    if (is.na(old_env)) Sys.unsetenv("RCPP_PARALLEL_NUM_THREADS")
    else Sys.setenv(RCPP_PARALLEL_NUM_THREADS = old_env)
  }, add = TRUE)
  
  # Stage A: single block over all features
  Kj <- vapply(sgL, length, integer(1L))
  eta_df  <- do.call(expand.grid, lapply(Kj, function(K) 0:(K - 1L)))
  eta_mat <- as.matrix(eta_df)
  eta_list <- lapply(seq_len(nrow(eta_mat)), function(i) as.integer(eta_mat[i, ]))
  L <- length(eta_list)
  if (L < 2L) stop(sprintf("[StageA-only-xwas] L = %d (< 2).", L))
  
  zero_idx <- 1L
  eta_init <- rep(0.5 / (L - 1L), L); eta_init[zero_idx] <- 0.5
  
  if (identical(solver, "loglinear")) {
    resA <- block_stageA_xwas_loglinear_cpp(
      X, S, C_list,
      sgL      = sgL,
      eta_list = eta_list,
      eta_init = eta_init,
      prior    = mixcompdist,
      tol      = stageA_ll_tol,
      progress = progress,
      progress_chunk = progress_chunk,
      use_parallel   = use_parallel,
      n_threads      = n_threads,
      verbose        = verbose
    )
  } else {
    resA <- block_stageA_xwas_cpp(
      X, S, C_list,
      sgL       = sgL,
      eta_list  = eta_list,
      eta_init  = eta_init,
      prior     = mixcompdist,
      solver    = solver,
      max_iter  = max_iter_A,
      tol       = tol_A,
      progress  = progress,
      progress_chunk = progress_chunk,
      use_parallel   = use_parallel,
      n_threads      = n_threads,
      verbose        = verbose
    )
  }
  
  omega      <- as.numeric(resA$omega)
  block_logf <- resA$block_logf
  
  cfg_res <- .apch_stageA_to_config(
    block_logf = block_logf,
    omega      = omega,
    eta_mat    = eta_mat,
    configs    = configs
  )
  
  if (output == "light") {
    list(
      w_mat = cfg_res$w_mat,
      delta = cfg_res$delta
    )
  } else {
    list(
      stageA     = list(resA),
      w_mat      = cfg_res$w_mat,
      delta      = cfg_res$delta,
      block_logf = list(cfg_res$log_fmat)
    )
  }
}

#' Fit APCH configuration-layer parameters (meta/xwas wrapper)
#'
#' Wrapper around the configuration layer for meta-analysis or xwas modes.
#' In \code{mode="meta"}, Stage A runs per block (or one block if \code{force_one_block}),
#' then Stage B fuses blocks using one of \code{solver_stageB}:
#' \code{"mcmc"}, \code{"mixsqp"}, \code{"em"}, or \code{"loglinear"} (Ising + ridge).
#' In \code{mode="xwas"}, a single global block is used for Stage A, and Stage B
#' is applied to the fused global configurations identically.
#' There is **no** EB/size penalty in Stage B.
#'
#' Stage A supports \code{"mixsqp"}, \code{"em"}, or \code{"loglinear"}; the
#' \code{"loglinear"} solver is a softmax + L2 ridge on the pattern weights
#' (over \eqn{\eta}-patterns in Stage A), controlled by \code{stageA_ll_tol}
#' (independent of \code{tol_A}). Stage B loglinear uses \code{stageB_ll_tol} as its
#' convergence tolerance (independent of \code{tol_B}).
#'
#' @inheritParams fit_config_meta
#' @param C Observation-noise structure.
#'   For \code{mode = "meta"}: a numeric \eqn{p \times p} matrix.
#'   For \code{mode = "xwas"}: a list of length \code{n} of \eqn{p \times p} matrices.
#' @param mode Either \code{"meta"} or \code{"xwas"}.
#' @param force_one_block Logical; when \code{TRUE} in meta-analysis mode,
#'   bypasses block decomposition in Stage A and treats all features as a
#'   single block, but still runs Stage B on the global configurations.
#'   In \code{mode = "xwas"}, a single block和 Stage-B 融合始终使用全局块。
#' @param solver_stageA Character; Stage-A solver used in the configuration
#'   layer (\code{"mixsqp"}, \code{"em"}, or \code{"loglinear"}).
#' @param solver_stageB Character; Stage-B solver (\code{"mcmc"} (default),
#'   \code{"mixsqp"}, \code{"em"}, or \code{"loglinear"}).
#' @param stageA_ll_tol Tolerance for Stage-A \code{"loglinear"} solver
#'   (default \code{0.01}); ignored by other Stage-A solvers.
#' @param stageB_ll_tol Tolerance for Stage-B \code{"loglinear"} solver
#'   (default \code{0.01}); ignored by other Stage-B solvers.
#'
#' @return A list mirroring \code{fit_config_meta()} or Stage-A-only solvers,
#'   with rows of \code{w_mat} normalized.
#' @export
apch_em_fit <- function(
    X, S, C,
    grid_list, pi_list = NULL,
    configs     = NULL,
    mixcompdist = "normal",
    tau_block   = 1e-4,
    mode        = c("meta", "xwas"),
    max_iter_A  = 1e6,
    tol_A       = 1e-6,
    max_iter_B  = 1e6,
    tol_B       = 1e-6,
    workers         = 8,
    progress_chunk  = 128,
    use_parallel    = TRUE,
    verbose         = TRUE,
    output          = c("full", "light"),
    progress        = interactive(),
    solver_stageA   = c("mixsqp","em","loglinear"),
    solver_stageB   = c("mcmc","mixsqp","em","loglinear"),
    force_one_block = FALSE,
    stageA_ll_tol   = 1e-2,
    stageB_ll_tol   = 1e-2
) {
  stopifnot(is.matrix(X), is.matrix(S), all(dim(X) == dim(S)))
  output        <- match.arg(output)
  mode          <- match.arg(mode, c("meta", "xwas"))
  solver_stageA <- match.arg(solver_stageA)
  solver_stageB <- match.arg(solver_stageB)
  p <- ncol(X)
  n <- nrow(X)
  
  ## feature-layer inputs
  if (!is.list(grid_list) || length(grid_list) != p) {
    stop("grid_list must be a list of length p (one grid per feature).")
  }
  if (is.null(pi_list)) {
    piL <- NULL
  } else {
    if (!is.list(pi_list) || length(pi_list) != p) {
      stop("pi_list (if provided) must be a list of length p.")
    }
    piL <- pi_list
  }
  
  ## configs
  if (is.null(configs)) {
    cfg <- generate_all_configs(p)
  } else {
    cfg <- as.matrix(configs)
    if (ncol(cfg) != p) stop("configs must have p columns.")
  }
  
  ## threads
  workers_em <- max(1L, min(as.integer(workers), 8L))
  
  ## solver availability guard (mixsqp -> em)
  if ((identical(solver_stageA, "mixsqp") || identical(solver_stageB, "mixsqp")) &&
      !requireNamespace("mixsqp", quietly = TRUE)) {
    message("[StageA/StageB] package 'mixsqp' not found; falling back to solver = 'em' where needed.")
    if (identical(solver_stageA, "mixsqp")) solver_stageA <- "em"
    if (identical(solver_stageB, "mixsqp")) solver_stageB <- "em"
  }
  
  if (mode == "meta") {
    ## C: global p x p
    if (is.null(C)) {
      C_mat <- diag(p)
    } else {
      C_mat <- as.matrix(C)
      if (!all(dim(C_mat) == c(p, p))) {
        stop("In apch_em_fit(..., mode='meta'), C must be a p x p matrix (or NULL).")
      }
    }
    
    ## meta: Stage A（可选单一大 block） + Stage B 融合
    config_res <- fit_config_meta(
      X = X, S = S, R_cor = C_mat,
      sgL = grid_list, piL = piL, configs = cfg,
      mixcompdist = mixcompdist,
      tau         = tau_block,
      max_iter_A  = max_iter_A, tol_A = tol_A,
      max_iter_B  = max_iter_B, tol_B = tol_B,
      progress    = progress,
      progress_chunk = progress_chunk,
      use_parallel   = use_parallel, n_threads = workers_em,
      verbose        = verbose,
      output         = output,
      force_single_block = isTRUE(force_one_block),
      solver_stageA = solver_stageA,
      solver_stageB = solver_stageB,
      stageA_ll_tol = stageA_ll_tol,
      stageB_ll_tol = stageB_ll_tol
    )
    
  } else {
    ## xwas: single block + Stage B
    if (is.null(C) || !is.list(C) || length(C) != n) {
      stop("In apch_em_fit(..., mode='xwas'), C must be a list of length n (p x p matrices).")
    }
    
    config_res <- fit_config_xwas(
      X = X, S = S, C_list = C,
      sgL        = grid_list, piL = piL, configs = cfg,
      mixcompdist = mixcompdist,
      max_iter_A  = max_iter_A, tol_A = tol_A,
      max_iter_B  = max_iter_B, tol_B = tol_B,
      progress    = progress,
      progress_chunk = progress_chunk,
      use_parallel   = use_parallel,
      n_threads      = workers_em,
      verbose        = verbose,
      output         = output,
      solver_stageA  = solver_stageA,
      solver_stageB  = solver_stageB,
      stageA_ll_tol  = stageA_ll_tol,
      stageB_ll_tol  = stageB_ll_tol
    )
  }
  
  ## Normalize w_mat row-wise
  if (!is.null(config_res$w_mat)) {
    w <- config_res$w_mat
    rs <- rowSums(w)
    rs[rs == 0] <- 1
    config_res$w_mat <- w / rs
  }
  
  config_res
}

