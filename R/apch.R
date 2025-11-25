#' Adaptive Partial Conjunction Hypotheses (APCH)
#'
#' APCH.R
#' Fit the Adaptive Partial Conjunction Hypotheses (APCH) model. APCH uses a
#' two-layer hierarchical structure for multi-feature partial
#' conjunction analysis. The first layer ("feature layer") fits adaptive
#' shrinkage models to each feature to learn flexible mixtures of null and
#' non-null effects. The second layer ("configuration layer") aggregates these
#' feature-wise posteriors across all features using a block-diagonal
#' representation of the noise correlation and a configuration-layer solver
#' (default: a log-linear / Ising + ridge model; alternatives include MCMC/EM/mixsqp).
#'
#' The high-level interface \code{apch()} is intended for routine use with
#' minimal tuning. The lower-level interface \code{apch_workhorse()} exposes
#' additional controls (e.g. ashr grid parameters, null thresholds, solver
#' options) and by default computes the full posterior.
#'
#' @details
#' The analysis proceeds in stages:
#' \enumerate{
#'   \item Fill and propagate names for rows (effects) and columns (features).
#'   \item Fit adaptive shrinkage (via \pkg{ashr}) independently to each feature
#'         to obtain mixture grids and weights (optionally with grid distillation).
#'   \item Construct all possible feature activation configurations and run a
#'         configuration-layer solver (C++ backend) to estimate configuration
#'         probabilities (\code{config_res$w_mat}). The default solver is a
#'         log-linear Ising model with ridge regularization; MCMC, EM and mixsqp
#'         are also available.
#'   \item Normalize and post-process the weights and perform level-by-level
#'         partial conjunction inference to detect multi-feature signals.
#'   \item Compute posterior summaries (means, variances, covariance) — done
#'         automatically in \code{apch_workhorse()}, and partially exposed by
#'         \code{apch()} via \code{post_mean_called}.
#' }
#'
#' The function is intended for moderate-to-large scale datasets (e.g.
#' thousands to millions of effects × moderate number of features). It
#' leverages \pkg{future} for R-level parallelism and \pkg{RcppParallel} for
#' C++-level multi-threading.
#'
#' @param X Numeric matrix of observed effect estimates with dimension
#'   \eqn{n \times p} (rows = effects, columns = features). Required.
#' @param S Numeric matrix of standard errors with the same dimensions as \code{X}.
#' @param C Observation-noise structure. For \code{mode = "meta"}, a numeric
#'   \eqn{p \times p} matrix shared across all effects (the global \eqn{C} in
#'   the APCH model, typically a correlation matrix in meta-analysis). If
#'   \code{NULL}, the identity matrix is used. For \code{mode = "xwas"},
#'   \code{C} encodes per-effect noise covariance: either a list of length
#'   \code{n} of \eqn{p \times p} matrices, or a \eqn{p \times p \times n}
#'   array. Internally this is normalized to a list.
#' @param mixcompdist Character specifying the mixture component distribution used
#'   by the adaptive shrinkage step. Typical values include \code{"normal"},
#'   \code{"uniform"},
#'   \code{"halfuniform"}, and \code{"halfnormal"}.
#'   Default: \code{"normal"}.
#' @param alpha Numeric overall significance level used by the level-by-level
#'   inference (e.g. FDR). Default: \code{0.05}.
#' @param mode Character string specifying the observation-noise model:
#'   \itemize{
#'     \item \code{"meta"}: a single global matrix \eqn{C} shared across all
#'       effects (standard meta-analysis setting).
#'     \item \code{"xwas"}: per-effect noise matrices \eqn{C_i}; \code{C} must
#'       be supplied as a list of length \code{n} of \eqn{p \times p} matrices
#'       or a \eqn{p \times p \times n} array. This mode is supported throughout
#'       the feature, EM and posterior layers, assuming the corresponding C++
#'       backends are available.
#'   }
#'   Additional labels are reserved for future extensions and are not yet
#'   supported.
#' @param workers Integer number of parallel workers to use at the R/future
#'   level. This will be capped at the number of features for the feature-level
#'   step and generally capped at 8 for the configuration layer. Default: \code{8}.
#' @param plan Optional character controlling the \pkg{future} plan:
#'   \code{"multisession"} or \code{"multicore"}. If \code{NULL}, the function
#'   chooses a reasonable default (multisession on Windows/RStudio,
#'   multicore on Unix-like systems). Default: \code{NULL}.
#' @param verbose Logical; when \code{TRUE}, print diagnostic and informational
#'   messages. Default: \code{TRUE}.
#'
#' @return
#' The high-level \code{apch()} function returns a named list in a “light” form:
#' \describe{
#'   \item{ashr_res}{Feature-layer adaptive shrinkage outputs in light form:
#'     a list with elements \code{signed_grids} and \code{pi_list} only.
#'     For full ashr fit objects, see \code{apch_workhorse()}.}
#'   \item{config_res}{Configuration-layer outputs in light form:
#'     currently the posterior configuration weights matrix \code{w_mat}.}
#'   \item{infer_res}{Level-by-level inference outputs:
#'      \code{calls} (data.frame of detected conjunctions), \code{est_fdr}
#'      (estimated FDR), and auxiliary items.}
#'   \item{post_mean_called}{Numeric matrix of posterior means for the called
#'      effects (rows = called effects, columns = features) when
#'      \code{mixcompdist = "normal"}. For non-Gaussian slabs the entries are
#'      filled with \code{NA_real_}. If there are no calls, this is a
#'      0 × p matrix.}
#' }
#'
#' Hyperparameters \code{lambda0_ash = 10}, \code{null_th = 0.9},
#' \code{ppa_min = 0.5} are fixed in \code{apch()}; advanced users can change
#' them via \code{apch_workhorse()}.
#'
#' @seealso \code{\link{apch_workhorse}}, \code{\link{fit_all_feature}},
#'   \code{\link{apch_posterior}}, \code{\link{level_by_level_inference}}
#' @export
apch <- function(
    X, S,
    C            = NULL,
    mixcompdist  = "normal",
    alpha        = 0.05,
    mode         = c("meta", "xwas"),
    workers      = 8,
    plan         = NULL,
    verbose      = TRUE
) {
  mode <- match.arg(mode)
  
  # Light wrapper: directly call workhorse with its default hyperparameters.
  # 从本版开始：apch() 默认使用 Stage-B = "loglinear"（Ising + ridge）
  res_full <- apch_workhorse(
    X = X,
    S = S,
    C = C,
    mode         = mode,
    mixcompdist  = mixcompdist,
    alpha        = alpha,
    workers      = workers,
    plan         = plan,
    verbose      = verbose,
    solver_stageB = "loglinear"
  )
  
  calls     <- res_full$infer_res$calls
  posterior <- res_full$posterior
  feature_names <- colnames(posterior$mean)
  
  is_gauss <- identical(tolower(mixcompdist), "normal")
  
  if (nrow(calls) > 0 && is_gauss) {
    eff_idx <- if (is.numeric(calls$effect)) {
      as.integer(calls$effect)
    } else {
      match(calls$effect, rownames(posterior$mean))
    }
    pm_mat <- posterior$mean[eff_idx, , drop = FALSE]
    rownames(pm_mat) <- calls$effect
    post_mean_called <- pm_mat
  } else if (nrow(calls) > 0 && !is_gauss) {
    post_mean_called <- matrix(
      NA_real_, nrow = nrow(calls), ncol = ncol(posterior$mean),
      dimnames = list(calls$effect, feature_names)
    )
  } else {
    post_mean_called <- matrix(
      numeric(0), nrow = 0, ncol = ncol(posterior$mean),
      dimnames = list(character(0), feature_names)
    )
  }
  
  ## ---- Light wrappers for ashr_res & config_res ----
  ashr_full <- res_full$ashr_res
  signed_grids <- if (!is.null(ashr_full$signed_grids)) ashr_full$signed_grids else ashr_full$sgL
  pi_list      <- if (!is.null(ashr_full$pi_list))      ashr_full$pi_list      else ashr_full$piL
  
  ashr_light <- list(
    signed_grids = signed_grids,
    pi_list      = pi_list
  )
  
  config_full <- res_full$config_res
  config_light <- list(
    w_mat = config_full$w_mat
  )
  
  list(
    ashr_res         = ashr_light,
    config_res       = config_light,
    infer_res        = res_full$infer_res,
    post_mean_called = post_mean_called
  )
}

# Internal helper: normalize input C by mode
# - mode = "meta": C_global is a p x p matrix shared by all effects
# - mode = "xwas": C is list(n) or a p x p x n array; converted to list for downstream
.apch_normalize_C <- function(X, S, C, mode) {
  stopifnot(is.matrix(X), is.matrix(S), all(dim(X) == dim(S)))
  n <- nrow(X); p <- ncol(X)
  
  if (!mode %in% c("meta", "xwas")) {
    stop("Unsupported mode in .apch_normalize_C: ", mode)
  }
  
  if (mode == "meta") {
    if (is.null(C)) {
      C_mat <- diag(p)
    } else {
      C_mat <- as.matrix(C)
      if (!all(dim(C_mat) == c(p, p))) {
        stop("In 'meta' mode, C must be a p x p matrix (or NULL).")
      }
    }
    return(list(
      mode     = "meta",
      C_global = C_mat,
      C_list   = NULL
    ))
  }
  
  ## -------- xwas mode: per-effect C_i --------
  if (is.null(C)) {
    stop("In 'xwas' mode, C must be provided per effect (list or p x p x n array).")
  }
  
  if (is.list(C)) {
    if (length(C) != n)
      stop("In 'xwas' mode, C must be a list of length n (one p x p matrix per effect).")
    C_list <- lapply(C, function(M) {
      M <- as.matrix(M)
      if (!all(dim(M) == c(p, p))) {
        stop("Each C[[i]] must be a p x p matrix.")
      }
      M
    })
  } else if (is.array(C) && length(dim(C)) == 3L &&
             dim(C)[1L] == p && dim(C)[2L] == p && dim(C)[3L] == n) {
    C_list <- lapply(seq_len(n), function(i) {
      as.matrix(C[, , i])
    })
  } else {
    stop("In 'xwas' mode, C must be a list of p x p matrices or a p x p x n array.")
  }
  
  list(
    mode     = "xwas",
    C_global = NULL,
    C_list   = C_list
  )
}

#' Workhorse APCH fitting function
#'
#' \code{apch_workhorse()} is a lower-level interface that exposes most modeling
#' and computing controls and by default computes the full posterior. Typical
#' users should call \code{apch()} instead.
#'
#' In addition to the arguments of \code{apch()}, it allows:
#' \itemize{
#'   \item Tuning the ashr null-bias prior (\code{lambda0_ash}).
#'   \item Changing level-by-level thresholds (\code{null_th}, \code{ppa_min}).
#'   \item Controlling grid distillation (\code{distill}).
#'   \item Passing detailed ashr/grid options via \code{ashr_control}, which
#'         are forwarded to \code{\link{fit_all_feature}}.
#'   \item Choosing separate solvers \code{solver_stageA} and
#'         \code{solver_stageB} for the configuration layer. By default,
#'         Stage A uses \code{"mixsqp"} and Stage B uses \code{"loglinear"}
#'         (Ising + ridge). Alternatives: \code{"mcmc"}, \code{"em"}, and a
#'         softmax+\eqn{L2} \code{"loglinear"} solver at Stage A that
#'         reparameterizes the pattern weights over \eqn{\eta}-configurations.
#'   \item Controlling the convergence tolerance of the Stage-A
#'         \code{"loglinear"} solver via \code{stageA_ll_tol} (separate from
#'         the EM/mixsqp tolerance \code{tol_A} used by other Stage-A solvers).
#'   \item Controlling the convergence tolerance of the Stage-B
#'         \code{"loglinear"} solver via \code{stageB_ll_tol} (separate from
#'         the EM/mixsqp tolerance \code{tol_B} used by other Stage-B solvers).
#'         All other hyperparameters for Stage-B \code{"loglinear"} are chosen
#'         internally in C++.
#'   \item Setting \code{force_one_block = TRUE} in meta-analysis mode to force
#'         a single global block at the configuration layer, with all features
#'         distilled and configuration probabilities inferred from Stage A
#'         only (Stage B is skipped). In \code{mode = "xwas"}, a single block
#'         and Stage-A-only inference are always used.
#'   \item Skipping the posterior layer by setting
#'         \code{compute_posterior = FALSE}, which avoids calling
#'         \code{\link{apch_posterior}} and can be useful for timing / debugging.
#'   \item Directly tuning the fine ashr grid range
#'         (\code{fine_s_min_mult}, \code{fine_s_max_mult}) and the
#'         distillation step multipliers
#'         (\code{fine_mult_distill}, \code{fine_mult_nodistill}), which are
#'         passed to \code{\link{fit_all_feature}}.
#' }
#'
#' Both \code{mode = "meta"} and \code{mode = "xwas"} are supported in the
#' feature, configuration and posterior layers, provided the corresponding
#' C++ backends (e.g. \code{block_stageA_xwas_cpp}) are available at compile time.
#'
#' @inheritParams apch
#' @param lambda0_ash Numeric regularization parameter used by the ashr
#'   feature-layer fit. Larger values encourage sparser solutions.
#'   Default: \code{10}.
#' @param null_th Numeric posterior probability threshold to treat an effect as
#'   null when computing calls (e.g. if posterior probability of null
#'   \eqn{\ge} \code{null_th} then it is considered null). Default: \code{0.9}.
#' @param ppa_min Numeric minimum posterior probability of activation (PPA)
#'   required to report a call. Default: \code{0.5}.
#' @param distill Logical, \code{"auto"}, integer vector, or character specifying
#'   whether to perform grid distillation at the feature-level. For
#'   \code{mode = "meta"}, \code{"auto"} uses a heuristic based on the global
#'   \code{C} and an internal correlation threshold. For \code{mode = "xwas"},
#'   all features are currently forced to be distilled.
#' @param fine_s_min_mult,fine_s_max_mult Numeric multipliers controlling the
#'   lower and upper range of the fine ashr grid (passed to
#'   \code{\link{fit_all_feature}}). Default: \code{0.5} and \code{1.0}.
#' @param fine_mult_distill,fine_mult_nodistill Numeric multipliers controlling
#'   the grid refinement step size for features selected for distillation and
#'   for non-distilled features, respectively. These are forwarded to
#'   \code{\link{fit_all_feature}} and then used as \code{fine_mult} in
#'   \code{distill_grid_path_from_xs()}. Default: \code{1.35} and \code{sqrt(2)}.
#' @param progress_chunk Integer chunk size for progress updates in the EM /
#'   configuration computations. Default: \code{128}.
#' @param ashr_control Optional named list of additional arguments forwarded to
#'   \code{\link{fit_all_feature}}. This can be used in meta-analysis mode to
#'   tweak the ashr grid construction (e.g. \code{fine_s_min_mult},
#'   \code{fine_s_max_mult}, \code{fine_max_len}, \code{min_grid_len},
#'   \code{pair_frac}, \code{rel_thresh}, \code{mll_drop_total_tol},
#'   \code{ks_increase_total_tol}, \code{mixsqp_control}), beyond what is
#'   controlled by \code{lambda0_ash} and \code{mixcompdist}. Arguments in
#'   \code{ashr_control} override the defaults set via the explicit parameters.
#' @param solver_stageA Character; solver for Stage A of the configuration
#'   layer. Options are \code{"mixsqp"} (optimized default),
#'   \code{"em"} (legacy EM+SQUAREM), or \code{"loglinear"} (softmax + L2
#'   penalty on the \eqn{\eta}-pattern weights at the block level).
#' @param solver_stageB Character; solver for Stage B of the configuration
#'   layer. Options are \code{"loglinear"} (default, Ising + ridge),
#'   \code{"mcmc"}, \code{"mixsqp"}, or \code{"em"}.
#'   Stage B is only used when \code{force_one_block = FALSE} in meta-analysis
#'   mode; in xwas mode a single-block Stage-A-only variant is used.
#' @param stageA_ll_tol Numeric tolerance for the Stage-A \code{"loglinear"}
#'   solver (softmax + L2 over \eqn{\eta}-patterns). Default \code{0.01}.
#'   Ignored by other Stage-A solvers.
#' @param stageB_ll_tol Numeric tolerance for the Stage-B \code{"loglinear"}
#'   solver (Ising + ridge). Default \code{0.01}. Ignored by non-loglinear
#'   Stage-B solvers.
#' @param force_one_block Logical; when \code{TRUE} in meta-analysis mode the
#'   configuration layer treats all features as a single block, all features
#'   are distilled at the feature layer, and only Stage A is used to infer
#'   configuration probabilities (Stage B is skipped). Ignored in
#'   \code{mode = "xwas"}, where a single-block Stage-A-only solver is always
#'   used.
#' @param compute_posterior Logical; when \code{FALSE}, skip the posterior
#'   layer and return \code{posterior = NULL}. This avoids calling
#'   \code{\link{apch_posterior}} and leaves \code{infer_res$calls$post_mean}
#'   as an empty list, which is useful for fast timing / debugging. Default:
#'   \code{TRUE}.
#'
#' @return
#' A named list in “full” form:
#' \describe{
#'   \item{ashr_res}{Feature-level ashr results (grids, mixture weights, and
#'      optionally full fit objects, depending on \code{fit_all_feature()}).}
#'   \item{config_res}{Configuration-layer results including \code{w_mat}
#'      (posterior configuration weights), convergence info, and diagnostics.}
#'   \item{infer_res}{Level-by-level inference outputs:
#'      \code{calls} (data.frame of detected conjunctions), \code{est_fdr}
#'      (estimated FDR), and auxiliary items. Each call also carries a
#'      \code{post_mean} entry (list of posterior means per feature) when
#'      Gaussian and \code{compute_posterior = TRUE}. When
#'      \code{compute_posterior = FALSE}, this field is an empty list.}
#'   \item{posterior}{Full posterior summaries:
#'      \code{mean}, \code{var_diag}, and optionally \code{cov_full}
#'      (3D array of covariances) when \code{compute_posterior = TRUE};
#'      otherwise \code{NULL}.}
#' }
#'
#' @rdname apch
#' @export
apch_workhorse <- function(
    X, S,
    C            = NULL,
    mode         = c("meta", "xwas"),
    mixcompdist  = "normal",
    lambda0_ash  = 10,
    alpha        = 0.05,
    null_th      = 0.9,
    ppa_min      = 0.5,
    distill      = "auto",
    fine_s_min_mult = 0.5,
    fine_s_max_mult = 1.0,
    fine_mult_distill   = 1.35,
    fine_mult_nodistill = sqrt(2),
    workers      = 8,
    plan         = NULL,
    verbose      = TRUE,
    progress_chunk = 128,
    ashr_control    = list(),
    solver_stageA   = c("mixsqp","em","loglinear"),          # Stage-A solver for config layer
    solver_stageB   = c("loglinear","mcmc","mixsqp","em"),   # Stage-B solver (default loglinear)
    stageA_ll_tol   = 1e-2,                                  # tolerance for Stage-A loglinear
    stageB_ll_tol   = 1e-2,                                  # tolerance for Stage-B loglinear
    force_one_block = FALSE,                                 # meta-only: force single block + Stage-A-only
    compute_posterior = TRUE                                 # whether to compute full posterior
) {
  stopifnot(is.matrix(X), is.matrix(S), all(dim(X) == dim(S)))
  mode          <- match.arg(mode, c("meta", "xwas"))
  solver_stageA <- match.arg(solver_stageA)
  solver_stageB <- match.arg(solver_stageB)
  p <- ncol(X)
  n <- nrow(X)
  
  # Always run in "full" mode internally
  output_internal <- "full"
  
  # ---------------- Normalize C according to mode ----------------
  noise <- .apch_normalize_C(X, S, C, mode)
  
  # ---------------- Fill & propagate dimnames ----------------
  if (noise$mode == "meta") {
    C_for_names <- noise$C_global
  } else {
    C_for_names <- diag(p)
  }
  
  named <- .apch_fill_dimnames(X, S, C_for_names)
  X     <- named$X
  S     <- named$S
  
  # meta: sync dimnames to global C
  if (noise$mode == "meta") {
    noise$C_global <- named$R_cor
  }
  
  feature_names <- colnames(X)
  effect_names  <- rownames(X)
  
  # Advisory for tiny off-diagonal correlations (meta mode only)
  if (noise$mode == "meta") {
    C_mat <- noise$C_global
    tiny_cor_thresh <- 0.05
    off <- row(C_mat) != col(C_mat)
    if (any(off)) {
      vals <- abs(C_mat[off])
      n_small <- sum(is.finite(vals) & vals > 0 & vals < tiny_cor_thresh)
      if (n_small > 0L) {
        message(sprintf(
          "[Advisory] C contains %d small off-diagonal correlations (|rho| < %.2f). ",
          n_small, tiny_cor_thresh
        ),
        "Consider setting them to 0 to ignore negligible dependencies.")
      }
    }
  }
  
  # ---------------- Parallelization setup ----------------
  workers_feat    <- max(1L, min(as.integer(workers), p))
  workers_config  <- max(1L, min(as.integer(workers), 8L))
  use_parallel_config <- (workers_config > 1L)
  
  # progress flag is hard-coded (not exposed to users)
  progress <- interactive()
  
  # raise the upper limit of the serialization volume for future's globals
  old_future_opts <- options(future.globals.maxSize = 20 * 1024^3)  # 20 GiB
  on.exit(options(old_future_opts), add = TRUE)
  
  if (is.null(plan)) {
    in_rstudio <- isTRUE(as.logical(Sys.getenv("RSTUDIO", "0") == "1"))
    if (.Platform$OS.type == "windows" || in_rstudio) {
      plan <- "multisession"
    } else {
      plan <- "multicore"
    }
  }
  if (identical(plan, "multisession")) {
    future::plan(future::multisession, workers = workers_feat)
  } else if (identical(plan, "multicore")) {
    future::plan(future::multicore, workers = workers_feat)
  } else {
    stop("Unsupported plan: use 'multisession' or 'multicore'.")
  }
  
  if (verbose) {
    cat(sprintf(
      "[Parallel] ashr workers=%d (plan=%s); config threads=%d (RcppParallel)\n",
      workers_feat, plan, workers_config
    ))
  }
  
  # ---------------- Enumerate all configurations (2^p x p) ----------------
  configs <- generate_all_configs(p)
  colnames(configs) <- feature_names
  
  # ---------------- Feature layer: ashr fitting ----------------
  base_ashr_args <- list(
    mixcompdist           = mixcompdist,
    lambda0               = lambda0_ash,
    fine_s_min_mult       = fine_s_min_mult,
    fine_s_max_mult       = fine_s_max_mult,
    fine_mult_distill     = fine_mult_distill,
    fine_mult_nodistill   = fine_mult_nodistill,
    fine_max_len          = 500,
    mixsqp_control        = NULL,
    mll_drop_total_tol    = 2e-4,
    ks_increase_total_tol = 1e-2,
    min_grid_len          = 2,
    max_merges            = Inf,
    pair_frac             = 0.20,
    rel_thresh            = 1e-6,
    distill               = distill,
    R_cor                 = if (noise$mode == "meta") noise$C_global else NULL,
    cor_thresh            = sqrt(.Machine$double.eps),
    workers               = workers_feat,
    plan                  = plan,
    future_seed           = TRUE,
    future_scheduling     = 1.0,
    output                = output_internal,
    verbose               = verbose
  )
  
  # For xwas: always distill all features.
  # For meta + force_one_block: also distill all features.
  if (noise$mode == "xwas" || (noise$mode == "meta" && isTRUE(force_one_block))) {
    base_ashr_args$distill <- TRUE
    if (noise$mode == "xwas") {
      base_ashr_args$R_cor <- NULL
    }
  }
  
  if (!is.null(ashr_control$output)) {
    ashr_control$output <- NULL
  }
  ashr_args <- utils::modifyList(base_ashr_args, ashr_control)
  
  ashr_res <- do.call(
    fit_all_feature,
    c(list(X = X, S = S), ashr_args)
  )
  
  grid_list <- if (!is.null(ashr_res$signed_grids)) ashr_res$signed_grids else ashr_res$sgL
  pi_list   <- if (!is.null(ashr_res$pi_list))      ashr_res$pi_list      else ashr_res$piL
  
  # ---------------- Configuration layer ----------------
  tau_block  <- 1e-4
  max_iter_A <- 1e5
  tol_A      <- 1e-5
  max_iter_B <- 1e5
  tol_B      <- 1e-5
  
  if (noise$mode == "meta") {
    C_for_config <- noise$C_global
    mode_config  <- "meta"
  } else {
    C_for_config <- noise$C_list
    mode_config  <- "xwas"
  }
  
  config_res <- apch_em_fit(
    X = X, S = S, C = C_for_config,
    grid_list   = grid_list,
    pi_list     = pi_list,
    configs     = configs,
    mixcompdist = mixcompdist,
    tau_block   = tau_block,
    mode        = mode_config,
    max_iter_A  = max_iter_A, tol_A = tol_A,
    max_iter_B  = max_iter_B, tol_B = tol_B,
    workers         = workers_config,
    progress_chunk  = progress_chunk,
    use_parallel    = use_parallel_config,
    verbose         = verbose,
    output          = output_internal,
    progress        = progress,
    solver_stageA   = solver_stageA,
    solver_stageB   = solver_stageB,
    force_one_block = isTRUE(force_one_block),
    stageA_ll_tol   = stageA_ll_tol,
    stageB_ll_tol   = stageB_ll_tol
  )
  
  # ---------------- Extract w_mat & set row names ----------------
  w_mat <- config_res$w_mat
  if (is.null(w_mat)) {
    stop("apch_workhorse: configuration result does not contain 'w_mat'.")
  }
  rownames(w_mat) <- effect_names
  
  # ---------------- Level-by-level inference ----------------
  infer_res <- level_by_level_inference(
    w_mat, configs, alpha = alpha, null_th = null_th, ppa_min = ppa_min,
    feature_names = feature_names
  )
  if (!is.null(infer_res$gamma_hat)) {
    dimnames(infer_res$gamma_hat) <- list(effect_names, feature_names)
  }
  
  # ---------------- Posterior layer (optional) ----------------
  is_gauss <- identical(tolower(mixcompdist), "normal")
  calls <- infer_res$calls
  
  if (isTRUE(compute_posterior)) {
    C_for_post <- if (noise$mode == "meta") noise$C_global else noise$C_list
    
    post_res <- apch_posterior(
      X = X, S = S, C = C_for_post,
      w_mat = w_mat, cfg = configs,
      grid_list = grid_list, pi_list = pi_list,
      mixcompdist = mixcompdist,
      return_cov_full = TRUE,
      w_thresh = 1e-12,
      mode = noise$mode
    )
    
    if (is_gauss && nrow(calls) > 0) {
      eff_idx <- if (is.numeric(calls$effect)) {
        as.integer(calls$effect)
      } else {
        match(calls$effect, rownames(w_mat))
      }
      pm_list <- lapply(
        eff_idx,
        function(i) stats::setNames(as.numeric(post_res$mean[i, ]), feature_names)
      )
      calls$post_mean <- I(pm_list)
    } else {
      calls$post_mean <- I(vector("list", nrow(calls)))
    }
  } else {
    # Skip posterior computation entirely
    post_res <- NULL
    calls$post_mean <- I(vector("list", nrow(calls)))
  }
  
  infer_res$calls <- calls
  
  list(
    ashr_res   = ashr_res,
    config_res = config_res,
    infer_res  = infer_res,
    posterior  = post_res
  )
}
