#' Level-by-Level Inference for Partial Conjunction Hypotheses
#'
#' The \code{level_by_level_inference()} function performs multi-level inference
#' on subset-level lfdr scores. It determines which
#' effects (rows) are active across subsets of features at varying conjunction
#' levels.
#'
#' @param lfdr_mat Matrix of subset-level local false discovery rates (n × M),
#'   where each column corresponds to one non-empty subset.
#' @param subset_vec_mat Binary subset matrix (M × p), representing all non-empty
#'   feature subsets.
#' @param alpha Significance level for false discovery rate control.
#' @param pnull Optional vector of per-effect posterior null probabilities.
#' @param feature_names Optional vector of feature names.
#' @param effect_names Optional vector of effect identifiers used in the
#'   \code{effect} column of \code{calls}.
#'
#' @return A list containing:
#' \describe{
#'   \item{calls}{Data frame of detected conjunctions, including effect name,
#'     level \code{L}, local FDR, and subset indicators.}
#'   \item{est_fdr}{Estimated overall false discovery rate.}
#' }
#'
#' @examples
#' \dontrun{
#' lvl_res <- level_by_level_inference(
#'   lfdr_mat = lfdr_mat,
#'   subset_vec_mat = subset_vec_mat,
#'   alpha = 0.05,
#'   pnull = pnull,
#'   feature_names = paste0("F", 1:p)
#' )
#'
#' cat("Calls returned:", nrow(lvl_res$calls), "\n")
#' head(lvl_res$calls)
#' }
#'
#' @noRd
.apch_select_lfdr <- function(
    lfdr_mat, subset_vec_mat,
    alpha = 0.05, pnull = NULL,
    feature_names = NULL,
    effect_names = NULL
) {
  active_idx <- .apch_active_effects_from_pnull(
    pnull = pnull,
    n = nrow(lfdr_mat),
    pnull_screen_thresh = 0.4
  )

  if (is.null(feature_names)) feature_names <- .apch_feature_names(subset_vec_mat)
  if (is.null(effect_names)) {
    eff_names <- .apch_effect_names(lfdr_mat)
  } else {
    eff_names <- as.character(effect_names)
  }

  if (length(active_idx) == 0L) {
    return(list(
      calls   = .apch_build_metric_calls(
        call_eff = integer(0),
        call_subset_id = integer(0),
        call_score = numeric(0),
        subset_vec_mat = subset_vec_mat,
        feature_names = feature_names,
        effect_names = eff_names,
        metric_name = "lfdr"
      ),
      est_fdr = 0
    ))
  }

  # C++ inference
  res <- level_infer_cpp(
    lfdr_mat[active_idx, , drop = FALSE],
    subset_vec_mat,
    alpha,
    NULL,
    0.4
  )
  
  list(
    calls   = .apch_build_metric_calls(
      call_eff = active_idx[as.integer(res$calls_eff)],
      call_subset_id = as.integer(res$calls_sid) + 1L,
      call_score = as.numeric(res$calls_lfdr),
      subset_vec_mat = subset_vec_mat,
      feature_names = feature_names,
      effect_names = eff_names,
      metric_name = "lfdr"
    ),
    est_fdr = res$est_fdr
  )
}

#' Level-by-level inference using subset-level LFSR
#'
#' Runs the APCH level-by-level search using subset-level LFSR scores rather
#' than subset-level LFDR scores. Effects with large posterior all-null
#' probability are skipped, and within each level LFSR is evaluated on all
#' subsets whose LFDR is below a screening threshold.
#'
#' @param fit Fitted object from \code{apch_em_fit()} or \code{apch_workhorse()}.
#' @param Bhat Numeric matrix of observed effects (n x p).
#' @param S Numeric matrix of standard errors (n x p).
#' @param C Noise structure (NULL, shared p x p, or list length n).
#' @param lfdr_bundle Subset-LFDR bundle from \code{apch_lfdr_bundle(joint_res)}.
#' @param alpha Significance level for the level-by-level selection rule.
#' @param pnull_screen_thresh Effects with posterior all-null probability at or
#'   above this threshold are treated as all-null and skipped.
#' @param level_skip_lfdr_thresh Within each level, if the effect's best LFDR is
#'   above this threshold, its LFSR is not evaluated for that subset. The
#'   default is \code{min(6 * alpha, 1)}.
#' @param workers Integer number of threads for likelihood evaluation.
#' @param use_parallel Logical; whether to enable parallel likelihood evaluation.
#' @param verbose Logical; print progress messages.
#' @param return_trace Logical; whether to return per-level lazy-search diagnostics.
#' @param pmvnorm_maxpts Maximum number of points for multivariate normal
#'   probability calculation (non-diagonal noise only).
#' @param pmvnorm_abseps Absolute tolerance for multivariate normal probability
#'   calculation (non-diagonal noise only).
#' @param pmvnorm_releps Relative tolerance for multivariate normal probability
#'   calculation (non-diagonal noise only).
#' @param pmvnorm_jitter Diagonal jitter added to posterior covariance blocks
#'   before calling \code{mvtnorm::pmvnorm} (non-diagonal noise only).
#'
#' @return A list with detected calls and summary metrics.
#'
#' @noRd
.apch_select_lfsr <- function(
    fit,
    Bhat, S,
    C = NULL,
    lfdr_bundle,
    alpha = 0.05,
    pnull_screen_thresh = 0.4,
    level_skip_lfdr_thresh = min(6 * alpha, 1),
    workers = 1,
    use_parallel = TRUE,
    verbose = TRUE,
    return_trace = FALSE,
    pmvnorm_maxpts = NULL,
    pmvnorm_abseps = NULL,
    pmvnorm_releps = NULL,
    pmvnorm_jitter = NULL
) {
  pmvnorm <- .apch_lfsr_pmvnorm_control(
    maxpts = pmvnorm_maxpts,
    abseps = pmvnorm_abseps,
    releps = pmvnorm_releps,
    jitter = pmvnorm_jitter
  )
  z_scaled <- .apch_scale_to_zscore(Bhat, S)
  Bhat <- z_scaled$Bhat
  S <- z_scaled$S
  lfdr_mat <- lfdr_bundle$lfdr_mat
  subset_vec_mat <- lfdr_bundle$subset_vec_mat
  effect_names <- rownames(Bhat)
  active_idx <- .apch_active_effects_from_pnull(
    pnull = lfdr_bundle$pnull,
    n = nrow(Bhat),
    pnull_screen_thresh = pnull_screen_thresh
  )

  if (length(active_idx) == 0L) {
    out <- list(
      calls = .apch_build_metric_calls(
        call_eff = integer(0),
        call_subset_id = integer(0),
        call_score = numeric(0),
        subset_vec_mat = subset_vec_mat,
        feature_names = colnames(Bhat),
        effect_names = effect_names,
        metric_name = "lfsr"
      ),
      est_avg_lfsr = 0,
      n_screened = nrow(Bhat),
      n_called = 0L
    )
    if (isTRUE(return_trace)) {
      p <- ncol(subset_vec_mat)
      out$trace <- do.call(rbind, lapply(p:1L, function(L) {
        .apch_trace_row(level = L, n_remaining = 0L, n_evaluated = 0L, n_pruned = 0L, n_accepted = 0L)
      }))
    }
    return(out)
  }

  alloc_state_active <- .apch_subset_lfsr_alloc_state(
    fit$joint_res$lfsr_alloc_state,
    active_idx
  )

  state <- .apch_prepare_lfsr_state(
    marginal_res = fit$marginal_res,
    joint_res = fit$joint_res,
    Bhat = Bhat[active_idx, , drop = FALSE],
    S = S[active_idx, , drop = FALSE],
    C = .apch_subset_effect_C_input(C, active_idx),
    workers = workers,
    use_parallel = use_parallel,
    verbose = verbose,
    alloc_state_override = alloc_state_active
  )
  if (!isTRUE(state$diag_ok) && isTRUE(state$use_shared) && !is.null(state$shared_lfsr_cache)) {
    state$shared_lfsr_cache_search <- .apch_attach_shared_subset_meta(
      shared_cache = state$shared_lfsr_cache,
      subset_vec_mat = subset_vec_mat,
      alloc_mat = state$alloc_mat
    )
  }

  res <- .apch_lfsr_level_search_fast(
    state = state,
    lfdr_mat = lfdr_mat[active_idx, , drop = FALSE],
    subset_vec_mat = subset_vec_mat,
    alpha = alpha,
    level_skip_lfdr_thresh = level_skip_lfdr_thresh,
    verbose = verbose,
    pmvnorm = pmvnorm
  )

  out <- list(
    calls = .apch_build_metric_calls(
      call_eff = res$call_eff,
      call_subset_id = res$call_subset_id,
      call_score = res$call_score,
      subset_vec_mat = subset_vec_mat,
      feature_names = state$feature_names,
      effect_names = state$effect_names,
      metric_name = "lfsr"
    ),
    est_avg_lfsr = .apch_mean_or_zero(res$call_score),
    n_screened = nrow(Bhat) - length(active_idx),
    n_called = length(res$call_eff)
  )

  if (isTRUE(return_trace)) {
    out$trace <- res$trace
  }
  out
}

.apch_lfsr_level_search_fast <- function(
    state,
    lfdr_mat,
    subset_vec_mat,
    alpha,
    level_skip_lfdr_thresh,
    verbose,
    pmvnorm
) {
  p <- ncol(subset_vec_mat)
  subset_level <- rowSums(subset_vec_mat)
  declared <- rep(FALSE, state$n)

  call_eff <- integer(0)
  call_subset_id <- integer(0)
  call_score <- numeric(0)
  trace_rows <- vector("list", p)

  for (L in p:1L) {
    subset_ids <- which(subset_level == L)
    remain <- which(!declared)
    if (length(subset_ids) == 0L || length(remain) == 0L) {
      trace_rows[[L]] <- .apch_trace_row(L, 0L, 0L, 0L, 0L)
      next
    }

    search_res <- .apch_search_level_fast(
      state = state,
      lfdr_mat = lfdr_mat,
      subset_vec_mat = subset_vec_mat,
      subset_ids = subset_ids,
      effect_idx = remain,
      level_skip_lfdr_thresh = level_skip_lfdr_thresh,
      pmvnorm = pmvnorm
    )

    cand_keep <- which(is.finite(search_res$best_score) & (search_res$best_score < 0.9))
    if (length(cand_keep) == 0L) {
      trace_rows[[L]] <- .apch_trace_row(
        L,
        length(remain),
        search_res$n_evaluated,
        search_res$n_pruned,
        0L
      )
      next
    }

    cand_eff <- remain[cand_keep]
    cand_sid <- search_res$best_sid[cand_keep]
    cand_score <- search_res$best_score[cand_keep]
    ord <- order(cand_score, cand_eff)
    cand_eff <- cand_eff[ord]
    cand_sid <- cand_sid[ord]
    cand_score <- cand_score[ord]

    keep_k <- .apch_prefix_keep(cand_score, alpha)
    if (keep_k > 0L) {
      take <- seq_len(keep_k)
      declared[cand_eff[take]] <- TRUE
      call_eff <- c(call_eff, cand_eff[take])
      call_subset_id <- c(call_subset_id, cand_sid[take])
      call_score <- c(call_score, cand_score[take])
    }

    trace_rows[[L]] <- .apch_trace_row(
      L,
      length(remain),
      search_res$n_evaluated,
      search_res$n_pruned,
      keep_k
    )
  }

  list(
    call_eff = call_eff,
    call_subset_id = call_subset_id,
    call_score = call_score,
    trace = do.call(rbind, trace_rows)
  )
}

.apch_search_level_fast <- function(
    state,
    lfdr_mat,
    subset_vec_mat,
    subset_ids,
    effect_idx,
    level_skip_lfdr_thresh,
    pmvnorm
) {
  lfdr_layer <- lfdr_mat[effect_idx, subset_ids, drop = FALSE]
  admissible <- is.finite(lfdr_layer) & (lfdr_layer <= level_skip_lfdr_thresh)
  has_any <- rowSums(admissible) > 0L
  n_pruned <- sum(!has_any)
  n_evaluated <- sum(admissible)

  best_score <- rep(Inf, length(effect_idx))
  best_sid_out <- rep(NA_integer_, length(effect_idx))

  if (n_evaluated > 0L) {
    sid_pos <- which(colSums(admissible) > 0L)

    if (!isTRUE(state$diag_ok) && isTRUE(state$use_shared) && !is.null(state$shared_lfsr_cache)) {
      cache_obj <- if (!is.null(state$shared_lfsr_cache_search)) {
        state$shared_lfsr_cache_search
      } else {
        state$shared_lfsr_cache
      }

      for (jj in sid_pos) {
        sid <- subset_ids[jj]
        pos <- which(admissible[, jj])
        eff_sub <- effect_idx[pos]
        val <- lfsr_nondiag_shared_cached_bestsid_cpp(
          z = state$z[eff_sub, , drop = FALSE],
          sid = rep.int(as.integer(sid), length(eff_sub)),
          subset_vec_mat = subset_vec_mat,
          alloc_mat = state$alloc_mat,
          alpha_mat = state$alpha_mat[eff_sub, , drop = FALSE],
          shared_cache = cache_obj,
          maxpts = pmvnorm$maxpts,
          abseps = pmvnorm$abseps,
          releps = pmvnorm$releps,
          jitter = pmvnorm$jitter
        )
        keep <- which(is.finite(val) & (val < best_score[pos]))
        if (length(keep) > 0L) {
          pos_keep <- pos[keep]
          best_score[pos_keep] <- val[keep]
          best_sid_out[pos_keep] <- as.integer(sid)
        }
      }
    } else {
      for (jj in sid_pos) {
        sid <- subset_ids[jj]
        pos <- which(admissible[, jj])
        eff_sub <- effect_idx[pos]
        val <- .apch_eval_lfsr_state(
          state = state,
          subset_vec_mat = subset_vec_mat[sid, , drop = FALSE],
          effect_idx = eff_sub,
          pmvnorm = pmvnorm,
          attach_dimnames = FALSE
        )[, 1L]
        keep <- which(is.finite(val) & (val < best_score[pos]))
        if (length(keep) > 0L) {
          pos_keep <- pos[keep]
          best_score[pos_keep] <- val[keep]
          best_sid_out[pos_keep] <- as.integer(sid)
        }
      }
    }
  }

  list(
    best_score = best_score,
    best_sid = best_sid_out,
    n_evaluated = n_evaluated,
    n_pruned = n_pruned
  )
}

.apch_attach_shared_subset_meta <- function(shared_cache, subset_vec_mat, alloc_mat) {
  out <- shared_cache
  out$subset_meta <- lfsr_nondiag_shared_build_subset_meta_cpp(
    subset_vec_mat = subset_vec_mat,
    alloc_mat = alloc_mat,
    shared_cache = shared_cache
  )
  out
}

.apch_prefix_keep <- function(score, alpha) {
  if (length(score) == 0L) {
    return(0L)
  }
  cs <- cumsum(score)
  keep <- which((cs / seq_along(score)) <= alpha)
  if (length(keep) == 0L) 0L else max(keep)
}

.apch_build_metric_calls <- function(
    call_eff,
    call_subset_id,
    call_score,
    subset_vec_mat,
    feature_names,
    effect_names,
    metric_name
) {
  if (length(call_eff) == 0L) {
    out <- data.frame(
      effect = character(0),
      L = integer(0),
      subset = character(0),
      subset01 = character(0),
      stringsAsFactors = FALSE
    )
    out[[metric_name]] <- numeric(0)
    return(out[, c("effect", "L", metric_name, "subset", "subset01"), drop = FALSE])
  }

  subset_ix <- lapply(call_subset_id, function(sid) which(subset_vec_mat[sid, ] > 0L))
  calls <- data.frame(
    effect = effect_names[call_eff],
    L = vapply(subset_ix, length, integer(1)),
    stringsAsFactors = FALSE
  )
  calls[[metric_name]] <- as.numeric(call_score)
  calls$subset <- vapply(
    subset_ix,
    function(ix) paste(feature_names[ix], collapse = "|"),
    character(1)
  )
  calls$subset01 <- vapply(
    subset_ix,
    function(ix) paste(as.integer(seq_len(length(feature_names)) %in% ix), collapse = "|"),
    character(1)
  )

  ord <- order(-calls$L, calls[[metric_name]], calls$effect)
  calls <- calls[ord, c("effect", "L", metric_name, "subset", "subset01"), drop = FALSE]
  rownames(calls) <- NULL
  calls
}

.apch_mean_or_zero <- function(x) {
  if (length(x) == 0L) 0 else mean(x)
}

.apch_trace_row <- function(level, n_remaining, n_evaluated, n_pruned, n_accepted) {
  data.frame(
    level = as.integer(level),
    n_remaining = as.integer(n_remaining),
    n_evaluated = as.integer(n_evaluated),
    n_pruned = as.integer(n_pruned),
    n_accepted = as.integer(n_accepted)
  )
}
