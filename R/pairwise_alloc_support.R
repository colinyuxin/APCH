.apch_pairwise_indep_shared_comp_id <- function(C_mode, C_shared, C_list, p) {
  if (!.apch_is_shared(C_mode)) {
    return(NULL)
  }
  comps <- .apch_noise_components(C_mode, C_shared, C_list, p)
  comp_id <- integer(p)
  for (m in seq_along(comps)) {
    comp_id[as.integer(comps[[m]])] <- m
  }
  comp_id
}

.apch_pairwise_is_independent <- function(
    C_mode,
    C_shared,
    C_list,
    j,
    k,
    shared_comp_id = NULL,
    tol = 1e-12
) {
  if (.apch_is_shared(C_mode)) {
    return(shared_comp_id[j] != shared_comp_id[k])
  }
  !any(vapply(C_list, function(Ci) abs(Ci[j, k]) > tol, logical(1)))
}

.apch_build_pairwise_univariate_loglik_cache <- function(
    Bhat,
    S,
    C_mode,
    C_shared,
    C_list,
    marginal_inputs
) {
  p <- ncol(Bhat)
  lapply(seq_len(p), function(j) {
    .apch_build_block_univariate(
      Bhat = Bhat,
      idx = j,
      slab_list = marginal_inputs$slab_list,
      pi_list = marginal_inputs$pi_list,
      C_mode = C_mode,
      C_shared = C_shared,
      C_list = C_list
    )$log_alloc_mat
  })
}

.apch_take_marginal_loglik_cache <- function(marginal_res, full_grid_list, n) {
  cache <- marginal_res$loglik_cache
  if (length(cache) == length(full_grid_list)) cache else NULL
}

.apch_build_pairwise_indep_loglik_cache <- function(
    log_alloc_j,
    log_alloc_k,
    Kj_pair
) {
  alloc_core <- .apch_alloc_core(Kj_pair)
  list(
    alloc_mat = alloc_core$alloc_mat,
    alloc_code = alloc_core$alloc_code,
    log_alloc_mat = apch_combine_log_alloc_blocks_cpp(
      log_alloc_blocks = list(log_alloc_j, log_alloc_k),
      blocks = list(1L, 2L),
      Kvec = as.integer(Kj_pair),
      alloc_mat = alloc_core$alloc_mat
    )
  )
}

.apch_pair_loglik_cache_key <- function(j, k) {
  idx <- sort.int(c(as.integer(j), as.integer(k)), method = "shell")
  paste(idx[1L], idx[2L], sep = "__")
}

.apch_triple_loglik_cache_key <- function(j, k, l) {
  idx <- sort.int(c(as.integer(j), as.integer(k), as.integer(l)), method = "shell")
  paste(idx[1L], idx[2L], idx[3L], sep = "__")
}

.apch_quad_loglik_cache_key <- function(j, k, l, m) {
  idx <- sort.int(c(as.integer(j), as.integer(k), as.integer(l), as.integer(m)), method = "shell")
  paste(idx[1L], idx[2L], idx[3L], idx[4L], sep = "__")
}

.apch_make_component_loglik_cache <- function(order1 = NULL, order2 = NULL, order3 = NULL, order4 = NULL) {
  list(
    order1 = if (is.null(order1)) list() else order1,
    order2 = if (is.null(order2)) list() else order2,
    order3 = if (is.null(order3)) list() else order3,
    order4 = if (is.null(order4)) list() else order4
  )
}

.apch_empty_alloc_support <- function(feature_names) {
  structure(
    list(
      feature_names = feature_names,
      eff_n_thresh = NA_real_,
      eff_over_alpha_thresh = NA_real_,
      pairwise_prune_method = "none",
      pairwise_prune_configs = character(0),
      triple_prune_configs = character(0),
      quad_prune_configs = character(0),
      pairs = list(),
      triples = list(),
      quads = list(),
      loglik_cache = .apch_make_component_loglik_cache(),
      pair_summary = data.frame(),
      pair_states = data.frame(),
      pair_edge_states = data.frame(),
      pair_edge_summary = data.frame(),
      triple_summary = data.frame(),
      triple_states = data.frame(),
      triple_edge_states = data.frame(),
      triple_edge_summary = data.frame(),
      quad_summary = data.frame(),
      quad_states = data.frame(),
      edge_weight_summary = data.frame()
    ),
    class = "apch_pairwise_alloc_support"
  )
}

.apch_parse_component_cache_keys <- function(cache_names, size) {
  if (length(cache_names) == 0L) {
    return(matrix(integer(0), ncol = size))
  }
  parts <- strsplit(as.character(cache_names), "__", fixed = TRUE)
  vals <- as.integer(unlist(parts, use.names = FALSE))
  mat <- matrix(vals, ncol = size, byrow = TRUE)
  mat <- if (size > 1L) t(apply(mat, 1L, sort.int)) else matrix(mat, ncol = 1L)
  unique(matrix(as.integer(mat), ncol = size))
}

.apch_singleton_cache_candidates <- function(singleton_idx, component_loglik_cache) {
  singleton_idx <- as.integer(sort(unique(singleton_idx)))
  pair_mat <- matrix(integer(0), ncol = 2L)
  triple_mat <- matrix(integer(0), ncol = 3L)

  if (length(component_loglik_cache$order2) > 0L) {
    pair_keys <- .apch_parse_component_cache_keys(names(component_loglik_cache$order2), 2L)
    if (nrow(pair_keys) > 0L) {
      keep2 <- pair_keys[, 1L] %in% singleton_idx & pair_keys[, 2L] %in% singleton_idx
      pair_mat <- pair_keys[keep2, , drop = FALSE]
    }
  }
  if (length(component_loglik_cache$order3) > 0L) {
    triple_keys <- .apch_parse_component_cache_keys(names(component_loglik_cache$order3), 3L)
    if (nrow(triple_keys) > 0L) {
      keep3 <- triple_keys[, 1L] %in% singleton_idx &
        triple_keys[, 2L] %in% singleton_idx &
        triple_keys[, 3L] %in% singleton_idx
      triple_mat <- triple_keys[keep3, , drop = FALSE]
    }
  }

  list(pair = pair_mat, triple = triple_mat)
}

.apch_pick_better_singleton_partition <- function(best, cand) {
  if (is.null(best)) {
    return(cand)
  }
  if (cand$score > best$score) {
    return(cand)
  }
  if (cand$score < best$score) {
    return(best)
  }
  if (cand$n3 > best$n3) {
    return(cand)
  }
  if (cand$n3 < best$n3) {
    return(best)
  }
  if (cand$n_blocks < best$n_blocks) {
    return(cand)
  }
  if (cand$n_blocks > best$n_blocks) {
    return(best)
  }
  cand_head <- if (length(cand$blocks) == 0L) 0L else length(cand$blocks[[1L]])
  best_head <- if (length(best$blocks) == 0L) 0L else length(best$blocks[[1L]])
  if (cand_head > best_head) {
    return(cand)
  }
  if (cand_head < best_head) {
    return(best)
  }
  best
}

.apch_partition_singletons_exact <- function(singleton_idx, pair_mat, triple_mat) {
  singleton_idx <- as.integer(sort(singleton_idx))
  if (length(singleton_idx) == 0L) {
    return(list())
  }

  pair_by_node <- setNames(vector("list", length(singleton_idx)), as.character(singleton_idx))
  if (nrow(pair_mat) > 0L) {
    for (r in seq_len(nrow(pair_mat))) {
      edge <- as.integer(pair_mat[r, ])
      pair_by_node[[as.character(edge[1L])]] <- c(pair_by_node[[as.character(edge[1L])]], list(edge))
      pair_by_node[[as.character(edge[2L])]] <- c(pair_by_node[[as.character(edge[2L])]], list(edge))
    }
  }

  triple_by_node <- setNames(vector("list", length(singleton_idx)), as.character(singleton_idx))
  if (nrow(triple_mat) > 0L) {
    for (r in seq_len(nrow(triple_mat))) {
      edge <- as.integer(triple_mat[r, ])
      for (v in edge) {
        triple_by_node[[as.character(v)]] <- c(triple_by_node[[as.character(v)]], list(edge))
      }
    }
  }

  memo <- new.env(hash = TRUE, parent = emptyenv())
  solve <- function(rem) {
    if (length(rem) == 0L) {
      return(list(score = 0L, n3 = 0L, n_blocks = 0L, blocks = list()))
    }
    key <- paste(rem, collapse = "__")
    if (exists(key, envir = memo, inherits = FALSE)) {
      return(get(key, envir = memo, inherits = FALSE))
    }

    v <- rem[1L]
    tail_res <- solve(rem[-1L])
    best <- list(
      score = tail_res$score,
      n3 = tail_res$n3,
      n_blocks = tail_res$n_blocks + 1L,
      blocks = c(list(as.integer(v)), tail_res$blocks)
    )

    pair_cands <- pair_by_node[[as.character(v)]]
    if (length(pair_cands) > 0L) {
      for (edge in pair_cands) {
        edge <- as.integer(edge)
        if (!all(edge %in% rem)) next
        rem2 <- rem[!rem %in% edge]
        res2 <- solve(rem2)
        cand <- list(
          score = res2$score + 1L,
          n3 = res2$n3,
          n_blocks = res2$n_blocks + 1L,
          blocks = c(list(edge), res2$blocks)
        )
        best <- .apch_pick_better_singleton_partition(best, cand)
      }
    }

    triple_cands <- triple_by_node[[as.character(v)]]
    if (length(triple_cands) > 0L) {
      for (edge in triple_cands) {
        edge <- as.integer(edge)
        if (!all(edge %in% rem)) next
        rem3 <- rem[!rem %in% edge]
        res3 <- solve(rem3)
        cand <- list(
          score = res3$score + 2L,
          n3 = res3$n3 + 1L,
          n_blocks = res3$n_blocks + 1L,
          blocks = c(list(edge), res3$blocks)
        )
        best <- .apch_pick_better_singleton_partition(best, cand)
      }
    }

    assign(key, best, envir = memo)
    best
  }

  solve(singleton_idx)$blocks
}

.apch_singleton_assembly_blocks <- function(singleton_idx, component_loglik_cache) {
  singleton_idx <- as.integer(sort(unique(singleton_idx)))
  if (length(singleton_idx) == 0L) {
    return(list())
  }
  cand <- .apch_singleton_cache_candidates(singleton_idx, component_loglik_cache)
  if (nrow(cand$pair) == 0L && nrow(cand$triple) == 0L) {
    return(lapply(singleton_idx, as.integer))
  }
  out <- .apch_partition_singletons_exact(
    singleton_idx = singleton_idx,
    pair_mat = cand$pair,
    triple_mat = cand$triple
  )
  out[order(vapply(out, function(x) min(as.integer(x)), integer(1)))]
}

.apch_joint_assembly_blocks <- function(noise_blocks, component_loglik_cache) {
  has_order2 <- length(component_loglik_cache$order2) > 0L
  has_order3 <- length(component_loglik_cache$order3) > 0L
  has_order4 <- length(component_loglik_cache$order4) > 0L
  if (!(has_order2 || has_order3 || has_order4)) {
    return(noise_blocks)
  }
  singleton_idx <- sort.int(unlist(Filter(function(x) length(x) == 1L, noise_blocks), use.names = FALSE))
  nonsingleton <- Filter(function(x) length(x) > 1L, noise_blocks)
  singleton_blocks <- .apch_singleton_assembly_blocks(singleton_idx, component_loglik_cache)
  out <- c(nonsingleton, singleton_blocks)
  out[order(vapply(out, function(x) min(as.integer(x)), integer(1)))]
}

.apch_trim_component_loglik_cache_for_joint <- function(component_loglik_cache, C_mode, C_shared, C_list, p) {
  if (!.apch_is_shared(C_mode)) {
    return(component_loglik_cache)
  }
  noise_blocks <- .apch_noise_components(
    C_mode = C_mode,
    C_shared = C_shared,
    C_list = C_list,
    p = p
  )
  assembly_blocks <- .apch_joint_assembly_blocks(noise_blocks, component_loglik_cache)
  keep2 <- vapply(Filter(function(x) length(x) == 2L, assembly_blocks), function(x) {
    .apch_pair_loglik_cache_key(x[1L], x[2L])
  }, character(1))
  keep3 <- vapply(Filter(function(x) length(x) == 3L, assembly_blocks), function(x) {
    .apch_triple_loglik_cache_key(x[1L], x[2L], x[3L])
  }, character(1))
  keep4 <- vapply(Filter(function(x) length(x) == 4L, assembly_blocks), function(x) {
    .apch_quad_loglik_cache_key(x[1L], x[2L], x[3L], x[4L])
  }, character(1))
  if (length(component_loglik_cache$order2) > 0L) {
    component_loglik_cache$order2 <- if (length(keep2) == 0L) {
      list()
    } else {
      component_loglik_cache$order2[intersect(names(component_loglik_cache$order2), keep2)]
    }
  }
  if (length(component_loglik_cache$order3) > 0L) {
    component_loglik_cache$order3 <- if (length(keep3) == 0L) {
      list()
    } else {
      component_loglik_cache$order3[intersect(names(component_loglik_cache$order3), keep3)]
    }
  }
  if (length(component_loglik_cache$order4) > 0L) {
    component_loglik_cache$order4 <- if (length(keep4) == 0L) {
      list()
    } else {
      component_loglik_cache$order4[intersect(names(component_loglik_cache$order4), keep4)]
    }
  }
  component_loglik_cache
}

.apch_joint_target_order3_keys <- function(C_mode, C_shared, C_list, p, order2_cache) {
  if (!.apch_is_shared(C_mode)) {
    return(character(0))
  }
  noise_blocks <- .apch_noise_components(
    C_mode = C_mode,
    C_shared = C_shared,
    C_list = C_list,
    p = p
  )
  singleton_idx <- sort.int(unlist(Filter(function(x) length(x) == 1L, noise_blocks), use.names = FALSE))
  if (length(singleton_idx) < 3L) {
    return(character(0))
  }
  triple_idx <- utils::combn(singleton_idx, 3L, simplify = FALSE)
  order3_stub <- setNames(
    lapply(triple_idx, function(x) list(log_alloc_mat = NULL)),
    vapply(triple_idx, function(x) .apch_triple_loglik_cache_key(x[1L], x[2L], x[3L]), character(1))
  )
  component_stub <- .apch_make_component_loglik_cache(
    order2 = order2_cache,
    order3 = order3_stub
  )
  assembly_blocks <- .apch_joint_assembly_blocks(noise_blocks, component_stub)
  unique(vapply(Filter(function(x) length(x) == 3L, assembly_blocks), function(x) {
    .apch_triple_loglik_cache_key(x[1L], x[2L], x[3L])
  }, character(1)))
}

.apch_joint_target_order4_keys <- function(C_mode, C_shared, C_list, p) {
  if (!.apch_is_shared(C_mode)) {
    return(character(0))
  }
  noise_blocks <- .apch_noise_components(
    C_mode = C_mode,
    C_shared = C_shared,
    C_list = C_list,
    p = p
  )
  unique(vapply(Filter(function(x) length(x) == 4L, noise_blocks), function(x) {
    .apch_quad_loglik_cache_key(x[1L], x[2L], x[3L], x[4L])
  }, character(1)))
}

.apch_strip_pairwise_alloc_support_cache <- function(pairwise_alloc_support) {
  if (is.null(pairwise_alloc_support)) {
    return(NULL)
  }
  pairwise_alloc_support$loglik_cache <- NULL
  pairwise_alloc_support
}

.apch_parse_alloc_state_mat <- function(alloc_state, p) {
  alloc_state <- as.character(alloc_state)
  if (length(alloc_state) == 0L) {
    return(matrix(integer(0), ncol = p))
  }
  parts <- strsplit(alloc_state, ",", fixed = TRUE)
  vals <- as.integer(unlist(parts, use.names = FALSE))
  matrix(vals, ncol = p, byrow = TRUE)
}

.apch_code_to_config01 <- function(code, p) {
  code <- as.integer(code)
  bits <- integer(p)
  x <- code
  for (ii in p:1) {
    bits[ii] <- bitwAnd(x, 1L)
    x <- bitwShiftR(x, 1L)
  }
  paste(bits, collapse = "")
}

.apch_edge_weight_summary_df <- function(edge_states) {
  if (nrow(edge_states) == 0L) {
    return(data.frame())
  }
  group_key <- interaction(
    edge_states$source,
    edge_states$probe,
    edge_states$edge,
    edge_states$config01,
    drop = TRUE,
    lex.order = TRUE
  )
  out <- lapply(split(edge_states, group_key), function(df) {
    w <- df$edge_weight
    data.frame(
      source = df$source[1L],
      probe = df$probe[1L],
      edge = df$edge[1L],
      edge_feature1 = df$edge_feature1[1L],
      edge_feature2 = df$edge_feature2[1L],
      edge_j = df$edge_j[1L],
      edge_k = df$edge_k[1L],
      config01 = df$config01[1L],
      n_edge_state = length(w),
      weight_sum = sum(w),
      w_min = min(w),
      w_q25 = stats::quantile(w, probs = 0.25, names = FALSE, type = 7),
      w_median = stats::median(w),
      w_q75 = stats::quantile(w, probs = 0.75, names = FALSE, type = 7),
      w_max = max(w),
      n_w_lt_1e6 = sum(w < 1e-6),
      n_w_lt_1e4 = sum(w < 1e-4),
      n_w_lt_1e3 = sum(w < 1e-3),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, out)
}

.apch_build_pair_edge_weight_tables <- function(pair_states) {
  if (nrow(pair_states) == 0L) {
    return(list(states = data.frame(), summary = data.frame()))
  }
  edge_states <- data.frame(
    source = rep("pairwise", nrow(pair_states)),
    probe = pair_states$pair,
    edge = pair_states$pair,
    edge_feature1 = pair_states$feature1,
    edge_feature2 = pair_states$feature2,
    edge_j = pair_states$j,
    edge_k = pair_states$k,
    config01 = pair_states$config01,
    edge_state = pair_states$alloc_state,
    edge_weight = pair_states$cond_weight,
    allowed = pair_states$allowed,
    prior_q = pair_states$prior_q,
    alpha = pair_states$alpha,
    eff_n = pair_states$eff_n,
    ratio_active = pair_states$ratio_active,
    eff_over_alpha = pair_states$eff_over_alpha,
    collapsed_from = rep(1L, nrow(pair_states)),
    stringsAsFactors = FALSE
  )
  list(
    states = edge_states,
    summary = .apch_edge_weight_summary_df(edge_states)
  )
}

.apch_build_triple_edge_weight_tables <- function(triple_states) {
  if (nrow(triple_states) == 0L) {
    return(list(states = data.frame(), summary = data.frame()))
  }

  alloc_mat <- .apch_parse_alloc_state_mat(triple_states$alloc_state, p = 3L)

  mk_edge_long <- function(left_col, right_col, feat_left, feat_right, idx_left, idx_right) {
    data.frame(
      source = rep("triple", nrow(triple_states)),
      probe = triple_states$triple,
      triple = triple_states$triple,
      feature1 = triple_states$feature1,
      feature2 = triple_states$feature2,
      feature3 = triple_states$feature3,
      config01 = triple_states$config01,
      edge = paste(triple_states[[feat_left]], triple_states[[feat_right]], sep = "__"),
      edge_feature1 = triple_states[[feat_left]],
      edge_feature2 = triple_states[[feat_right]],
      edge_j = triple_states[[idx_left]],
      edge_k = triple_states[[idx_right]],
      edge_state = paste(alloc_mat[, left_col], alloc_mat[, right_col], sep = ","),
      cond_weight = triple_states$cond_weight,
      stringsAsFactors = FALSE
    )
  }

  long_df <- rbind(
    mk_edge_long(1L, 2L, "feature1", "feature2", "j", "k"),
    mk_edge_long(1L, 3L, "feature1", "feature3", "j", "l"),
    mk_edge_long(2L, 3L, "feature2", "feature3", "k", "l")
  )

  key_cols <- c(
    "source", "probe", "triple", "feature1", "feature2", "feature3",
    "config01", "edge", "edge_feature1", "edge_feature2", "edge_j", "edge_k", "edge_state"
  )
  agg_weight <- stats::aggregate(
    cond_weight ~ .,
    data = long_df[, c(key_cols, "cond_weight"), drop = FALSE],
    FUN = sum
  )
  names(agg_weight)[names(agg_weight) == "cond_weight"] <- "edge_weight"

  agg_count <- stats::aggregate(
    cond_weight ~ .,
    data = long_df[, c(key_cols, "cond_weight"), drop = FALSE],
    FUN = length
  )
  names(agg_count)[names(agg_count) == "cond_weight"] <- "collapsed_from"

  edge_states <- merge(
    agg_weight,
    agg_count,
    by = key_cols,
    all.x = TRUE,
    sort = FALSE
  )

  list(
    states = edge_states,
    summary = .apch_edge_weight_summary_df(edge_states)
  )
}

.apch_local_prune_eval <- function(
    cache,
    alloc_fit_state,
    prior_q,
    use_prior,
    dirichlet_count,
    prior_alpha_min,
    prior_alpha_cap,
    prior_alpha_cap_null,
    eff_n_thresh,
    eff_over_alpha_thresh,
    cond_weight_thresh,
    prune_codes
) {
  apch_local_prune_eval_cpp(
    alloc_mat = cache$alloc_mat,
    alloc_code = as.integer(cache$alloc_code),
    alloc_logw_cond = alloc_fit_state$alloc_logw_cond,
    alloc_post_colsum = as.numeric(alloc_fit_state$alloc_post_colsum),
    prior_q = if (isTRUE(use_prior)) as.numeric(prior_q) else numeric(0),
    use_prior = isTRUE(use_prior),
    prior_frac = dirichlet_count,
    n_effect = nrow(cache$log_alloc_mat),
    prior_alpha_min = prior_alpha_min,
    prior_alpha_cap = prior_alpha_cap,
    prior_alpha_cap_null = prior_alpha_cap_null,
    eff_n_thresh = eff_n_thresh,
    eff_over_alpha_thresh = eff_over_alpha_thresh,
    cond_weight_thresh = cond_weight_thresh,
    prune_codes = as.integer(prune_codes)
  )
}

.apch_local_state_slices <- function(
    alloc_mat,
    alloc_code,
    prune_configs,
    eval_res,
    prior_q,
    use_prior,
    keep_diagnostics = FALSE
) {
  out <- setNames(vector("list", length(prune_configs)), prune_configs)
  for (cfg in prune_configs) {
    code <- strtoi(cfg, base = 2L)
    idx_cfg <- which(alloc_code == code)
    alloc_cfg <- alloc_mat[idx_cfg, , drop = FALSE]
    allowed_cfg <- as.logical(eval_res$allowed[idx_cfg])
    states_df <- NULL
    if (isTRUE(keep_diagnostics)) {
      states_df <- data.frame(
        config01 = rep(cfg, nrow(alloc_cfg)),
        alloc_state = if (nrow(alloc_cfg) > 0L) apply(alloc_cfg, 1L, paste, collapse = ",") else character(0),
        cond_weight = as.numeric(eval_res$cond_weight[idx_cfg]),
        prior_q = if (isTRUE(use_prior)) as.numeric(prior_q[idx_cfg]) else rep(NA_real_, length(idx_cfg)),
        alpha = as.numeric(eval_res$alpha[idx_cfg]),
        eff_n = as.numeric(eval_res$eff_n[idx_cfg]),
        ratio_active = as.logical(eval_res$ratio_active[idx_cfg]),
        eff_over_alpha = as.numeric(eval_res$eff_over_alpha[idx_cfg]),
        allowed = allowed_cfg,
        stringsAsFactors = FALSE
      )
    }
    out[[cfg]] <- list(
      idx = idx_cfg,
      alloc = alloc_cfg,
      allowed = allowed_cfg,
      states = states_df
    )
  }
  out
}

.apch_build_allowed_tensors_from_slices <- function(slices, Kvec) {
  out <- setNames(vector("list", length(slices)), names(slices))
  for (cfg in names(slices)) {
    cfg_bits <- as.integer(strsplit(cfg, "", fixed = TRUE)[[1L]])
    active_pos <- which(cfg_bits > 0L)
    dims <- Kvec[active_pos]
    if (length(active_pos) == 1L) {
      tensor <- rep(TRUE, dims[1L])
      if (nrow(slices[[cfg]]$alloc) > 0L) {
        tensor[slices[[cfg]]$alloc[, active_pos[1L]]] <- slices[[cfg]]$allowed
      }
    } else if (length(active_pos) == 2L) {
      tensor <- matrix(TRUE, nrow = dims[1L], ncol = dims[2L])
      if (nrow(slices[[cfg]]$alloc) > 0L) {
        alloc_cfg <- slices[[cfg]]$alloc
        tensor[cbind(alloc_cfg[, active_pos[1L]], alloc_cfg[, active_pos[2L]])] <- slices[[cfg]]$allowed
      }
    } else if (length(active_pos) == 3L) {
      tensor <- array(TRUE, dim = c(dims[1L], dims[2L], dims[3L]))
      if (nrow(slices[[cfg]]$alloc) > 0L) {
        alloc_cfg <- slices[[cfg]]$alloc
        tensor[cbind(
          alloc_cfg[, active_pos[1L]],
          alloc_cfg[, active_pos[2L]],
          alloc_cfg[, active_pos[3L]]
        )] <- slices[[cfg]]$allowed
      }
    } else if (length(active_pos) == 4L) {
      tensor <- array(TRUE, dim = c(dims[1L], dims[2L], dims[3L], dims[4L]))
      if (nrow(slices[[cfg]]$alloc) > 0L) {
        alloc_cfg <- slices[[cfg]]$alloc
        tensor[cbind(
          alloc_cfg[, active_pos[1L]],
          alloc_cfg[, active_pos[2L]],
          alloc_cfg[, active_pos[3L]],
          alloc_cfg[, active_pos[4L]]
        )] <- slices[[cfg]]$allowed
      }
    } else {
      stop("unsupported local support dimension")
    }
    out[[cfg]] <- tensor
  }
  out
}

.apch_local_allowed_lookup <- function(tensor, idx_mat) {
  if (nrow(idx_mat) == 0L) {
    return(logical(0))
  }
  if (ncol(idx_mat) == 1L) {
    return(as.logical(tensor[idx_mat[, 1L]]))
  }
  if (ncol(idx_mat) == 2L) {
    return(as.logical(tensor[cbind(idx_mat[, 1L], idx_mat[, 2L])]))
  }
  if (ncol(idx_mat) == 3L) {
    return(as.logical(tensor[cbind(idx_mat[, 1L], idx_mat[, 2L], idx_mat[, 3L])]))
  }
  if (ncol(idx_mat) == 4L) {
    return(as.logical(tensor[cbind(idx_mat[, 1L], idx_mat[, 2L], idx_mat[, 3L], idx_mat[, 4L])]))
  }
  stop("unsupported local support dimension")
}

.apch_prune_alloc_mat_by_support_item <- function(alloc_mat, idx, allowed, prune_configs) {
  if (length(prune_configs) == 0L) {
    return(alloc_mat)
  }
  idx <- as.integer(idx)
  active_mat <- alloc_mat[, idx, drop = FALSE] > 0L
  keep <- rep(TRUE, nrow(alloc_mat))
  for (cfg in prune_configs) {
    tensor <- allowed[[cfg]]
    cfg_bits <- as.integer(strsplit(cfg, "", fixed = TRUE)[[1L]])
    mask <- rowSums(active_mat != matrix(cfg_bits > 0L, nrow = nrow(active_mat), ncol = length(idx), byrow = TRUE)) == 0L
    if (!any(mask)) next
    active_pos <- which(cfg_bits > 0L)
    keep[mask] <- keep[mask] & .apch_local_allowed_lookup(
      tensor = tensor,
      idx_mat = alloc_mat[mask, idx[active_pos], drop = FALSE]
    )
  }
  alloc_mat[keep, , drop = FALSE]
}

.apch_get_component_loglik_cache <- function(component_loglik_cache, idx) {
  idx <- as.integer(idx)
  bm <- length(idx)
  if (bm == 1L) {
    cache1 <- component_loglik_cache$order1
    if (length(cache1) < idx[1L]) return(NULL)
    return(cache1[[idx[1L]]])
  }
  if (bm == 2L) {
    cache2 <- component_loglik_cache$order2
    if (length(cache2) == 0L) return(NULL)
    key <- .apch_pair_loglik_cache_key(idx[1L], idx[2L])
    cache <- cache2[[key]]
    if (length(cache) == 0L) return(NULL)
    return(cache$log_alloc_mat)
  }
  if (bm == 3L) {
    cache3 <- component_loglik_cache$order3
    if (length(cache3) == 0L) return(NULL)
    key <- .apch_triple_loglik_cache_key(idx[1L], idx[2L], idx[3L])
    cache <- cache3[[key]]
    if (length(cache) == 0L) return(NULL)
    return(cache$log_alloc_mat)
  }
  if (bm == 4L) {
    cache4 <- component_loglik_cache$order4
    if (length(cache4) == 0L) return(NULL)
    key <- .apch_quad_loglik_cache_key(idx[1L], idx[2L], idx[3L], idx[4L])
    cache <- cache4[[key]]
    if (length(cache) == 0L) return(NULL)
    return(cache$log_alloc_mat)
  }
  NULL
}

.apch_prune_alloc_mat_by_pair_support_local <- function(alloc_mat, idx, pairwise_alloc_support) {
  if (is.null(pairwise_alloc_support)) {
    return(alloc_mat)
  }
  prune_configs <- pairwise_alloc_support$pairwise_prune_configs
  for (pr in pairwise_alloc_support$pairs) {
    pos_j <- match(pr$j, idx)
    pos_k <- match(pr$k, idx)
    if (is.na(pos_j) || is.na(pos_k)) next
    alloc_mat <- .apch_prune_alloc_mat_by_support_item(
      alloc_mat = alloc_mat,
      idx = c(pos_j, pos_k),
      allowed = pr$allowed,
      prune_configs = prune_configs
    )
  }
  alloc_mat
}

.apch_prune_alloc_mat_by_triple_support_local <- function(alloc_mat, idx, pairwise_alloc_support) {
  if (is.null(pairwise_alloc_support) || length(pairwise_alloc_support$triples) == 0L) {
    return(alloc_mat)
  }
  prune_configs <- pairwise_alloc_support$triple_prune_configs
  if (length(prune_configs) == 0L) {
    return(alloc_mat)
  }
  for (tr in pairwise_alloc_support$triples) {
    pos_j <- match(tr$j, idx)
    pos_k <- match(tr$k, idx)
    pos_l <- match(tr$l, idx)
    if (is.na(pos_j) || is.na(pos_k) || is.na(pos_l)) next
    alloc_mat <- .apch_prune_alloc_mat_by_support_item(
      alloc_mat = alloc_mat,
      idx = c(pos_j, pos_k, pos_l),
      allowed = tr$allowed,
      prune_configs = prune_configs
    )
  }
  alloc_mat
}

.apch_build_triple_alloc_loglik_cache <- function(
    Bhat,
    S,
    C_mode,
    C_shared,
    C_list,
    marginal_res,
    pairwise_alloc_support,
    workers,
    verbose
) {
  feature_names <- .apch_feature_names(Bhat)
  p <- ncol(Bhat)
  if (p < 3L) {
    return(list())
  }
  triple_idx <- utils::combn(seq_len(p), 3L, simplify = FALSE)
  marginal_inputs <- .apch_prepare_marginal_inputs(
    slab_list = marginal_res$slab_list,
    pi_list = marginal_res$pi_list,
    p = p
  )
  full_grid_list <- marginal_inputs$full_grid_list
  component_loglik_cache <- .apch_make_component_loglik_cache(
    order1 = .apch_take_marginal_loglik_cache(
      marginal_res = marginal_res,
      full_grid_list = full_grid_list,
      n = nrow(Bhat)
    ),
    order2 = pairwise_alloc_support$loglik_cache$order2
  )
  target_order3_keys <- .apch_joint_target_order3_keys(
    C_mode = C_mode,
    C_shared = C_shared,
    C_list = C_list,
    p = p,
    order2_cache = pairwise_alloc_support$loglik_cache$order2
  )
  workers_density <- max(1L, min(as.integer(workers), 12L))
  use_parallel_density <- workers_density > 1L

  RcppParallel::setThreadOptions(
    numThreads = if (use_parallel_density) workers_density else 1L
  )

  lapply(seq_along(triple_idx), function(ii) {
    idx_trip <- as.integer(triple_idx[[ii]])
    triple_key <- .apch_triple_loglik_cache_key(idx_trip[1L], idx_trip[2L], idx_trip[3L])
    keep_full_for_joint <- triple_key %in% target_order3_keys
    Kj_trip <- vapply(full_grid_list[idx_trip], length, integer(1))
    alloc_core <- .apch_alloc_core(Kj_trip)
    alloc_mat <- .apch_prune_alloc_mat_by_pair_support_local(
      alloc_mat = alloc_core$alloc_mat,
      idx = idx_trip,
      pairwise_alloc_support = pairwise_alloc_support
    )
    alloc_code <- .apch_alloc_code_from_mat(alloc_mat)

    noise_trip <- .apch_subset_C(C_mode, C_shared, C_list, idx_trip)
    noise_blocks <- .apch_noise_components(
      C_mode = noise_trip$C_mode,
      C_shared = noise_trip$C_shared,
      C_list = noise_trip$C_list,
      p = 3L
    )
    if (length(noise_blocks) == 3L) {
      assembly_global <- .apch_singleton_assembly_blocks(idx_trip, component_loglik_cache)
      assembly_blocks <- lapply(assembly_global, function(block_global) {
        as.integer(match(as.integer(block_global), idx_trip))
      })
    } else {
      assembly_blocks <- noise_blocks
    }
    block_logs <- lapply(assembly_blocks, function(block_local) {
      block_local <- as.integer(block_local)
      block_global <- idx_trip[block_local]
      cache_block <- .apch_get_component_loglik_cache(
        component_loglik_cache = component_loglik_cache,
        idx = block_global
      )
      if (length(cache_block) > 0L) {
        return(cache_block)
      }
      if (length(block_local) == 1L) {
        return(.apch_build_block_univariate(
          Bhat = Bhat,
          idx = block_global,
          slab_list = marginal_inputs$slab_list,
          pi_list = marginal_inputs$pi_list,
          C_mode = C_mode,
          C_shared = C_shared,
          C_list = C_list
        )$log_alloc_mat)
      }
      .apch_build_block_multivariate_log_alloc_mat(
        Bhat_b = Bhat[, idx_trip[block_local], drop = FALSE],
        Sb = S[, idx_trip[block_local], drop = FALSE],
        idx = block_local,
        full_grid_list = full_grid_list[idx_trip],
        C_mode = noise_trip$C_mode,
        C_shared = noise_trip$C_shared,
        C_list = noise_trip$C_list,
        use_parallel = use_parallel_density
      )$log_alloc_mat
    })
    log_alloc_mat_full <- NULL
    if (keep_full_for_joint) {
      log_alloc_mat_full <- apch_combine_log_alloc_blocks_cpp(
        log_alloc_blocks = block_logs,
        blocks = assembly_blocks,
        Kvec = as.integer(Kj_trip),
        alloc_mat = alloc_core$alloc_mat
      )
      if (nrow(alloc_mat) == nrow(alloc_core$alloc_mat)) {
        log_alloc_mat <- log_alloc_mat_full
      } else {
        state_full <- apply(alloc_core$alloc_mat, 1L, paste, collapse = ",")
        state_keep <- apply(alloc_mat, 1L, paste, collapse = ",")
        keep_idx <- match(state_keep, state_full)
        log_alloc_mat <- log_alloc_mat_full[, keep_idx, drop = FALSE]
      }
    } else {
      log_alloc_mat <- apch_combine_log_alloc_blocks_cpp(
        log_alloc_blocks = block_logs,
        blocks = assembly_blocks,
        Kvec = as.integer(Kj_trip),
        alloc_mat = alloc_mat
      )
    }

    list(
      triple = paste(feature_names[idx_trip], collapse = "__"),
      triple_key = triple_key,
      feature1 = feature_names[idx_trip[1L]],
      feature2 = feature_names[idx_trip[2L]],
      feature3 = feature_names[idx_trip[3L]],
      j = idx_trip[1L],
      k = idx_trip[2L],
      l = idx_trip[3L],
      alloc_mat = alloc_mat,
      alloc_code = alloc_code,
      log_alloc_mat = log_alloc_mat,
      log_alloc_mat_full = log_alloc_mat_full
    )
  })
}

.apch_build_triple_alloc_support_from_cache <- function(
    triple_cache,
    marginal_res,
    eff_n_thresh = 1,
    eff_over_alpha_thresh = 10,
    dirichlet_count,
    workers,
    keep_diagnostics = FALSE,
    verbose
) {
  triple_prune_configs <- c("001", "010", "011", "100", "101", "110", "111")
  prune_codes_triple <- strtoi(triple_prune_configs, base = 2L)
  prior_alpha_min <- 1e-4
  prior_alpha_cap <- 5
  prior_alpha_cap_null <- 10
  cond_weight_thresh <- 1e-6
  if (length(triple_cache) == 0L) {
    return(list(
      triples = list(),
      order3_cache = list(),
      triple_prune_configs = triple_prune_configs,
      triple_summary = data.frame(),
      triple_states = data.frame(),
      triple_edge_states = data.frame(),
      triple_edge_summary = data.frame()
    ))
  }
  workers_triple <- max(1L, min(length(triple_cache), as.integer(workers)))

  triple_fun <- function(cache) {
    idx_trip <- c(cache$j, cache$k, cache$l)
    pi_block <- marginal_res$pi_list[idx_trip]
    pnull_block <- marginal_res$p_null[idx_trip]
    use_prior <- (dirichlet_count > 0)
    prior_q <- if (use_prior) {
      .apch_alloc_prior_center_from_marginals(
        alloc_mat = cache$alloc_mat,
        p_null = pnull_block,
        pi_list = pi_block
      )
    } else {
      NULL
    }
    alloc_fit_state <- .apch_build_apchs_alloc_logw_cond_from_log_alloc(
      log_alloc_mat = cache$log_alloc_mat,
      alloc_mat = cache$alloc_mat,
      alloc_code = cache$alloc_code,
      idx = 1:3,
      pi_list = pi_block,
      prior_q = prior_q,
      prior_frac = dirichlet_count,
      prior_alpha_min = prior_alpha_min,
      prior_alpha_cap = prior_alpha_cap,
      prior_alpha_cap_null = prior_alpha_cap_null,
      keep_alloc_post_colsum = TRUE,
      blas_threads = 1L
    )
    K1 <- length(pi_block[[1L]])
    K2 <- length(pi_block[[2L]])
    K3 <- length(pi_block[[3L]])
    eval_res <- .apch_local_prune_eval(
      cache = cache,
      alloc_fit_state = alloc_fit_state,
      prior_q = prior_q,
      use_prior = use_prior,
      dirichlet_count = dirichlet_count,
      prior_alpha_min = prior_alpha_min,
      prior_alpha_cap = prior_alpha_cap,
      prior_alpha_cap_null = prior_alpha_cap_null,
      eff_n_thresh = eff_n_thresh,
      eff_over_alpha_thresh = eff_over_alpha_thresh,
      cond_weight_thresh = cond_weight_thresh,
      prune_codes = prune_codes_triple
    )
    slices <- .apch_local_state_slices(
      alloc_mat = cache$alloc_mat,
      alloc_code = cache$alloc_code,
      prune_configs = triple_prune_configs,
      eval_res = eval_res,
      prior_q = prior_q,
      use_prior = use_prior,
      keep_diagnostics = keep_diagnostics
    )
    allowed <- .apch_build_allowed_tensors_from_slices(
      slices = slices,
      Kvec = c(K1, K2, K3)
    )

    states_df <- NULL
    summary_df <- NULL
    if (isTRUE(keep_diagnostics)) {
      states_df <- do.call(rbind, lapply(slices, `[[`, "states"))
      states_df <- data.frame(
        triple = rep(cache$triple, nrow(states_df)),
        feature1 = rep(cache$feature1, nrow(states_df)),
        feature2 = rep(cache$feature2, nrow(states_df)),
        feature3 = rep(cache$feature3, nrow(states_df)),
        j = rep(cache$j, nrow(states_df)),
        k = rep(cache$k, nrow(states_df)),
        l = rep(cache$l, nrow(states_df)),
        states_df,
        alloc111 = ifelse(states_df$config01 == "111", states_df$alloc_state, NA_character_),
        cond_weight111 = ifelse(states_df$config01 == "111", states_df$cond_weight, NA_real_),
        stringsAsFactors = FALSE
      )
      count_cfg <- function(x) c(n_alloc = length(x$allowed), n_allowed = sum(x$allowed), n_pruned = sum(!x$allowed))
      c001 <- count_cfg(slices[["001"]])
      c010 <- count_cfg(slices[["010"]])
      c011 <- count_cfg(slices[["011"]])
      c100 <- count_cfg(slices[["100"]])
      c101 <- count_cfg(slices[["101"]])
      c110 <- count_cfg(slices[["110"]])
      c111 <- count_cfg(slices[["111"]])

      summary_df <- data.frame(
        triple = cache$triple,
        feature1 = cache$feature1,
        feature2 = cache$feature2,
        feature3 = cache$feature3,
        j = cache$j,
        k = cache$k,
        l = cache$l,
        n_alloc001 = c001["n_alloc"],
        n_allowed001 = c001["n_allowed"],
        n_pruned001 = c001["n_pruned"],
        n_alloc010 = c010["n_alloc"],
        n_allowed010 = c010["n_allowed"],
        n_pruned010 = c010["n_pruned"],
        n_alloc011 = c011["n_alloc"],
        n_allowed011 = c011["n_allowed"],
        n_pruned011 = c011["n_pruned"],
        n_alloc100 = c100["n_alloc"],
        n_allowed100 = c100["n_allowed"],
        n_pruned100 = c100["n_pruned"],
        n_alloc101 = c101["n_alloc"],
        n_allowed101 = c101["n_allowed"],
        n_pruned101 = c101["n_pruned"],
        n_alloc110 = c110["n_alloc"],
        n_allowed110 = c110["n_allowed"],
        n_pruned110 = c110["n_pruned"],
        n_alloc111 = c111["n_alloc"],
        n_allowed111 = c111["n_allowed"],
        n_pruned111 = c111["n_pruned"],
        n_eff_lt_thresh = sum(states_df$eff_n < eff_n_thresh),
        n_ratio_lt_thresh = sum(states_df$ratio_active & states_df$eff_over_alpha < eff_over_alpha_thresh),
        n_w_lt_1e6 = sum(states_df$cond_weight < 1e-6),
        stringsAsFactors = FALSE
      )
    }

    list(
      triple = cache$triple,
      feature1 = cache$feature1,
      feature2 = cache$feature2,
      feature3 = cache$feature3,
      j = cache$j,
      k = cache$k,
      l = cache$l,
      allowed = allowed,
      states = states_df,
      summary = summary_df
    )
  }

  triple_res <- if (workers_triple > 1L && identical(.Platform$OS.type, "unix")) {
    parallel::mclapply(triple_cache, triple_fun, mc.cores = workers_triple)
  } else {
    lapply(triple_cache, triple_fun)
  }

  triple_cache_full <- Filter(function(cache) !is.null(cache$log_alloc_mat_full), triple_cache)
  order3_cache <- setNames(
    lapply(triple_cache_full, function(cache) {
      list(log_alloc_mat = cache$log_alloc_mat_full)
    }),
    vapply(triple_cache_full, `[[`, character(1), "triple_key")
  )
  triple_summary <- data.frame()
  triple_states <- data.frame()
  triple_edge <- list(states = data.frame(), summary = data.frame())
  if (isTRUE(keep_diagnostics)) {
    triple_summary <- do.call(rbind, lapply(triple_res, `[[`, "summary"))
    triple_states <- do.call(rbind, lapply(triple_res, `[[`, "states"))
    triple_edge <- .apch_build_triple_edge_weight_tables(triple_states)
  }

  list(
    triples = lapply(triple_res, function(x) {
      x[c("triple", "feature1", "feature2", "feature3", "j", "k", "l", "allowed")]
    }),
    order3_cache = order3_cache,
    triple_prune_configs = triple_prune_configs,
    triple_summary = triple_summary,
    triple_states = triple_states,
    triple_edge_states = triple_edge$states,
    triple_edge_summary = triple_edge$summary
  )
}

.apch_build_quad_alloc_loglik_cache <- function(
    Bhat,
    S,
    C_mode,
    C_shared,
    C_list,
    marginal_res,
    pairwise_alloc_support,
    workers,
    verbose
) {
  feature_names <- .apch_feature_names(Bhat)
  p <- ncol(Bhat)
  if (p < 4L) {
    return(list())
  }
  quad_idx <- utils::combn(seq_len(p), 4L, simplify = FALSE)
  marginal_inputs <- .apch_prepare_marginal_inputs(
    slab_list = marginal_res$slab_list,
    pi_list = marginal_res$pi_list,
    p = p
  )
  full_grid_list <- marginal_inputs$full_grid_list
  component_loglik_cache <- .apch_make_component_loglik_cache(
    order1 = .apch_take_marginal_loglik_cache(
      marginal_res = marginal_res,
      full_grid_list = full_grid_list,
      n = nrow(Bhat)
    ),
    order2 = pairwise_alloc_support$loglik_cache$order2,
    order3 = pairwise_alloc_support$loglik_cache$order3
  )
  target_order4_keys <- .apch_joint_target_order4_keys(
    C_mode = C_mode,
    C_shared = C_shared,
    C_list = C_list,
    p = p
  )
  workers_density <- max(1L, min(as.integer(workers), 12L))
  use_parallel_density <- workers_density > 1L

  RcppParallel::setThreadOptions(
    numThreads = if (use_parallel_density) workers_density else 1L
  )

  lapply(seq_along(quad_idx), function(ii) {
    idx_quad <- as.integer(quad_idx[[ii]])
    quad_key <- .apch_quad_loglik_cache_key(idx_quad[1L], idx_quad[2L], idx_quad[3L], idx_quad[4L])
    keep_full_for_joint <- quad_key %in% target_order4_keys
    Kj_quad <- vapply(full_grid_list[idx_quad], length, integer(1))
    alloc_core <- .apch_alloc_core(Kj_quad)
    alloc_mat <- .apch_prune_alloc_mat_by_pair_support_local(
      alloc_mat = alloc_core$alloc_mat,
      idx = idx_quad,
      pairwise_alloc_support = pairwise_alloc_support
    )
    alloc_mat <- .apch_prune_alloc_mat_by_triple_support_local(
      alloc_mat = alloc_mat,
      idx = idx_quad,
      pairwise_alloc_support = pairwise_alloc_support
    )
    alloc_code <- .apch_alloc_code_from_mat(alloc_mat)

    noise_quad <- .apch_subset_C(C_mode, C_shared, C_list, idx_quad)
    noise_blocks <- .apch_noise_components(
      C_mode = noise_quad$C_mode,
      C_shared = noise_quad$C_shared,
      C_list = noise_quad$C_list,
      p = 4L
    )
    if (length(noise_blocks) == 4L) {
      assembly_global <- .apch_singleton_assembly_blocks(idx_quad, component_loglik_cache)
      assembly_blocks <- lapply(assembly_global, function(block_global) {
        as.integer(match(as.integer(block_global), idx_quad))
      })
    } else {
      assembly_blocks <- noise_blocks
    }
    block_logs <- lapply(assembly_blocks, function(block_local) {
      block_local <- as.integer(block_local)
      block_global <- idx_quad[block_local]
      cache_block <- .apch_get_component_loglik_cache(
        component_loglik_cache = component_loglik_cache,
        idx = block_global
      )
      if (length(cache_block) > 0L) {
        return(cache_block)
      }
      if (length(block_local) == 1L) {
        return(.apch_build_block_univariate(
          Bhat = Bhat,
          idx = block_global,
          slab_list = marginal_inputs$slab_list,
          pi_list = marginal_inputs$pi_list,
          C_mode = C_mode,
          C_shared = C_shared,
          C_list = C_list
        )$log_alloc_mat)
      }
      .apch_build_block_multivariate_log_alloc_mat(
        Bhat_b = Bhat[, idx_quad[block_local], drop = FALSE],
        Sb = S[, idx_quad[block_local], drop = FALSE],
        idx = block_local,
        full_grid_list = full_grid_list[idx_quad],
        C_mode = noise_quad$C_mode,
        C_shared = noise_quad$C_shared,
        C_list = noise_quad$C_list,
        use_parallel = use_parallel_density
      )$log_alloc_mat
    })
    log_alloc_mat_full <- NULL
    if (keep_full_for_joint) {
      log_alloc_mat_full <- apch_combine_log_alloc_blocks_cpp(
        log_alloc_blocks = block_logs,
        blocks = assembly_blocks,
        Kvec = as.integer(Kj_quad),
        alloc_mat = alloc_core$alloc_mat
      )
      if (nrow(alloc_mat) == nrow(alloc_core$alloc_mat)) {
        log_alloc_mat <- log_alloc_mat_full
      } else {
        state_full <- apply(alloc_core$alloc_mat, 1L, paste, collapse = ",")
        state_keep <- apply(alloc_mat, 1L, paste, collapse = ",")
        keep_idx <- match(state_keep, state_full)
        log_alloc_mat <- log_alloc_mat_full[, keep_idx, drop = FALSE]
      }
    } else {
      log_alloc_mat <- apch_combine_log_alloc_blocks_cpp(
        log_alloc_blocks = block_logs,
        blocks = assembly_blocks,
        Kvec = as.integer(Kj_quad),
        alloc_mat = alloc_mat
      )
    }

    list(
      quad = paste(feature_names[idx_quad], collapse = "__"),
      quad_key = quad_key,
      feature1 = feature_names[idx_quad[1L]],
      feature2 = feature_names[idx_quad[2L]],
      feature3 = feature_names[idx_quad[3L]],
      feature4 = feature_names[idx_quad[4L]],
      j = idx_quad[1L],
      k = idx_quad[2L],
      l = idx_quad[3L],
      m = idx_quad[4L],
      alloc_mat = alloc_mat,
      alloc_code = alloc_code,
      log_alloc_mat = log_alloc_mat,
      log_alloc_mat_full = log_alloc_mat_full
    )
  })
}

.apch_build_quad_alloc_support_from_cache <- function(
    quad_cache,
    marginal_res,
    eff_n_thresh,
    eff_over_alpha_thresh,
    dirichlet_count,
    workers,
    keep_diagnostics = FALSE,
    verbose
) {
  quad_prune_configs <- vapply(seq_len(15L), function(code) {
    .apch_code_to_config01(code, 4L)
  }, character(1))
  prune_codes_quad <- strtoi(quad_prune_configs, base = 2L)
  prior_alpha_min <- 1e-4
  prior_alpha_cap <- 5
  prior_alpha_cap_null <- 10
  cond_weight_thresh <- 1e-6
  if (length(quad_cache) == 0L) {
    return(list(
      quads = list(),
      order4_cache = list(),
      quad_prune_configs = quad_prune_configs,
      quad_summary = data.frame(),
      quad_states = data.frame()
    ))
  }
  workers_quad <- max(1L, min(length(quad_cache), as.integer(workers)))

  quad_fun <- function(cache) {
    idx_quad <- c(cache$j, cache$k, cache$l, cache$m)
    pi_block <- marginal_res$pi_list[idx_quad]
    pnull_block <- marginal_res$p_null[idx_quad]
    use_prior <- (dirichlet_count > 0)
    prior_q <- if (use_prior) {
      .apch_alloc_prior_center_from_marginals(
        alloc_mat = cache$alloc_mat,
        p_null = pnull_block,
        pi_list = pi_block
      )
    } else {
      NULL
    }
    alloc_fit_state <- .apch_build_apchs_alloc_logw_cond_from_log_alloc(
      log_alloc_mat = cache$log_alloc_mat,
      alloc_mat = cache$alloc_mat,
      alloc_code = cache$alloc_code,
      idx = 1:4,
      pi_list = pi_block,
      prior_q = prior_q,
      prior_frac = dirichlet_count,
      prior_alpha_min = prior_alpha_min,
      prior_alpha_cap = prior_alpha_cap,
      prior_alpha_cap_null = prior_alpha_cap_null,
      keep_alloc_post_colsum = TRUE,
      blas_threads = 1L
    )
    Kvec <- vapply(pi_block, length, integer(1))
    eval_res <- .apch_local_prune_eval(
      cache = cache,
      alloc_fit_state = alloc_fit_state,
      prior_q = prior_q,
      use_prior = use_prior,
      dirichlet_count = dirichlet_count,
      prior_alpha_min = prior_alpha_min,
      prior_alpha_cap = prior_alpha_cap,
      prior_alpha_cap_null = prior_alpha_cap_null,
      eff_n_thresh = eff_n_thresh,
      eff_over_alpha_thresh = eff_over_alpha_thresh,
      cond_weight_thresh = cond_weight_thresh,
      prune_codes = prune_codes_quad
    )
    slices <- .apch_local_state_slices(
      alloc_mat = cache$alloc_mat,
      alloc_code = cache$alloc_code,
      prune_configs = quad_prune_configs,
      eval_res = eval_res,
      prior_q = prior_q,
      use_prior = use_prior,
      keep_diagnostics = keep_diagnostics
    )
    allowed <- .apch_build_allowed_tensors_from_slices(
      slices = slices,
      Kvec = Kvec
    )

    states_df <- NULL
    summary_df <- NULL
    count_cfg <- function(x) c(n_alloc = length(x$allowed), n_allowed = sum(x$allowed), n_pruned = sum(!x$allowed))
    summary_list <- list(
      quad = cache$quad,
      feature1 = cache$feature1,
      feature2 = cache$feature2,
      feature3 = cache$feature3,
      feature4 = cache$feature4,
      j = cache$j,
      k = cache$k,
      l = cache$l,
      m = cache$m
    )
    if (isTRUE(keep_diagnostics)) {
      states_df <- do.call(rbind, lapply(slices, `[[`, "states"))
      states_df <- data.frame(
        quad = rep(cache$quad, nrow(states_df)),
        feature1 = rep(cache$feature1, nrow(states_df)),
        feature2 = rep(cache$feature2, nrow(states_df)),
        feature3 = rep(cache$feature3, nrow(states_df)),
        feature4 = rep(cache$feature4, nrow(states_df)),
        j = rep(cache$j, nrow(states_df)),
        k = rep(cache$k, nrow(states_df)),
        l = rep(cache$l, nrow(states_df)),
        m = rep(cache$m, nrow(states_df)),
        states_df,
        stringsAsFactors = FALSE
      )

      for (cfg in quad_prune_configs) {
        counts <- count_cfg(slices[[cfg]])
        summary_list[[paste0("n_alloc", cfg)]] <- counts["n_alloc"]
        summary_list[[paste0("n_allowed", cfg)]] <- counts["n_allowed"]
        summary_list[[paste0("n_pruned", cfg)]] <- counts["n_pruned"]
      }
      summary_list[["n_eff_lt_thresh"]] <- sum(states_df$eff_n < eff_n_thresh)
      summary_list[["n_ratio_lt_thresh"]] <- sum(states_df$ratio_active & states_df$eff_over_alpha < eff_over_alpha_thresh)
      summary_list[["n_w_lt_1e6"]] <- sum(states_df$cond_weight < 1e-6)
      summary_df <- as.data.frame(summary_list, stringsAsFactors = FALSE, check.names = FALSE)
    }

    list(
      quad = cache$quad,
      feature1 = cache$feature1,
      feature2 = cache$feature2,
      feature3 = cache$feature3,
      feature4 = cache$feature4,
      j = cache$j,
      k = cache$k,
      l = cache$l,
      m = cache$m,
      allowed = allowed,
      states = states_df,
      summary = summary_df
    )
  }

  quad_res <- if (workers_quad > 1L && identical(.Platform$OS.type, "unix")) {
    parallel::mclapply(quad_cache, quad_fun, mc.cores = workers_quad)
  } else {
    lapply(quad_cache, quad_fun)
  }

  quad_cache_full <- Filter(function(cache) !is.null(cache$log_alloc_mat_full), quad_cache)
  order4_cache <- setNames(
    lapply(quad_cache_full, function(cache) {
      list(log_alloc_mat = cache$log_alloc_mat_full)
    }),
    vapply(quad_cache_full, `[[`, character(1), "quad_key")
  )
  quad_summary <- data.frame()
  quad_states <- data.frame()
  if (isTRUE(keep_diagnostics)) {
    quad_summary <- do.call(rbind, lapply(quad_res, `[[`, "summary"))
    quad_states <- do.call(rbind, lapply(quad_res, `[[`, "states"))
  }

  list(
    quads = lapply(quad_res, function(x) {
      x[c("quad", "feature1", "feature2", "feature3", "feature4", "j", "k", "l", "m", "allowed")]
    }),
    order4_cache = order4_cache,
    quad_prune_configs = quad_prune_configs,
    quad_summary = quad_summary,
    quad_states = quad_states
  )
}

.apch_build_pairwise_alloc_loglik_cache <- function(
    Bhat,
    S,
    C_mode,
    C_shared,
    C_list,
    marginal_res,
    workers,
    verbose
) {
  feature_names <- .apch_feature_names(Bhat)
  p <- ncol(Bhat)
  pair_idx <- utils::combn(seq_len(p), 2L, simplify = FALSE)
  marginal_inputs <- .apch_prepare_marginal_inputs(
    slab_list = marginal_res$slab_list,
    pi_list = marginal_res$pi_list,
    p = p
  )
  full_grid_list <- marginal_inputs$full_grid_list
  shared_comp_id <- .apch_pairwise_indep_shared_comp_id(C_mode, C_shared, C_list, p)
  pair_is_indep <- vapply(pair_idx, function(idx_pair) {
    .apch_pairwise_is_independent(
      C_mode = C_mode,
      C_shared = C_shared,
      C_list = C_list,
      j = idx_pair[1L],
      k = idx_pair[2L],
      shared_comp_id = shared_comp_id
    )
  }, logical(1))
  use_indep_fast_path <- any(pair_is_indep)
  cache_from_marginal <- FALSE
  uni_loglik_cache <- if (use_indep_fast_path) {
    cache0 <- .apch_take_marginal_loglik_cache(
      marginal_res = marginal_res,
      full_grid_list = full_grid_list,
      n = nrow(Bhat)
    )
    if (!is.null(cache0)) {
      cache_from_marginal <- TRUE
      cache0
    } else {
      .apch_build_pairwise_univariate_loglik_cache(
        Bhat = Bhat,
        S = S,
        C_mode = C_mode,
        C_shared = C_shared,
        C_list = C_list,
        marginal_inputs = marginal_inputs
      )
    }
  } else {
    NULL
  }
  workers_density <- max(1L, min(as.integer(workers), 12L))
  use_parallel_density <- workers_density > 1L

  RcppParallel::setThreadOptions(
    numThreads = if (use_parallel_density) workers_density else 1L
  )

  lapply(seq_along(pair_idx), function(ii) {
    idx_pair <- pair_idx[[ii]]
    j <- idx_pair[1L]
    k <- idx_pair[2L]
    alloc_state <- if (use_indep_fast_path && pair_is_indep[ii]) {
      .apch_build_pairwise_indep_loglik_cache(
        log_alloc_j = uni_loglik_cache[[j]],
        log_alloc_k = uni_loglik_cache[[k]],
        Kj_pair = vapply(full_grid_list[idx_pair], length, integer(1))
      )
    } else {
      noise_pair <- .apch_subset_C(C_mode, C_shared, C_list, idx_pair)
      .apch_alloc_loglik_joint(
        Bhat = Bhat[, idx_pair, drop = FALSE],
        S = S[, idx_pair, drop = FALSE],
        full_grid_list = full_grid_list[idx_pair],
        slab_list = marginal_inputs$slab_list[idx_pair],
        pi_list = marginal_inputs$pi_list[idx_pair],
        C_mode = noise_pair$C_mode,
        C_shared = noise_pair$C_shared,
        C_list = noise_pair$C_list,
        pairwise_alloc_support = NULL,
        marginal_loglik_cache = NULL,
        use_parallel = use_parallel_density,
        verbose = FALSE
      )
    }
    list(
      pair = paste(feature_names[j], feature_names[k], sep = "__"),
      pair_key = .apch_pair_loglik_cache_key(j, k),
      feature1 = feature_names[j],
      feature2 = feature_names[k],
      j = j,
      k = k,
      indep_fast = isTRUE(use_indep_fast_path && pair_is_indep[ii]),
      alloc_mat = alloc_state$alloc_mat,
      alloc_code = alloc_state$alloc_code,
      log_alloc_mat = alloc_state$log_alloc_mat
    )
  })
}

.apch_build_pairwise_alloc_support_from_cache <- function(
    pair_cache,
    marginal_res,
    eff_n_thresh = 1,
    eff_over_alpha_thresh = 10,
    dirichlet_count,
    workers,
    keep_diagnostics = FALSE,
    verbose
) {
  pairwise_prune_configs <- c("01", "10", "11")
  prune_codes_pair <- strtoi(pairwise_prune_configs, base = 2L)
  prior_alpha_min <- 1e-4
  prior_alpha_cap <- 5
  prior_alpha_cap_null <- 10
  cond_weight_thresh <- 1e-6
  workers_pair <- max(1L, min(length(pair_cache), as.integer(workers)))

  pair_fun <- function(cache) {
    idx_pair <- c(cache$j, cache$k)
    pi_block <- marginal_res$pi_list[idx_pair]
    pnull_block <- marginal_res$p_null[idx_pair]
    use_prior <- (dirichlet_count > 0)
    prior_q <- if (use_prior) {
      .apch_alloc_prior_center_from_marginals(
        alloc_mat = cache$alloc_mat,
        p_null = pnull_block,
        pi_list = pi_block
      )
    } else {
      NULL
    }
    alloc_fit_state <- .apch_build_apchs_alloc_logw_cond_from_log_alloc(
      log_alloc_mat = cache$log_alloc_mat,
      alloc_mat = cache$alloc_mat,
      alloc_code = cache$alloc_code,
      idx = 1:2,
      pi_list = pi_block,
      prior_q = prior_q,
      prior_frac = dirichlet_count,
      prior_alpha_min = prior_alpha_min,
      prior_alpha_cap = prior_alpha_cap,
      prior_alpha_cap_null = prior_alpha_cap_null,
      keep_alloc_post_colsum = TRUE,
      blas_threads = 1L
    )
    eval_res <- .apch_local_prune_eval(
      cache = cache,
      alloc_fit_state = alloc_fit_state,
      prior_q = prior_q,
      use_prior = use_prior,
      dirichlet_count = dirichlet_count,
      prior_alpha_min = prior_alpha_min,
      prior_alpha_cap = prior_alpha_cap,
      prior_alpha_cap_null = prior_alpha_cap_null,
      eff_n_thresh = eff_n_thresh,
      eff_over_alpha_thresh = eff_over_alpha_thresh,
      cond_weight_thresh = cond_weight_thresh,
      prune_codes = prune_codes_pair
    )
    slices <- .apch_local_state_slices(
      alloc_mat = cache$alloc_mat,
      alloc_code = cache$alloc_code,
      prune_configs = pairwise_prune_configs,
      eval_res = eval_res,
      prior_q = prior_q,
      use_prior = use_prior,
      keep_diagnostics = keep_diagnostics
    )
    K1 <- length(pi_block[[1L]])
    K2 <- length(pi_block[[2L]])
    allowed <- .apch_build_allowed_tensors_from_slices(
      slices = slices,
      Kvec = c(K1, K2)
    )

    states_df <- NULL
    summary_df <- NULL
    if (isTRUE(keep_diagnostics)) {
      states01 <- slices[["01"]]
      states10 <- slices[["10"]]
      states11 <- slices[["11"]]
      states_df <- rbind(states01$states, states10$states, states11$states)
      states_df <- data.frame(
        pair = cache$pair,
        feature1 = cache$feature1,
        feature2 = cache$feature2,
        j = cache$j,
        k = cache$k,
        states_df,
        alloc11 = ifelse(states_df$config01 == "11", states_df$alloc_state, NA_character_),
        cond_weight11 = ifelse(states_df$config01 == "11", states_df$cond_weight, NA_real_),
        stringsAsFactors = FALSE
      )

      summary_df <- data.frame(
        pair = cache$pair,
        feature1 = cache$feature1,
        feature2 = cache$feature2,
        j = cache$j,
        k = cache$k,
        pairwise_prune_method = "safe_or_tinyw_and_eff_ratio",
        n_alloc01 = length(states01$allowed),
        n_alloc10 = length(states10$allowed),
        n_alloc11 = length(states11$allowed),
        n_allowed01 = sum(states01$allowed),
        n_allowed10 = sum(states10$allowed),
        n_allowed11 = sum(states11$allowed),
        n_pruned01 = sum(!states01$allowed),
        n_pruned10 = sum(!states10$allowed),
        n_pruned11 = sum(!states11$allowed),
        n_eff_lt_thresh = sum(states_df$eff_n < eff_n_thresh),
        n_ratio_lt_thresh = sum(states_df$ratio_active & states_df$eff_over_alpha < eff_over_alpha_thresh),
        n_w_lt_1e6 = sum(states_df$cond_weight < 1e-6),
        stringsAsFactors = FALSE
      )
    }

    list(
      pair = cache$pair,
      feature1 = cache$feature1,
      feature2 = cache$feature2,
      j = cache$j,
      k = cache$k,
      allowed = allowed,
      states = states_df,
      summary = summary_df
    )
  }

  pair_res <- if (workers_pair > 1L && identical(.Platform$OS.type, "unix")) {
    parallel::mclapply(pair_cache, pair_fun, mc.cores = workers_pair)
  } else {
    lapply(pair_cache, pair_fun)
  }
  feature_names_res <- unique(c(
    vapply(pair_res, `[[`, character(1), "feature1"),
    vapply(pair_res, `[[`, character(1), "feature2")
  ))

  order2_cache <- setNames(
    lapply(pair_cache, function(cache) {
      list(log_alloc_mat = cache$log_alloc_mat)
    }),
    vapply(pair_cache, `[[`, character(1), "pair_key")
  )
  pair_summary <- data.frame()
  pair_states <- data.frame()
  pair_edge <- list(states = data.frame(), summary = data.frame())
  if (isTRUE(keep_diagnostics)) {
    pair_summary <- do.call(rbind, lapply(pair_res, `[[`, "summary"))
    pair_states <- do.call(rbind, lapply(pair_res, `[[`, "states"))
    pair_edge <- .apch_build_pair_edge_weight_tables(pair_states)
  }

  structure(
    list(
      feature_names = feature_names_res,
      eff_n_thresh = eff_n_thresh,
      eff_over_alpha_thresh = eff_over_alpha_thresh,
      pairwise_prune_method = "safe_or_tinyw_and_eff_ratio",
      pairwise_prune_configs = pairwise_prune_configs,
      triple_prune_configs = character(0),
      quad_prune_configs = character(0),
      pairs = lapply(pair_res, function(x) {
        x[c("pair", "feature1", "feature2", "j", "k", "allowed")]
      }),
      triples = list(),
      quads = list(),
      loglik_cache = .apch_make_component_loglik_cache(order2 = order2_cache),
      pair_summary = pair_summary,
      pair_states = pair_states,
      pair_edge_states = pair_edge$states,
      pair_edge_summary = pair_edge$summary,
      triple_summary = data.frame(),
      triple_states = data.frame(),
      triple_edge_states = data.frame(),
      triple_edge_summary = data.frame(),
      quad_summary = data.frame(),
      quad_states = data.frame(),
      edge_weight_summary = pair_edge$summary
    ),
    class = "apch_pairwise_alloc_support"
  )
}

.apch_build_pairwise_alloc_support_prepared <- function(
    Bhat,
    S,
    C_mode,
    C_shared,
    C_list,
    marginal_res,
    dirichlet_count,
    eff_n_thresh = 1,
    eff_over_alpha_thresh = 10,
    workers,
    keep_diagnostics = FALSE,
    enable_quad = FALSE,
    verbose
) {
  feature_names <- .apch_feature_names(Bhat)
  p <- ncol(Bhat)
  if (p <= 3L) {
    return(.apch_empty_alloc_support(feature_names))
  }
  marginal_res <- marginal_res[c("slab_list", "p_null", "pi_list", "loglik_cache")]
  pair_cache <- .apch_build_pairwise_alloc_loglik_cache(
    Bhat = Bhat,
    S = S,
    C_mode = C_mode,
    C_shared = C_shared,
    C_list = C_list,
    marginal_res = marginal_res,
    workers = workers,
    verbose = verbose
  )
  support <- .apch_build_pairwise_alloc_support_from_cache(
    pair_cache = pair_cache,
    marginal_res = marginal_res,
    eff_n_thresh = eff_n_thresh,
    eff_over_alpha_thresh = eff_over_alpha_thresh,
    dirichlet_count = dirichlet_count,
    workers = workers,
    keep_diagnostics = keep_diagnostics,
    verbose = verbose
  )
  if (p >= 5L) {
    triple_cache <- .apch_build_triple_alloc_loglik_cache(
      Bhat = Bhat,
      S = S,
      C_mode = C_mode,
      C_shared = C_shared,
      C_list = C_list,
      marginal_res = marginal_res,
      pairwise_alloc_support = support,
      workers = workers,
      verbose = verbose
    )
    triple_support <- .apch_build_triple_alloc_support_from_cache(
      triple_cache = triple_cache,
      marginal_res = marginal_res,
      eff_n_thresh = eff_n_thresh,
      eff_over_alpha_thresh = eff_over_alpha_thresh,
      dirichlet_count = dirichlet_count,
      workers = workers,
      keep_diagnostics = keep_diagnostics,
      verbose = verbose
    )
    support$triples <- triple_support$triples
    support$triple_prune_configs <- triple_support$triple_prune_configs
    support$triple_summary <- triple_support$triple_summary
    support$triple_states <- triple_support$triple_states
    support$triple_edge_states <- triple_support$triple_edge_states
    support$triple_edge_summary <- triple_support$triple_edge_summary
    support$loglik_cache$order3 <- triple_support$order3_cache
  } else {
    support$triples <- list()
    support$triple_prune_configs <- character(0)
    support$triple_summary <- data.frame()
    support$triple_states <- data.frame()
    support$triple_edge_states <- data.frame()
    support$triple_edge_summary <- data.frame()
    support$loglik_cache$order3 <- list()
  }
  if (isTRUE(enable_quad) && ncol(Bhat) >= 4L) {
    quad_cache <- .apch_build_quad_alloc_loglik_cache(
      Bhat = Bhat,
      S = S,
      C_mode = C_mode,
      C_shared = C_shared,
      C_list = C_list,
      marginal_res = marginal_res,
      pairwise_alloc_support = support,
      workers = workers,
      verbose = verbose
    )
    quad_support <- .apch_build_quad_alloc_support_from_cache(
      quad_cache = quad_cache,
      marginal_res = marginal_res,
      eff_n_thresh = eff_n_thresh,
      eff_over_alpha_thresh = eff_over_alpha_thresh,
      dirichlet_count = dirichlet_count,
      workers = workers,
      keep_diagnostics = keep_diagnostics,
      verbose = verbose
    )
    support$quads <- quad_support$quads
    support$quad_prune_configs <- quad_support$quad_prune_configs
    support$quad_summary <- quad_support$quad_summary
    support$quad_states <- quad_support$quad_states
    support$loglik_cache$order4 <- quad_support$order4_cache
  } else {
    support$quads <- list()
    support$quad_prune_configs <- character(0)
    support$quad_summary <- data.frame()
    support$quad_states <- data.frame()
    support$loglik_cache$order4 <- list()
  }
  if (isTRUE(keep_diagnostics)) {
    edge_parts <- Filter(function(x) nrow(x) > 0L, list(
      support$pair_edge_summary,
      support$triple_edge_summary
    ))
    support$edge_weight_summary <- if (length(edge_parts) == 0L) data.frame() else do.call(rbind, edge_parts)
  } else {
    support$edge_weight_summary <- data.frame()
  }
  support$loglik_cache <- .apch_trim_component_loglik_cache_for_joint(
    component_loglik_cache = support$loglik_cache,
    C_mode = C_mode,
    C_shared = C_shared,
    C_list = C_list,
    p = ncol(Bhat)
  )
  support$feature_names <- feature_names
  support
}

.apch_alloc_code_from_mat <- function(alloc_mat) {
  .apch_config_code((alloc_mat > 0L) + 0L)
}

.apch_validate_pairwise_alloc_support <- function(pairwise_alloc_support, feature_names) {
  if (is.null(pairwise_alloc_support)) {
    return(NULL)
  }
  support_names <- pairwise_alloc_support$feature_names
  if (!is.null(support_names) && length(support_names) == length(feature_names)) {
    if (!identical(as.character(support_names), as.character(feature_names))) {
      stop("pairwise_alloc_support feature_names do not match the current feature order.")
    }
  }
  pairwise_alloc_support
}

.apch_prune_alloc_mat_by_pair_support <- function(alloc_mat, pairwise_alloc_support) {
  if (is.null(pairwise_alloc_support)) {
    return(list(
      alloc_mat = alloc_mat,
      n_init = nrow(alloc_mat),
      n_after_pair = nrow(alloc_mat),
      n_after_triple = nrow(alloc_mat),
      n_after_quad = nrow(alloc_mat)
    ))
  }

  n_init <- nrow(alloc_mat)
  pairwise_prune_configs <- pairwise_alloc_support$pairwise_prune_configs
  for (pr in pairwise_alloc_support$pairs) {
    alloc_mat <- .apch_prune_alloc_mat_by_support_item(
      alloc_mat = alloc_mat,
      idx = c(pr$j, pr$k),
      allowed = pr$allowed,
      prune_configs = pairwise_prune_configs
    )
  }
  n_after_pair <- nrow(alloc_mat)
  if (length(pairwise_alloc_support$triples) > 0L) {
    triple_prune_configs <- pairwise_alloc_support$triple_prune_configs
    for (tr in pairwise_alloc_support$triples) {
      alloc_mat <- .apch_prune_alloc_mat_by_support_item(
        alloc_mat = alloc_mat,
        idx = c(tr$j, tr$k, tr$l),
        allowed = tr$allowed,
        prune_configs = triple_prune_configs
      )
    }
  }
  n_after_triple <- nrow(alloc_mat)
  if (length(pairwise_alloc_support$quads) > 0L) {
    quad_prune_configs <- pairwise_alloc_support$quad_prune_configs
    for (qd in pairwise_alloc_support$quads) {
      alloc_mat <- .apch_prune_alloc_mat_by_support_item(
        alloc_mat = alloc_mat,
        idx = c(qd$j, qd$k, qd$l, qd$m),
        allowed = qd$allowed,
        prune_configs = quad_prune_configs
      )
    }
  }
  list(
    alloc_mat = alloc_mat,
    n_init = n_init,
    n_after_pair = n_after_pair,
    n_after_triple = n_after_triple,
    n_after_quad = nrow(alloc_mat)
  )
}
