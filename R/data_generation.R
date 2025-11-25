# ============================== Simulation functions (formal API) ==============================

# Internal: default δ (only effective when p=3; order: 000, 100, 010, 001, 110, 101, 011, 111)
DEFAULT_DELTA_P3 <- c(0.82, 0.03, 0.03, 0.03, 0.025, 0.025, 0.025, 0.015)

# Internal: generate configuration weights δ
# If 'preset' is provided (length = 2^p) → use directly after normalization;
# If p == 3 and 'preset' is not provided → use DEFAULT_DELTA_P3;
# For other p values → automatically generate using hierarchical base_w.
make_delta_true <- function(p, preset = NULL,
                            base_w = c(0.05, 0.02, 0.01),
                            delta0 = 0.5) {
  if (!is.null(preset)) {
    stopifnot(length(preset) == 2^p)
    d <- as.numeric(preset)
    return(d / sum(d))
  } else if (p == 3) {
    d <- DEFAULT_DELTA_P3
    return(d / sum(d))
  } else {
    max_base_k <- length(base_w)
    raw_w <- numeric(p)
    for (k in 1:p) {
      raw_w[k] <- if (k <= max_base_k) base_w[k] else raw_w[k - 1] / 2
    }
    total_raw_mass <- sum(choose(p, 1:p) * raw_w)

    configs <- generate_all_configs(p)
    delta_true <- numeric(nrow(configs))
    for (i in seq_len(nrow(configs))) {
      k <- sum(configs[i, ])
      delta_true[i] <- if (k == 0) {
        delta0
      } else {
        raw_w[k] / total_raw_mass * (1 - delta0)
      }
    }
    return(delta_true / sum(delta_true))
  }
}

# Internal: mixture scenarios used in data_generation_1
get_scenario_params <- function(scenario) {
  if (scenario == "spiky") {
    list(
      weights = c(0.2, 0.4, 0.2, 0.2),
      means   = c(0, 0, 0, 0),
      sds     = c(0.25, 0.5, 1, 2)
    )
  } else if (scenario == "near_normal") {
    list(
      weights = c(2 / 3, 1 / 3),
      means   = c(0, 0),
      sds     = c(1, 2)
    )
  } else if (scenario == "skew") {
    list(
      weights = c(1 / 4, 1 / 4, 1 / 3, 1 / 6),
      means   = c(-2, -1, 0, 1),
      sds     = c(2, 1.5, 1, 1)
    )
  } else if (scenario == "flattop") {
    list(
      weights = rep(1 / 7, 7),
      means   = c(-1.5, -1, -0.5, 0, 0.5, 1, 1.5),
      sds     = rep(0.5, 7)
    )
  } else if (scenario == "big_normal") {
    list(
      weights = 1,
      means   = 0,
      sds     = 4
    )
  } else if (scenario == "bimodal") {
    list(
      weights = c(0.5, 0.5),
      means   = c(-3, 1),
      sds     = c(1, 1)
    )
  } else if (scenario %in% c("std_normal1", "std_normal2")) {
    s <- if (scenario == "std_normal1") 1 else 2
    list(weights = 1, means = 0, sds = s)
  } else if (scenario == "two_component") {
    list(
      weights = c(0.7, 0.3),
      means   = c(0, 0),
      sds     = c(1, 2)
    )
  } else {
    stop(paste("unknown scenario:", scenario))
  }
}

#' Simulate data from the APCH model (NCP parameterization)
#'
#' \code{data_generation_1()} simulates data under the APCH configuration
#' model with a non-centrality (NCP) parameterization. It generates a set of
#' latent configurations, effect sizes, and observed statistics with correlated
#' noise.
#'
#' This function is a direct, self-contained simulator corresponding to the
#' model used in the methodological development.
#'
#' @param n Integer; number of effects (rows). Default \code{15000}.
#' @param p Integer; number of features (columns).
#' @param delta Optional numeric vector of length \code{2^p} giving the
#'   configuration weights. If \code{NULL}, a default hierarchical scheme
#'   is used (with a special case for \code{p = 3}).
#' @param distributions Character vector of length \code{p} specifying the
#'   marginal effect-size scenarios for each feature. Allowed values include
#'   \code{"spiky"}, \code{"near_normal"}, \code{"skew"}, \code{"flattop"},
#'   \code{"big_normal"}, \code{"bimodal"}, \code{"std_normal1"},
#'   \code{"std_normal2"}, and \code{"two_component"}.
#' @param lambda Numeric scalar or length-\code{p} vector giving the NCP scale
#'   parameters for each feature. A scalar is recycled to length \code{p}.
#' @param rho Numeric \code{p x p} correlation matrix for the observation noise.
#'
#' @return A list with components:
#' \describe{
#'   \item{configs}{A \code{2^p x p} binary matrix of all configurations used
#'     in the simulation.}
#'   \item{delta_used}{The normalized configuration weights actually used.}
#'   \item{gamma_true}{An \code{n x p} binary matrix of sampled configurations
#'     for each effect.}
#'   \item{lambda_used}{The length-\code{p} vector of NCP scales used.}
#'   \item{a_scale}{Length-\code{p} vector of scaling factors applied to each
#'     feature's mixture draws.}
#'   \item{B_true}{An \code{n x p} matrix of true underlying effects.}
#'   \item{B_hat}{An \code{n x p} matrix of observed effect estimates
#'     (true effects + noise).}
#'   \item{R}{The final \code{p x p} correlation matrix used for the noise
#'     (possibly nudged to be positive definite).}
#' }
#'
#' @export
data_generation_1 <- function(
    n = 15000,
    p,
    delta = NULL,
    distributions,
    lambda,
    rho
) {
  stopifnot(length(distributions) == p)
  if (is.null(delta)) delta <- make_delta_true(p)
  delta <- as.numeric(delta) / sum(delta)
  if (length(lambda) == 1L) lambda <- rep(lambda, p)

  # rho is a p x p correlation matrix
  stopifnot(is.matrix(rho), all(dim(rho) == c(p, p)))
  R <- 0.5 * (rho + t(rho))
  diag(R) <- 1
  ev <- eigen(R, symmetric = TRUE, only.values = TRUE)$values
  if (min(ev) <= 0) R <- R + diag(1e-6, p)

  # configurations
  configs <- generate_all_configs(p)
  cfg_idx <- sample.int(nrow(configs), n, replace = TRUE, prob = delta)
  gamma_true <- configs[cfg_idx, , drop = FALSE]

  # generate B_true (scaled by NCP)
  B_true <- matrix(0, n, p)
  a_vec  <- numeric(p)
  for (j in 1:p) {
    act <- which(gamma_true[, j] == 1)
    if (length(act)) {
      sc <- get_scenario_params(distributions[j])
      comp <- sample.int(length(sc$weights), length(act),
                         replace = TRUE, prob = sc$weights)
      u <- rnorm(length(act), mean = sc$means[comp], sd = sc$sds[comp])
      rms <- sqrt(mean(u^2))
      a_j <- if (rms > 0) lambda[j] / rms else 0
      B_true[act, j] <- a_j * u
      a_vec[j] <- a_j
    }
  }

  # observation noise ~ N(0, R)
  U <- chol(R)
  Eps <- matrix(rnorm(n * p), n, p) %*% U
  B_hat <- B_true + Eps

  list(
    configs     = configs,
    delta_used  = delta,
    gamma_true  = gamma_true,
    lambda_used = as.numeric(lambda),
    a_scale     = a_vec,
    B_true      = B_true,
    B_hat       = B_hat,
    R           = R
  )
}

# ============================== Second simulation design (single dataset, given L) ==============================

# Internal: flattop generator used in data_generation2
r_flattop <- function(n) {
  w  <- rep(1 / 7, 7)
  mu <- c(-1.5, -1, -0.5, 0, 0.5, 1, 1.5)
  sd <- rep(0.5, 7)
  comp <- sample.int(7, size = n, replace = TRUE, prob = w)
  rnorm(n, mean = mu[comp], sd = sd[comp])
}

# Internal: generate one (L) dataset with optional correlated noise
make_data_oneL <- function(
    m, K, m1, L, se,
    beta_star       = NULL,
    flip_prob       = 0.5,
    flip_within_L   = TRUE,
    use_corr_noise  = TRUE,
    rho_cor_noise   = 0.8
) {
  traits_lab <- paste0("trait", 1:K)
  snp_ids    <- paste0("snp",   1:m)

  if (is.null(beta_star)) beta_star <- r_flattop(m1)

  # start from independent noise
  betahat <- matrix(
    rnorm(m * K, 0, se),
    nrow = m, ncol = K,
    dimnames = list(snp_ids, traits_lab)
  )
  S <- matrix(se, nrow = m, ncol = K, dimnames = dimnames(betahat))

  # replace first L traits with equicorrelated noise if requested
  if (use_corr_noise && L > 0L) {
    # equicorrelated covariance: se^2 * ((1 - rho) I + rho * J)
    Sigma_L <- (se^2) * ((1 - rho_cor_noise) * diag(L) +
                           rho_cor_noise * matrix(1, L, L))
    Rchol   <- chol(Sigma_L)
    Z       <- matrix(rnorm(m * L), nrow = m, ncol = L)
    corr_noise <- Z %*% Rchol
    betahat[, 1:L] <- corr_noise
  }

  # add true effects for first L traits, with optional sign flipping
  if (L > 0L) {
    base_delta <- tcrossprod(beta_star, rep(1, L))  # m1 x L, each column beta_star

    if (flip_prob > 0) {
      if (isTRUE(flip_within_L)) {
        flip_mask_L <- matrix(runif(m1 * L) < flip_prob, nrow = m1, ncol = L)
        sign_L <- ifelse(flip_mask_L, -1, 1)
      } else {
        flip_mask_K <- matrix(runif(m1 * K) < flip_prob, nrow = m1, ncol = K)
        sign_L <- ifelse(flip_mask_K[, 1:L, drop = FALSE], -1, 1)
      }
      delta <- base_delta * sign_L
    } else {
      delta <- base_delta
    }

    betahat[1:m1, 1:L] <- betahat[1:m1, 1:L] + delta
  }

  list(
    betahat   = betahat,
    S         = S,
    beta_star = beta_star
  )
}

#' Single-dataset simulation with given conjunction level L
#'
#' \code{data_generation2()} generates a single simulated dataset for a given
#' conjunction level \code{L}. The first \code{L} traits carry non-zero effects
#' among the first \code{m1} rows, under a flattop prior, with optional sign
#' heterogeneity and optional equicorrelated noise among the first \code{L}
#' traits.
#'
#' This design is useful when you want one clean dataset at a specific
#' conjunction level (e.g. \code{L = 5, 4, 3, 2, 1}) without additional
#' replicates.
#'
#' @param m Total number of SNPs (rows). Default \code{100000}.
#' @param K Total number of traits (columns). Default \code{5}.
#' @param m1 Number of causal SNPs (first \code{m1} rows). Default \code{300}.
#' @param L Conjunction level: number of traits (among the first \code{K})
#'   that carry a non-zero effect. Typically \code{1 <= L <= K}. Default \code{5}.
#' @param se Scalar standard error for marginal noise. Default \code{0.3}.
#' @param flip_prob Probability of flipping the sign of each added effect entry
#'   (controls sign heterogeneity). Default \code{0.5}. Set to \code{0} to
#'   disable sign flipping.
#' @param flip_within_L Logical; if \code{TRUE}, the sign-flip mask is generated
#'   only on the first \code{L} traits; if \code{FALSE}, it is generated on all
#'   \code{K} traits (effects are still only added in the first \code{L} traits).
#'   Default \code{TRUE}.
#' @param use_corr_noise Logical; if \code{TRUE}, the first \code{L} traits
#'   receive equicorrelated noise with correlation \code{rho_cor_noise}.
#'   Otherwise, all traits use independent noise with variance \code{se^2}.
#'   Default \code{TRUE}.
#' @param rho_cor_noise Scalar correlation used for the equicorrelated noise
#'   among the first \code{L} traits. Default \code{0.8}.
#' @param seed Optional integer random seed for reproducibility. If
#'   \code{NULL}, the current RNG state is used. Default \code{NULL}.
#'
#' @return A list with components:
#' \describe{
#'   \item{BetaHat}{An \code{m x K} matrix of observed effects.}
#'   \item{S}{An \code{m x K} matrix of standard errors.}
#'   \item{L}{The conjunction level used.}
#'   \item{m,K,m1,se}{Basic design parameters.}
#'   \item{traits_lab}{Character vector of trait names.}
#'   \item{is_causal}{Length-\code{m} indicator vector (1 for causal rows).}
#'   \item{beta_star}{Length-\code{m1} vector of latent effect sizes reused
#'     across the first \code{L} traits.}
#'   \item{flip_prob,flip_within_L,use_corr_noise,rho_cor_noise,seed}{The
#'     control parameters actually used.}
#' }
#'
#' @export
data_generation2 <- function(
    m              = 100000L,
    K              = 5L,
    m1             = 300L,
    L              = 5L,
    se             = 0.3,
    flip_prob      = 0.5,
    flip_within_L  = TRUE,
    use_corr_noise = TRUE,
    rho_cor_noise  = 0.8,
    seed           = NULL
) {
  stopifnot(L >= 0L, L <= K)

  if (!is.null(seed)) {
    set.seed(seed)
  }

  traits_lab <- paste0("trait", 1:K)
  is_causal  <- c(rep(1L, m1), rep(0L, m - m1))

  beta_star <- r_flattop(m1)
  dat <- make_data_oneL(
    m = m, K = K, m1 = m1, L = L, se = se,
    beta_star      = beta_star,
    flip_prob      = flip_prob,
    flip_within_L  = flip_within_L,
    use_corr_noise = use_corr_noise,
    rho_cor_noise  = rho_cor_noise
  )

  list(
    BetaHat        = dat$betahat,
    S              = dat$S,
    L              = L,
    m              = m,
    K              = K,
    m1             = m1,
    se             = se,
    traits_lab     = traits_lab,
    is_causal      = is_causal,
    beta_star      = beta_star,
    flip_prob      = flip_prob,
    flip_within_L  = flip_within_L,
    use_corr_noise = use_corr_noise,
    rho_cor_noise  = rho_cor_noise,
    seed           = if (is.null(seed)) NA_integer_ else as.integer(seed)
  )
}

