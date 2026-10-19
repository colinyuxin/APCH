simulate_small <- function(n = 2000, p = 2, seed = 1) {
  set.seed(seed)
  gamma <- matrix(rbinom(n * p, 1, 0.1), n, p)
  gamma[rbinom(n, 1, 0.05) == 1, ] <- 1
  B <- gamma * matrix(rnorm(n * p, sd = 2), n, p)
  Bhat <- B + matrix(rnorm(n * p), n, p)
  colnames(Bhat) <- paste0("f", seq_len(p))
  list(Bhat = Bhat, S = matrix(1, n, p), B = B)
}

dat <- simulate_small()
fit <- apch(dat$Bhat, dat$S, verbose = FALSE)

test_that("apch() returns an apch fit with lfsr calls", {
  expect_s3_class(fit, "apch")
  expect_identical(fit$joint_mode, "apch_s")
  expect_identical(fit$method, "lfsr")
  calls <- fit$calls
  expect_named(calls, c("effect", "L", "lfsr", "subset", "subset01"))
  expect_true(all(calls$L %in% 1:2))
  expect_gt(sum(calls$L == 2), 0)
  expect_output(print(fit), "APCH fit")
})

test_that("apch_select() switches criterion and level without refitting", {
  lfdr_calls <- apch_select(fit, alpha = 0.05, method = "lfdr")
  expect_named(lfdr_calls, c("effect", "L", "lfdr", "subset", "subset01"))
  strict <- apch_select(fit, alpha = 0.01)
  expect_lte(nrow(strict), nrow(fit$calls))
  expect_identical(apch_select(fit, alpha = 0.05), fit$calls)
})

test_that("subset-level lfsr is never below lfdr, and lfdr grows with the subset", {
  lfdr <- apch_lfdr(fit)
  lfsr <- apch_lfsr(fit)
  expect_identical(dim(lfdr), c(nrow(dat$Bhat), 3L))
  expect_identical(colnames(lfsr), colnames(lfdr))
  expect_true(all(lfsr >= lfdr - 1e-10))
  expect_true(all(lfdr[, "11"] >= pmax(lfdr[, "10"], lfdr[, "01"]) - 1e-12))
})

test_that("a separate marginal fit gives the same joint fit", {
  m <- apch_marginal(dat$Bhat, dat$S, verbose = FALSE)
  expect_s3_class(m, "apch_marginal")
  fit2 <- apch_joint(dat$Bhat, dat$S, marginal = m, verbose = FALSE)
  expect_equal(fit2$joint$post_config_prob, fit$joint$post_config_prob)
  other <- simulate_small(seed = 2)
  rownames(other$Bhat) <- paste0("x", seq_len(nrow(other$Bhat)))
  expect_error(apch_joint(other$Bhat, other$S, marginal = m, verbose = FALSE), "different effects")
})

test_that("missing entries give NA for subsets that include them", {
  d <- dat
  d$Bhat[1:20, 2] <- NA
  d$S[1:20, 2] <- NA
  fit_na <- apch_joint(d$Bhat, d$S, verbose = FALSE)
  lfdr <- apch_lfdr(fit_na)
  expect_true(all(is.na(lfdr[1:20, c("11", "01")])))
  expect_false(anyNA(lfdr[, "10"]))
  calls <- apch_select(fit_na)
  expect_false(anyNA(calls$lfsr))
  expect_false(any(calls$effect[calls$L == 2] %in% rownames(d$Bhat)[1:20]))
  fit_cor <- apch_joint(d$Bhat, d$S, C = matrix(c(1, 0.3, 0.3, 1), 2), verbose = FALSE)
  expect_false(anyNA(apch_lfdr(fit_cor)[, "10"]))
})

test_that("a single feature runs through the same interface", {
  d1 <- simulate_small(p = 1)
  fit1 <- apch(d1$Bhat, d1$S, verbose = FALSE)
  expect_true(all(fit1$calls$L == 1))
})

test_that("stage functions reject inputs that are not fits", {
  expect_error(apch_select(list()), "apch\\(\\)")
  expect_error(apch_joint(dat$Bhat, dat$S, marginal = list(), verbose = FALSE), "apch_marginal")
})
