
library (abseil)

context ("GR Descent in Regression")

mse <- function(y, yhat = rep(0, length(y))) c(crossprod(y - yhat)) / length(y)


test_that ("Linear regression optimization works", {

  cat ("\nLinear regression example:\n")

  n <- 200
  P <- 20

  X <- cbind(1, matrix(rnorm(n * (P-1)), n))
  b <- rnorm(P, sd = sqrt(5))
  sigma.sq <- 1
  y <- rnorm(n, X %*% b, sqrt(sigma.sq))


  ## loglik <- function(theta, y, X, sigma.sq) {
  loglik <- function(theta) {
    resid <- c(y - X %*% theta)
    0.5 / sigma.sq * c(crossprod(resid))
  }

  ## loglik.grad <- function(theta, y, X, sigma.sq) {
  loglik.grad <- function(theta) {
    -c(1 / sigma.sq * t(X) %*% c(y - X %*% theta))
  }


  x0 <- rnorm(P)
  expect_equal (mse(loglik.grad(x0), numDeriv::grad(loglik, x0)), 0, 0.01)


  opt <- gr.descent(rnorm(P), loglik, .learning.rate = 0.1,
                    .tol = 1e-6, .velocity.decay = 0.999)


  ## system.time(
  opt <- gr.descent(rnorm(P), loglik, loglik.grad, .learning.rate = 0.5,
                    .tol = 1e-5, .velocity.decay = 0.999)
  ## )

  ## system.time( opt2 <- optim(rnorm(P), loglik, loglik.grad, method = "L-BFGS-B") )

  fit0 <- lm(y ~ X - 1)

  for (i in 1:P)
    expect_equal (unname(fit0$coefficients[i]), unname(opt$par[i]), 0.01)

  ## cbind(lm = coef(lm(y ~ X - 1)), AdaM = opt$par)
})




test_that ("Logistic regression optimization works", {

  cat ("\n\nLogistic regression example:\n")

  n <- 200
  P <- 5

  X <- cbind(1, matrix(rnorm(n * (P-1)), n))
  b <- rnorm(P, sd = sqrt(5))
  y <- rbinom(n, 1, plogis(X %*% b))


  ## loglik <- function(theta, y, X, sigma.sq) {
  loglik <- function(theta) {
    Xb <- c(X %*% theta)
    sum(log(1 + exp(-Xb)) + Xb * (1 - y))
  }

  ## loglik.grad <- function(theta, y, X, sigma.sq) {
  loglik.grad <- function(theta) {
    Xb <- c(X %*% theta)
    -c(t(X) %*% (1 / (1 + exp(Xb)) + y - 1))
  }


  x0 <- rnorm(P)
  expect_equal (mse(loglik.grad(x0), numDeriv::grad(loglik, x0)), 0, 0.01)


  opt <- gr.descent(rnorm(P), loglik, .learning.rate = 0.1,
                    .tol = 1e-6, .max.iter = 1000)


  system.time(
  opt <- gr.descent(rnorm(P), loglik, loglik.grad, .learning.rate = 1.1,
                    .tol = 1e-6, .max.iter = 1000)
  )

  fit0 <- glm(y ~ X - 1, family = binomial())

  for (i in 1:P)
    expect_equal (unname(fit0$coefficients[i]), unname(opt$par[i]), 0.01)

  cat ("\n")
  if (P <= 100)
    print (cbind(lm = coef(fit0), AdaM = opt$par))
})






test_that ("Laplace priors in logistic regression", {

  cat ("\n\nLogistic-Laplace regression example:\n")

  n <- 200
  P <- 50

  X <- cbind(1, matrix(rnorm(n * (P-1)), n))
  b <- rnorm(P, sd = sqrt(5))
  y <- rbinom(n, 1, plogis(X %*% b))

  ## sig.b <- 0.8345
  sig.b <- 0.4627564  ## > sig.b <- -log(4) / log(2 * (1 - 0.975))
                      ## > exp(qlaplace(0.975, 0, sig))
                      ## [1] 4


  ## qlaplace <- function(p, location = 0, scale = 1) {
  ##   location - scale * sign(p - 0.5) * log(1 - 2 * abs(p - 0.5))
  ## }


  ## loglik <- function(theta, y, X, sigma.sq) {
  loglik <- function(theta) {
    Xb <- c(X %*% theta)
    sum(log(1 + exp(-Xb)) + Xb * (1 - y)) + sum(log(2 * sig.b) + abs(theta) / sig.b)
  }



  ## loglik.grad <- function(theta, y, X, sigma.sq) {
  loglik.grad <- function(theta) {
    Xb <- c(X %*% theta)
    -c(t(X) %*% (1 / (1 + exp(Xb)) + y - 1)) + sign(theta) / sig.b
  }


  x0 <- rnorm(P)
  expect_equal (mse(loglik.grad(x0), numDeriv::grad(loglik, x0)), 0, 0.01)


  ## opt <- gr.descent(rnorm(P), loglik, .learning.rate = 0.1,
  ##                   .tol = 1e-6, .max.iter = 1000)


  system.time(
    opt <- gr.descent(rnorm(P), loglik, loglik.grad, .learning.rate = 1.1,
                      .tol = 1e-6, .max.iter = 1000)
  )



  fit0 <- glm(y ~ X - 1, family = binomial())

  ## for (i in 1:P)
  ##   expect_equal (unname(sign(b[i])), unname(sign(opt$par[i])))

  cat ("\n")
  if (P <= 100)
    print (cbind(True = b, glm = coef(fit0), AdaM = opt$par))


  fit.glmnet <- glmnet::cv.glmnet(X[, -1], y, family = "binomial")

  data.frame(True = b, glm = coef(fit0),
             LASSO = unname(c(as.matrix(coef(fit.glmnet, s = "lambda.min")))),
             AdaM = opt$par) %>%
    reshape2::melt("True") %>%
    ggplot(aes(True, value)) +
    geom_abline(slope = 1, intercept = 0, color = "darkgray") +
    geom_point() +
    facet_wrap(~ variable, nrow = 1, scales = "free_y") ->
    G

  dev.new (units = "in", height = 2, width = 6)
  print (G)
})

