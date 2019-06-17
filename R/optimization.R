
gr.descent <- function(par, fn, gr = NULL, ..., .tol = 1e-4, .max.iter = 500,
                       .learning.rate = 0.01, .momentum.decay = 0.9,
                       .velocity.decay = 0.999
) {
  if (is.null(gr))
    gr <- function(par) numDeriv::grad(fn, par, ...)
  .Call("optimizeFromR", par, fn, gr, .tol, .max.iter,
        .learning.rate, .momentum.decay, .velocity.decay,
        PACKAGE = "abseil")
}

## > args(numDeriv::grad)
## function (func, x, method = "Richardson", side = NULL, method.args = list(),
##     ...)

## l... <- list(func = fn, ...)
## gr <- function(par, l...) do.call(numDeriv::grad, c(x = par, l...))
