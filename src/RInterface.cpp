
#include <Rcpp.h>

#include "abseil_exports.h"
// #include "utils.h"




template< typename T >
T wrapRFunction(const T &arg, const Rcpp::Function &fun) {
  return fun(arg);
};



extern "C" SEXP optimizeFromR(const SEXP par_, const SEXP objective_, const SEXP gradient_,
			      const SEXP tol_, const SEXP maxIter_, const SEXP learningRate_,
			      const SEXP momentumDecay_, const SEXP velocityDecay_
) {
  try {
    const Rcpp::Function objectiveFun(objective_), gradientFun(gradient_);
    const double tol(Rcpp::as<double>(tol_));
    const int maxIter(Rcpp::as<int>(maxIter_));
  
    bool converged = false;
    double currentObjective, previousObjective;
    Rcpp::NumericVector par(Rcpp::clone(par_));
  
    abseil::AdaM<Rcpp::NumericVector> sgd(par, Rcpp::as<double>(learningRate_),
					  Rcpp::as<double>(momentumDecay_),
					  Rcpp::as<double>(velocityDecay_));
    // sgd.toggleRMSprop(true);
  
    previousObjective = Rcpp::as<double>(objectiveFun(par));
    while (!converged && sgd.iteration() < maxIter) {
      sgd.update<Rcpp::NumericVector, Rcpp::NumericVector, const Rcpp::Function&>
	(par, wrapRFunction, gradientFun);
      currentObjective = Rcpp::as<double>(objectiveFun(par));
      // sgd.eta(sgd.eta() * ((currentObjective <= previousObjective) ? 1.1 : 0.5));
      converged = sgd.converged(tol) && (currentObjective <= previousObjective);
    }
  
    return Rcpp::wrap(Rcpp::List::create(
      Rcpp::Named("par") = par,
      Rcpp::Named("value") = currentObjective,
      Rcpp::Named("iteration") = sgd.iteration(),
      Rcpp::Named("convergence") = converged)
      );
  }
  catch (std::exception& ex) {
    forward_exception_to_r(ex);
  }
  catch (...) {
    ::Rf_error("C++ exception (unknown cause)");
  }
  return R_NilValue;  // not reached
};
