
#include <Eigen/Core>
#include <random>
#include <Rcpp.h>
#include <type_traits>


#ifndef _EXTRA_MATH_
#define _EXTRA_MATH_


// Squared 2-norm: these should be relocated at some point
// (general ExtraMath.h/cpp file or something)

template< typename T = double >
double squaredNorm(const T &x);

double squaredNorm(const Eigen::VectorXd &x);
double squaredNorm(const Eigen::VectorXf &x);
double squaredNorm(const Eigen::ArrayXd &x);

float squaredNorm(const Eigen::ArrayXf &x);


// template< typename Derived >
// Derived squaredNorm(const Eigen::ArrayBase<Derived> &x);

double squaredNorm(const Rcpp::NumericVector &x);


// Eigen::VectorXf pow(const Eigen::VectorXf &base, const double &exp);
// Eigen::VectorXf sqrt(const Eigen::VectorXf &arg);

// template< typename T = double >
// T gaussianNoise(const T &scl, std::mt19937 &rng);

Eigen::ArrayXd gaussianNoise(const Eigen::ArrayXd &scl, std::mt19937 &rng);
double gaussianNoise(const double &scl, std::mt19937 &rng);
// Eigen::VectorXd gaussianNoise(const Eigen::VectorXd &scl, std::mt19937 &rng);


Rcpp::NumericVector gaussianNoise(const Rcpp::NumericVector &scl, std::mt19937 &rng);



template< typename T, typename S >
void updateTheta(T &theta, S &delta) {
  theta -= delta;
};



void updateTheta(Rcpp::NumericVector &theta, Rcpp::NumericVector &delta) {
#ifndef NDEBUG
  if (theta.size() != delta.size())
    throw (std::logic_error("updateTheta Rcpp::NumericVector dimension mismatch\n"));
#endif
  Rcpp::NumericVector::iterator itTheta = theta.begin(), itDelta = delta.begin();
  for (; itTheta != theta.end() || itDelta != delta.end(); itTheta++, itDelta++)
    (*itTheta) -= (*itDelta);
};


template< typename T >
T signum(const T &x) {
  if constexpr (std::is_signed_v<T>)
    return T(0) < x;
  else
    return (T(0) < x) - (T(0) > x);
};



#include "extraMath.inl"


#endif  // _EXTRA_MATH_
