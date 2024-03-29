
#include <Eigen/Core>
#include <random>
#include <Rcpp.h>

// #include "extraMath.h"

/* 
All my shame lives in this file.
 */


template< typename T >
double squaredNorm(const T &x) {  // Squared 2-norm
  return x * x;
};

double squaredNorm(const Eigen::VectorXd &x) {
  return x.squaredNorm();
};

double squaredNorm(const Eigen::VectorXf &x) {
  return x.squaredNorm();
};

double squaredNorm(const Eigen::ArrayXd &x) {
  return x.matrix().squaredNorm();
};

float squaredNorm(const Eigen::ArrayXf &x) {
  return x.matrix().squaredNorm();
};


// template< typename Derived >
// Derived squaredNorm(const Eigen::Array<Derived> &x) {
//   return x.pow(Derived(2)).sum();
// };



double squaredNorm(const Rcpp::NumericVector &x) {
  double result = 0.0;
  for (int i = 0; i < x.size(); i++)
    result += x[i] * x[i];
  return result;
};




// Eigen::VectorXf pow(const Eigen::VectorXf &base, const double &exp) {
//   Eigen::VectorXf result(base.size());
//   for (int i = 0; i < result.size(); i++)
//     result[i] = std::pow(base[i], exp);
//   return result;
// };


// Eigen::VectorXf sqrt(const Eigen::VectorXf &arg) {
//   Eigen::VectorXf result(arg.size());
//   for (int i = 0; i < result.size(); i++)
//     result[i] = std::sqrt(arg[i]);
//   return result;
// };




// template< typename T >
// T gaussianNoise(const T &scl, std::mt19937 &rng) {
//   std::normal_distribution<T> _z(0, 1);
//   return (scl * _z(rng));
// };

// Eigen::ArrayXd gaussianNoise(const Eigen::ArrayXd &scl, std::mt19937 &rng) {
//   static std::normal_distribution<double> _z(0, 1);
//   Eigen::ArrayXd noise = scl.unaryExpr([&](const double &x) {
//       return (x * _z(rng)); });
//   return (noise);
// };


// Eigen::ArrayXf gaussianNoise(const Eigen::ArrayXf &scl, std::mt19937 &rng) {
//   static std::normal_distribution<float> _z(0, 1);
//   Eigen::ArrayXf noise = scl.unaryExpr([&](const float &x) {
//       return (x * _z(rng)); });
//   return (noise);
// };



// template< typename Derived, typename RNG >
// Eigen::ArrayBase<Derived> gaussianNoise(
//   Eigen::ArrayBase<Derived> scl,
//   RNG &rng) {
//   typedef typename Derived::value_type value_type;
//   static std::normal_distribution<value_type> _StdNormal_(0, 1);
//   // Eigen::ArrayBase<Derived> noise(scl.size());
//   for (int i = 0; i < scl.size(); i++)
//     scl.coeff(i) *= _StdNormal_(rng);
//     // noise.template operater[](i) = scl[i] * _StdNormal_(rng);
//   // return noise;
//   return scl;
// };




// double gaussianNoise(const double &scl, std::mt19937 &rng) {
//   static std::normal_distribution<double> _z(0, 1);
//   return (scl * _z(rng));
// };



// template< typename RealType >
// Rcpp::NumericVector gaussianNoise(const Rcpp::NumericVector &scl, std::mt19937 &rng) {
//   std::normal_distribution<RealType> _StdNormal_(0, 1);
//   Rcpp::NumericVector noise = Rcpp::clone(scl);
//   for (int i = 0; i < noise.size(); i++)
//     noise[i] *= _StdNormal_(rng);
//   return noise;
// };

// Eigen::VectorXd gaussianNoise(const Eigen::VectorXd &scl, std::mt19937 &rng) {
//   static std::normal_distribution<double> _z(0, 1);
//   Eigen::VectorXd noise = scl.array().unaryExpr([&](const double &x) {
//       return (x * _z(rng)); }).matrix();
//   return (noise);
// };
