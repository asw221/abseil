
#include <Eigen/Core>
#include <random>
#include <Rcpp.h>
#include <string>
#include <typeinfo>
#include <type_traits>

#include "type_definitions.h"



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




// template< typename RealType = double >
// Rcpp::NumericVector gaussianNoise(const Rcpp::NumericVector &scl, std::mt19937 &rng);



template< typename RealType = double, typename T, typename RNG >
void gaussianNoise(T &scl, RNG &rng) {
  std::normal_distribution<RealType> _StdNormal_(0, 1);
  if constexpr (is_iterable_v<T>) {
      for (typename T::iterator it = scl.begin(); it != scl.end(); ++it)
	*it *= _StdNormal_(rng);
  }
  else if constexpr (is_indexable_v<T>) {
      for (int i = 0; i < scl.size(); i++)
	scl[i] *= _StdNormal_(rng);
  }
  else if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
      scl *= _StdNormal_(rng);
  }
  else {
    const std::string msg = "Don't know how to deal with objects of type: ";
    throw std::logic_error((msg + typeid(scl).name()).c_str());
  }
  // return scl;
};



// template< typename T, typename RNG >
// T gaussianNoise(T scl, RNG &rng) {
//   std::normal_distribution<> _StdNormal_(0, 1);
//   if constexpr (is_vector_like_v<T>) {
//       /*
//       if constexpr (has_value_type_v<T> &&
// 		    std::is_floating_point_v<typename T::value_type>) {
// 	  typedef typename T::value_type value_type;
// 	  std::normal_distribution<value_type> _StdNormal_(0, 1);
//       }
//       else {
// 	std::normal_distribution<> _StdNormal_(0, 1);
// 	// ^^ potentially sloppy, but hopefully this works
//       }
//       */      
//       if constexpr (is_iterable_v<T>) {
// 	  // consider adding an is_castable type def or something
// 	  if constexpr (has_value_type_v<T>) {
// 	      typedef typename T::value_type value_type;
// 	      try {
// 		for (typename T::iterator it = scl.begin();
// 		     it != scl.end(); ++it)
// 		  *it *= (value_type)_StdNormal_(rng);
// 	      }
// 	      catch (...) {
// 		throw std::logic_error("Don't know how to deal with containers",
// 				       " with base type ",
// 				       typeid(*(scl.begin)).name(), "\n");
// 	      }
// 	  }
// 	  else {
// 	    throw std::logic_error("Don't know how to deal with containers",
// 				   " with base type ",
// 				   typeid(*(scl.begin)).name(), "\n");
// 	  }
//       }
//       else {
// 	if constexpr (has_value_type<T>) {
// 	    typedef typename T::value_type value_type;
// 	    try {
// 	      for (int i = 0; i < scl.size(); i++)
// 		scl[i] *= (value_type)_StdNormal_(rng);
// 	    }
// 	    catch (...) {
// 	      throw std::logic_error("Don't know how to deal with containers",
// 				     " with base type ",
// 				     typeid(scl[0]).name(), "\n");
// 	    }
// 	}
// 	else {
// 	  throw std::logic_error("Don't know how to deal with containers",
// 				 " with base type ",
// 				 typeid(scl[0]).name(), "\n");
// 	}
//       }
//   }
//   else {  // assuming this must be an std::is_floating_point type
//     // std::normal_distribution<T> _StdNormal_(0, 1);
//     scl *= _StdNormal_(rng);
//   }
//   return scl;
// };


// Eigen::ArrayXd gaussianNoise(const Eigen::ArrayXd &scl, std::mt19937 &rng);
// double gaussianNoise(const double &scl, std::mt19937 &rng);
// Eigen::VectorXd gaussianNoise(const Eigen::VectorXd &scl, std::mt19937 &rng);
// Eigen::ArrayXf gaussianNoise(const Eigen::ArrayXf &scl, std::mt19937 &rng);


// template< typename Derived, typename RNG >
// Eigen::ArrayBase<Derived> gaussianNoise(
//   Eigen::ArrayBase<Derived> scl,
//   RNG &rng);




template< typename T, typename S = T >
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
