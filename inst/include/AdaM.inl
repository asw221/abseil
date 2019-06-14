
#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "extraMath.h"


using std::sqrt;
using std::pow;



// May, 2018 - Andrew Whiteman
//
// Implementation file for AdaM.h
// ===================================================================

// template< typename T, typename S >
// void abseil::updateTheta(T &theta, S &delta) {
//   theta -= delta;
// };

// Define:
// void updateTheta(Other &theta, Eigen::ArrayXd &delta);
// As long as you can do math with the Other class, this should be good

// Then define a linear threshold coefficient object (inheriting from Eigen)
//                                            ^ Eigen::ArrayXd (hence namespace)
// along with a type-specific overloaded updateTheta()
// Might require clever handling of namespaces




// Public Methods
// -------------------------------------------------------------------

template< typename T >
abseil::AdaM<T>::AdaM(
  const T &theta,
  const double &eta,
  const double &gamma1,
  const double &gamma2,
  const double &eps
) {
  if (eta <= 0)
    throw (std::logic_error("Learning rate must be > 0"));
  if (gamma1 <= 0 || gamma1 >= 1)
    throw (std::logic_error("Momentum decay rate must be between (0, 1)"));
  if (gamma2 <= 0 || gamma2 >= 1)
    throw (std::logic_error("Velocity decay rate must be between (0, 1)"));
  if (eps <= 0)
    throw (std::logic_error("eps must be between (0, inf)"));

  _eta = eta;
  _etaScl = 1.0;
  _gamma[0] = gamma1;
  _gamma[1] = gamma2;
  _eps = eps;

  _mt = theta * 0;
  _vt = theta * 0;
  _useLangevin = false;
  _useRMS = false;
  _dtheta = 0.0;
  _iter = 1;
};


// Batch update
// type R is the same as the type returned by the gradient function
template< typename T >
template< typename S, typename R, typename... Args >
void abseil::AdaM<T>::update(S &theta, R gradient(const S &theta, Args&&...), Args&&... args) {
  T gt = std::function< R(const S&, Args&&...) >
    (gradient)(theta, std::forward<Args>(args)...);  // compute gradient
  updateMomentum(gt);
  updateVelocity(gt);
  updatePosition(theta);
  _iter++;
};


// // Minibatch update
// // type R is the same as the type returned by the gradient function
// template< typename T >
// template< typename S, typename R, typename... Args >
// void abseil::AdaM<T>::minibatchUpdate(
//   S &theta,
//   R unitGradient(const S &theta, const int &i, Args&&...),
//   const int &batchSize,
//   std::vector<int> &index,
//   Args&&... args
// ) {
//   const int N = index.size();
//   std::shuffle(index.begin(), index.end(), abseil::abseil_rng::_rng_);
//   std::function<R(const S&, const int&, Args&&...)> Grad(unitGradient);
//   T gt = _mt * 0;
//   int n = 0, j = 1;
//   for (std::vector<int>::iterator it(index.begin());
//        it != index.end();
//        it++, j++, n++) {
//     // gt += std::function< R(const S&, const int&, Args&&...) >
//     //   (unitGradient)(theta, (*it), std::forward<Args>(args)...);
//     gt += Grad(theta, (*it), std::forward<Args>(args)...);
//     if (j % batchSize == 0 || j == N) {
//       // gt *= N / n;
//       _etaScl = N / n;
//       updateMomentum(gt);
//       updateVelocity(gt);
//       updatePosition(theta);
//       gt *= 0;
//       // _etaScl = 1.0;
//       n = 0;
//     }
//   }
//   _iter++;
// };




// 'Approximate' aversion of the minibatch update
// type R is the same as the type returned by the gradient function
// (default is typename S)
template< typename T >
template< typename S, typename R, typename... Args >
void abseil::AdaM<T>::minibatchUpdate(
  S &theta,
  R unitGradient(const S &theta, const int &i, Args&&...),
  const int &batchSize,
  const int &dataSize,
  Args&&... args
) {
  std::uniform_int_distribution<int> UniformInteger(0, dataSize);
  std::function<R(const S&, const int&, Args&&...)> Grad(unitGradient);
  T gt = _mt * 0;
  for (int i = 0; i < batchSize; i++) {
    gt += Grad(theta, UniformInteger(abseil::abseil_rng::_rng_),
	       std::forward<Args>(args)...);
  }
  _etaScl = dataSize / batchSize;
  updateMomentum(gt);
  updateVelocity(gt);
  updatePosition(theta);
  _iter++;
  _iter++;
};





// template< typename T >
// template< typename S, typename R, typename... Args >
// void abseil::AdaM<T>::virtualMinibatch(
//   S &theta,
//   R unitGradient(const S&theta, const int &i, Args&&...),
//   const int &batchSize,
//   std::vector<int> &index,
//   Args&&... args
// ) {
//   static int callCount = 0;
//   std::shuffle(index.begin(), index.end(), abseil::abseil_rng::_rng_);
//   std::vector<int>::iterator it(index.begin());
//   std::function<R(const S&, const int&, Args&&...)> Grad(unitGradient);
//   T gt = _mt * 0;
//   for (int j = 0; j < batchSize; j++, it++)
//     gt += Grad(theta, (*it), std::forward<Args>(args)...);
//   updateMomentum(gt);
//   updateVelocity(gt);
//   // T delta = computeDelta(rng);
//   // _dtheta = squaredNorm(delta);
//   callCount++;
//   if (batchSize * callCount >= index.size()) {
//     _iter++;
//     callCount = 0;  // <- bug w/ clearHistory()
//   }
//   // return (delta);
// };




// Getter methods
// -------------------------------------------------------------------

template< typename T >
const T& abseil::AdaM<T>::momentum() const {
  return (_mt);
};

template< typename T >
const T& abseil::AdaM<T>::velocity() const {
  return (_vt);
};


template< typename T >
bool abseil::AdaM<T>::converged(const double &tol) const {
  return (_iter > 1 && _dtheta <= tol);
};

template< typename T >
int abseil::AdaM<T>::iteration() const {
  return (_iter - 1);
};

template< typename T >
double abseil::AdaM<T>::dtheta() const {
  return (_dtheta);
};


template< typename T >
double abseil::AdaM<T>::eta() const {
  double eta = _eta * _etaScl * sqrt(1 - pow(_gamma[1], _iter));
  return (_useRMS ? eta : eta / (1 - pow(_gamma[0], _iter)));
};



template< typename T >
void abseil::AdaM<T>::eta(const double &eta) {
  _eta = eta;
};






template< typename T >
void abseil::AdaM<T>::clear() {
  _mt *= 0;
  _vt *= 0;
  _etaScl = 1.0;
  _useLangevin = false;
  _useRMS = false;
  _dtheta *= 0;
  _iter = 1;
};


template< typename T >
void abseil::AdaM<T>::incrementIteration() {
  _iter++;
};


template< typename T >
void abseil::AdaM<T>::toggleLangevinDynamics(const bool &useLD) {
  _useLangevin = useLD;
};


template< typename T >
void abseil::AdaM<T>::toggleRMSprop(const bool &useRMS) {
  _useRMS = useRMS;
};






// Private Methods
// -------------------------------------------------------------------

template< typename T >
T abseil::AdaM<T>::computeDelta() const {
  T delta = eta() / (sqrt(_vt) + _eps) * _mt;
  // T scale = eta() / (sqrt(_vt) + _eps);
  // T delta = scale * _mt;
  if (_useLangevin) {
    double scl = M_SQRT2 * eta() / _etaScl;
    if (!_useRMS)
      scl *= 1 - pow(_gamma[0], _iter);
    delta += gaussianNoise(sqrt(scl / (sqrt(_vt) + _eps)), abseil::abseil_rng::_rng_);
  }
  return (delta);
};


template< typename T >
void abseil::AdaM<T>::updateMomentum(const T &gt) {
  if (!_useRMS)
    _mt = _gamma[0] * _mt + (1 - _gamma[0]) * gt;
  else
    _mt = gt;
};


template< typename T >
void abseil::AdaM<T>::updateVelocity(const T &gt) {
  _vt = _gamma[1] * _vt + (1 - _gamma[1]) * pow(gt, 2.0);
};







template< typename T >
template< typename S >
void abseil::AdaM<T>::updatePosition(S &theta) {
  T delta = computeDelta();
  updateTheta(theta, delta);
  _dtheta = squaredNorm(delta);
};







