
#include <algorithm>
#include <cmath>
#include <random>


template< typename T >
void abseil::RandomWalk<T>::RandomWalk(
  const double &eps,
  const double &targetJumpProb,
  const double &gamma
) :
  _updateEps(true), _cumJumpProb(0.0), _logPcurrent(1.0),
  _tolJump(0.1), _iter(0), _normal(0.0, 1.0), _uniform(0.0, 1.0)
{
  if (eps <= 0)
    throw (std::logic_error("Step size must be > 0"));
  if (targetJumpProb <= 0 || targetJumpProb >= 1)
    throw (std::logic_error("Target jump rate must be between (0, 1"));
  if (gamma <= 0 || gamma >= 1)
    throw (std::logic_error("Decay rate must be between (0, 1)"));
  _eps = eps;
  _targetJumpProb = targetJumpProb;
  _gamma = gamma;

  _updateEps = true;
  _cumJumpProb = 0.0;
  _logPcurrent = 1.0;
  _tolJump = 0.1;
  _iter = 0;
  _normal(0.0, 1.0);
  _uniform(0.0, 1.0);
};



template< typename T >
template< typename S, typename... Args >
void abseil::RandomWalk<T>::update(
  S &theta,
  double logPosterior(const S &theta, Args&&...),
  std::mt19937 &rng,
  Args&&... args
) {
  if (_iter == 0)
    _logPcurrent = std::function<double(const S&, Args&&...)>
      (logPosterior)(theta, std::forward<Args>(args)...);
  
  // Propose new theta with Normal kernel and Metropolis correction
  const S proposal = theta + _eps * _normal(rng);
  const double lpProposal = std::function<double(const S&, Args&&...)>
      (logPosterior)(proposal, std::forward<Args>(args)...);
  const double ratio = std::exp(lpProposal - _logPcurrent);
  if (_uniform(rng) < ratio) {
    theta = proposal;
    _logPcurrent = lpProposal;
  }

  // Adjust random walk parameters
  _cumJumpProb = _gamma * _cumJumpProb +
    (1 - _gamma) * std::min(ratio, 1.0);
  if (_updateEps && _iter % 100 == 0)
    updateStepSize();
  if (std::abs(jumpProbability() - _targetJumpProb) < _tolJump)
    _updateEps = false;
  _iter++;
};





template< typename T >
double abseil::RandomWalk<T>::epsilon() const {
  return _eps;
};

template< typename T >
double abseil::RandomWalk<T>::jumpProbability() const {
  double p = 0.0;
  if (_iter)
    p = _cumJumpProb / (1 - std::pow(_gamma, _iter));
  return (p);
};

template< typename T >
int abseil::RandomWalk<T>::iteration() const {
  return _iter;
};



template< typename T >
void abseil::RandomWalk<T>::fixStepSize() {
  _updateEps = false;
};


template< typename T >
void abseil::RandomWalk<T>::updateStepSize(const double &k) {
  if (_cumJumpProb > _targetJumpProb)
    _eps *= 1 - k;
  else if (_cumJumpProb < _targetJumpProb)
    _eps *= 1 + k;
}

