
#include <cmath>
#include <random>
#include <vector>


// May, 2018 - Andrew Whiteman
//
/*! \class AdaM
\brief ADAptive Moment SGD optimization routine

Details.
 */



#ifndef _ABSEIL_
#define _ABSEIL_


namespace abseil {

  namespace abseil_rng {
    extern std::mt19937 _rng_;
    void set_seed(unsigned int seed);
  };

  

  template< typename T = double >
  class AdaM
  {
  private :
    T _mt;              // current momentum
    T _vt;              // current velocity
    bool _useLangevin;  // flag to adopt Langevin Dynamics
    bool _useRMS;       // flag to adopt "RMSprop" updates
    double _eta;        // learning rate
    double _etaScl;     // extra learning rate scaling parameter
    double _gamma[2];   // momentum/velocity decay rates
    double _eps;        // avoid division by zero constant
    double _dtheta;     // || \theta^(t) - \theta^(t-1) ||^2
    int _iter;          // current number of updates

    T computeDelta() const;
    void updateMomentum(const T &gt);
    void updateVelocity(const T &gt);

  
    template< typename S >
    void updatePosition(S &theta);


  public :
    AdaM(
      const T &theta,                // parameter(s) to optimize
      const double &eta = 0.01,      // SGD learning rate, range (0, 1]
      const double &gamma1 = 0.90,   // momentum decay rate, range (0, 1)
      const double &gamma2 = 0.999,  // velocity decay rate, range (0, 1)
      const double &eps = 1e-8       // small positive constant
    );


    // Update methods
    // type R is the same as the type returned by the gradient function
    template< typename S, typename R = S, typename... Args >
    void update(S &theta, R gradient(const S &theta, Args&&...), Args&&... args);

    // Exact version where each data point is used exactly once in each
    // full pass through the data
    template< typename S, typename R = S, typename... Args >
    void minibatchUpdate(
      S &theta,
      R unitGradient(const S &theta, const int &i, Args&&...),
      const int &batchSize,
      std::vector<int> &index,
      Args&&... args
    );

    // Approximate version for large data sets where data indices are
    // sampled with replacement for each minibatch
    template< typename S, typename R = S, typename... Args >
    void minibatchUpdate(
      S &theta,
      R unitGradient(const S &theta, const int &i, Args&&...),
      const int &batchSize,
      const int &dataSize,
      Args&&... args
    );

    

    // template< typename S, typename R, typename... Args >
    // void virtualMinibatch(
    //   S &theta,
    //   R unitGradient(const S&theta, const int &i, Args&&...),
    //   const int &batchSize,
    //   std::vector<int> &index,
    //   Args&&... args
    // );

    // Simple getter methods
    const T& momentum() const;
    const T& velocity() const;
    bool converged(const double &tol = 1e-6) const;
    int iteration() const;
    double dtheta() const;
    double eta() const;

    // Setter methods
    void eta(const double &eta);
  
    void clear();
    void incrementIteration();
    void toggleLangevinDynamics(const bool &useLD = true);
    void toggleRMSprop(const bool &useRMS = true);
  };





  template< typename T = double >
  class RandomWalk {
  private:
    bool _updateEps;
  
    double _cumJumpProb;
    double _eps;
    double _gamma;  // jump prob decay parameter
    double _logPcurrent;
    double _targetJumpProb;
    double _tolJump;
  
    int _iter;
  
    std::normal_distribution<double> _normal;
    std::uniform_real_distribution<double> _uniform;

    void updateStepSize(const double &k = 0.1);

  public:
    RandomWalk(
      const double &eps = 2.4,
      const double &targetJumpProb = 0.44,
      const double &gamma = 0.9
    );

    template< typename S, typename... Args >
    void update(
      S &theta,
      double logPosterior(const S &theta, Args&&...),
      std::mt19937 &rng,
      Args&&... args
    );

    // getters
    double epsilon() const;
    double jumpProbability() const;
    int iteration() const;

    void fixStepSize();
  };



  // template< typename T, typename S >
  // void updateTheta(T &theta, S &delta);


};


#include "AdaM.inl"
#include "abseil_rng.inl"

#endif  // _ADA_M_
