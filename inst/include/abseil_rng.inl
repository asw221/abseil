

// #include "abseil_exports.h"


std::mt19937 abseil::abseil_rng::_rng_(42);

void abseil::abseil_rng::set_seed(unsigned int seed) {
  abseil::abseil_rng::_rng_.seed(seed);
};


