#ifndef BASE_H
#define BASE_H

#include <cstddef>
#include <armadillo>
#include <limits>
#include <cmath>
#include <complex>

typedef std::ptrdiff_t Index;

template<typename T>
T scalarConj(const T &x) {
    return x;
}

template<typename T>
std::complex<T> scalarConj(const std::complex<T> &x) {
    return std::conj(x);
}

#endif //BASE_H
