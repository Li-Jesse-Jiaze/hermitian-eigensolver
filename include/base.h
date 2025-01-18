#ifndef BASE_H
#define BASE_H

#include <cstddef>
#include <complex>

namespace base {
    typedef std::ptrdiff_t Index;

    template<typename T>
    T conjugate(const T &x) {
        return x;
    }

    template<typename T>
    std::complex<T> conjugate(const std::complex<T> &x) {
        return std::conj(x);
    }
}

#endif //BASE_H
