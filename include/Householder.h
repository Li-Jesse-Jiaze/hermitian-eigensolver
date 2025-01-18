#ifndef HOUSEHOLDER_H
#define HOUSEHOLDER_H

#include <armadillo>
#include <limits>
#include <cmath>
#include <complex>
#include "base.h"

using namespace base;

template<typename VectorType, typename Scalar, typename RealScalar>
void make_householder_inplace(VectorType &vector, Scalar &tau, RealScalar &beta) {
    using std::sqrt;
    using std::abs;
    using std::real;

    const Index n = vector.n_elem;
    if (n == 0) {
        tau = Scalar(0);
        beta = RealScalar(0);
        return;
    }

    Scalar c0 = vector[0];
    RealScalar tailSqNorm = RealScalar(0);
    if (n > 1) {
        tailSqNorm = arma::norm(vector.subvec(1, n - 1)) * arma::norm(vector.subvec(1, n - 1));
    }

    const RealScalar tol = std::numeric_limits<RealScalar>::min();
    if ((tailSqNorm <= tol) && ((std::imag(c0) * std::imag(c0)) <= tol)) {
        tau = Scalar(0);
        beta = real(c0);
        if (n > 1)
            vector.subvec(1, n - 1).zeros();
    } else {
        beta = sqrt(std::norm(c0) + tailSqNorm);
        if (real(c0) >= RealScalar(0)) {
            beta = -beta;
        }
        if (n > 1) {
            vector.subvec(1, n - 1) /= (c0 - beta);
        }
        tau = conjugate((beta - c0) / beta);
    }
}

template<typename MatrixType, typename VectorType, typename Scalar>
void apply_householder_on_the_left(MatrixType &M,
                                   const VectorType &v,
                                   const Scalar &tau,
                                   Scalar *workspace) {
    if (M.n_rows == 1) {
        M *= (Scalar(1) - tau);
        return;
    }
    if (tau == Scalar(0)) {
        return;
    }

    // 3) 构造一个在外部指针 workspace 上的临时向量 tmp
    arma::Row<Scalar> tmp(workspace, M.n_cols, false, false);
    arma::subview<Scalar> bottom = M.rows(1, M.n_rows - 1);
    tmp = v.t() * bottom;
    tmp += M.row(0);
    M.row(0) -= tau * tmp;
    bottom -= tau * v * tmp;
}

#endif  // HOUSEHOLDER_H
