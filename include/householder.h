#ifndef HOUSEHOLDER_H
#define HOUSEHOLDER_H

#include "base.h"

/**
 * @brief Make a reflector H s.t. H v = ||v|| e1 = [beta, 0, ..., 0]^T where H = I - tau u u^*, see Golub's "Matrix Computations" algorithm 5.1.1.
 * 
 * @param[in,out] vector The input v and the output u with u[1:] in v[1:]
 * @param tau The scalar factor of the Householder
 * @param beta The norm of v
 */
template<typename VectorType, typename Scalar, typename RealScalar>
void make_householder(VectorType &vector, Scalar &tau, RealScalar &beta) {
    const Index n = vector.n_elem;
    if (n == 0) {
        tau = Scalar(0);
        beta = RealScalar(0);
        return;
    }

    Scalar c0 = vector[0];
    RealScalar tail_abs2 = RealScalar(0);
    if (n > 1) {
        tail_abs2 = arma::norm(vector.subvec(1, n - 1)) * arma::norm(vector.subvec(1, n - 1));
    }

    const RealScalar tol = std::numeric_limits<RealScalar>::min();
    if ((tail_abs2 <= tol) && ((std::imag(c0) * std::imag(c0)) <= tol)) {
        // If is already "done"
        tau = Scalar(0);
        beta = std::real(c0);
        if (n > 1)
            vector.subvec(1, n - 1).zeros();
    } else {
        // beta = -sign(real(c0)) * ||v||_2
        // std::norm() is the absolute square not the Euclidean norm
        beta = std::sqrt(std::norm(c0) + tail_abs2);
        if (std::real(c0) >= RealScalar(0)) {
            beta = -beta;
        }
        if (n > 1) {
            vector.subvec(1, n - 1) /= (c0 - beta);
        }
        tau = scalarConj((beta - c0) / beta);
    }
}

/**
 * @brief A = H A where H = I - tau u u^* and u = [1 v]
 * 
 * @param A The input matrix
 * @param v Tail of the Householder vector
 * @param tau The scalar factor of the Householder
 * @param workspace Pointer for memory reuse with at least A.n_cols entries
 */
template<typename MatrixType, typename VectorType, typename Scalar>
void apply_householder_left(MatrixType &A, const VectorType &v, const Scalar &tau, Scalar *workspace) {
    if (A.n_rows == 1) {
        A *= (Scalar(1) - tau);
        return;
    }
    if (tau == Scalar(0)) {
        return;
    }
    arma::Row<Scalar> tmp(workspace, A.n_cols, false, false);
    arma::subview<Scalar> bottom = A.rows(1, A.n_rows - 1);
    tmp = v.t() * bottom;
    tmp += A.row(0);
    A.row(0) -= tau * tmp;
    bottom -= tau * v * tmp;
}

#endif  // HOUSEHOLDER_H
