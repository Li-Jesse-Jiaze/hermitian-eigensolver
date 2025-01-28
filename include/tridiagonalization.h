#ifndef TRIDIAGONALIZATION_H
#define TRIDIAGONALIZATION_H

#include <armadillo>
#include "base.h"
#include "householder.h"

/**
 * @brief Tridiagonalization the Hermitian matrix A s.t. A = Q T Q*
 * 
 * @param[in, out] A The input Hermitian matrix and the output Q matrix
 * @param[out] diag The diagonal of T
 * @param[out] sub_diag The subdiagonal of T
 * @param[in] workspace Pointer for memory reuse with at least 2 * A.n_cols entries
 * @param[in] withQ If true, store the matrix Q in A
 */
template<typename MatrixType, typename RealVectorType, typename VectorType>
void tridiagonalization(MatrixType &A, RealVectorType &diag, RealVectorType &sub_diag,
                        VectorType &workspace, bool withQ) {
    typedef typename MatrixType::elem_type Scalar;
    typedef typename arma::get_pod_type<Scalar>::result RealScalar;
    Index n = A.n_rows;
    // Need n - 1 more scalars for householder coefficients and tmp w
    VectorType householders(workspace.memptr(), n - 1, false, false);
    // Use Householder reflection to eliminate elements below the subdiagonal
    for (Index i = 0; i < n - 1; ++i) {
        Index remain = n - i - 1; // In column i size(TBD) = n - i - 1
        arma::subview_col<Scalar> tail = A.col(i).tail(remain);

        RealScalar beta;
        Scalar tau;
        make_householder(tail, tau, beta);

        A(i + 1, i) = Scalar(1); // Reconstruct the Householder vector u in A[i+1:, i]

        // Apply H to A[i+1:, i+1:] A = H A H*, see Golub's "Matrix Computations" Chapter 5.1.4
        arma::subview<Scalar> A_sub = A.submat(i + 1, i + 1, n - 1, n - 1);

        // do scalar * vector firstly to avoid scalar * matrix, since "*" is left associative in C++
        householders.tail(remain) = A_sub * (scalar_conj(tau) * tail);
        Scalar alpha = scalar_conj(tau) * RealScalar(-0.5) * arma::cdot(householders.tail(remain), tail);
        householders.tail(remain) += alpha * tail;
        arma::Mat<Scalar> uw = tail * householders.tail(remain).t();
        A_sub -= (uw + uw.t());

        sub_diag(i) = beta;
        householders(i) = tau; // Now householders[:i] is useless so use it to store tau
    }
    diag = arma::real(A.diag());
    if (withQ) {
        // Q = H_n-1 ... H_1 I
        MatrixType Q = arma::eye<MatrixType>(n, n);
        Index num_vec = n - 1;
        for (Index k = num_vec - 1; k >= 0; --k) {
            arma::subview<Scalar> Q_sub = Q.submat(k + 1, k + 1, n - 1, n - 1);
            arma::subview_col<Scalar> u = k + 2 < n ? A.col(k).subvec(k + 2, n - 1) : A.col(k).subvec(0, 0);
            apply_householder_left(Q_sub, u, scalar_conj(householders(k)), workspace.memptr() + n);
        }
        A = Q;
    }
}


#endif // TRIDIAGONALIZATION_H
