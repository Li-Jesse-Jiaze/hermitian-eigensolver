#ifndef TRIDIAGONALIZATION_H
#define TRIDIAGONALIZATION_H

#include <armadillo>
#include "Householder.h"
#include "base.h"

using namespace base;

template <typename MatrixType, typename VectorType>
void tridiagonalization_inplace(MatrixType& A, VectorType& hCoeffs) {
    typedef typename MatrixType::elem_type Scalar;
    typedef typename arma::get_pod_type<Scalar>::result RealScalar;

    Index n = A.n_rows;
    for(Index i = 0; i < n - 1; ++i)
    {
        Index remainingSize = n - i - 1;
        arma::subview_col<Scalar> tail = A.col(i).tail(remainingSize);

        RealScalar beta;
        Scalar tau;
        make_householder_inplace(tail, tau, beta);

        A(i + 1, i) = Scalar(1);

        arma::subview<Scalar> subA = A.submat(i + 1, i + 1, n - 1, n - 1);
        hCoeffs.tail(remainingSize) = subA * (conjugate(tau) * tail);
        Scalar alpha = conjugate(tau) * RealScalar(-0.5) * arma::cdot(hCoeffs.tail(remainingSize), tail);
        hCoeffs.tail(remainingSize) += alpha * tail;
        subA -= (tail * hCoeffs.tail(remainingSize).t() + hCoeffs.tail(remainingSize) * tail.t());

        A(i + 1, i) = beta;
        hCoeffs(i) = tau;
    }
}

template <typename MatrixType, typename VectorType, typename Scalar>
MatrixType formQ(const MatrixType& A, const VectorType& hCoeffs, Scalar *workspace)
{
    Index n = A.n_rows;
    MatrixType Q = arma::eye<MatrixType>(n, n);
    Index vecs = n - 1;
    for (int k = vecs - 1; k >= 0; --k) {
        arma::subview<Scalar> Q_sub = Q.submat(k + 1, k + 1, n - 1, n - 1);
        arma::subview_col<Scalar> essential = k + 2 < n ? A.col(k).subvec(k + 2, n - 1) : A.col(k).subvec(0, 0);
        apply_householder_on_the_left(Q_sub, essential, conjugate(hCoeffs(k)), workspace);
    }
    return Q;
}

template <typename MatrixType, typename RealVectorType, typename VectorType>
void tridiagonalization(MatrixType& mat, RealVectorType& diag, RealVectorType& subdiag, VectorType& hCoeffs, VectorType& workspace, bool extractQ) {
    tridiagonalization_inplace(mat, hCoeffs);
    diag = arma::real(mat.diag());
    subdiag = arma::real(mat.diag(-1));
    if (extractQ) {
        mat = formQ(mat, hCoeffs, workspace.memptr());
    }
}


#endif // TRIDIAGONALIZATION_H
