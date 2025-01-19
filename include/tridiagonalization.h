#ifndef TRIDIAGONALIZATION_H
#define TRIDIAGONALIZATION_H

#include <armadillo>
#include "base.h"
#include "householder.h"

template<typename MatrixType, typename RealVectorType, typename VectorType>
void tridiagonalization(MatrixType &mat, RealVectorType &diag, RealVectorType &subDiag,
                        VectorType &householders, VectorType &workspace, bool extractQ) {
    typedef typename MatrixType::elem_type Scalar;
    typedef typename arma::get_pod_type<Scalar>::result RealScalar;
    Index n = mat.n_rows;
    for (Index i = 0; i < n - 1; ++i) {
        Index remainingSize = n - i - 1;
        arma::subview_col<Scalar> tail = mat.col(i).tail(remainingSize);

        RealScalar beta;
        Scalar tau;
        make_householder(tail, tau, beta);

        mat(i + 1, i) = Scalar(1);

        arma::subview<Scalar> subA = mat.submat(i + 1, i + 1, n - 1, n - 1);
        householders.tail(remainingSize) = subA * (scalarConj(tau) * tail);
        Scalar alpha = scalarConj(tau) * RealScalar(-0.5) * arma::cdot(householders.tail(remainingSize), tail);
        householders.tail(remainingSize) += alpha * tail;
        subA -= (tail * householders.tail(remainingSize).t() + householders.tail(remainingSize) * tail.t());

        mat(i + 1, i) = beta;
        householders(i) = tau;
    }
    diag = arma::real(mat.diag());
    subDiag = arma::real(mat.diag(-1));
    if (extractQ) {
        MatrixType Q = arma::eye<MatrixType>(n, n);
        Index numVec = n - 1;
        for (Index k = numVec - 1; k >= 0; --k) {
            arma::subview<Scalar> Q_sub = Q.submat(k + 1, k + 1, n - 1, n - 1);
            arma::subview_col<Scalar> essential = k + 2 < n ? mat.col(k).subvec(k + 2, n - 1) : mat.col(k).subvec(0, 0);
            apply_householder_left(Q_sub, essential, scalarConj(householders(k)), workspace.memptr());
        }
        mat = Q;
    }
}


#endif // TRIDIAGONALIZATION_H
