#ifndef HERMITIAN_EIGEN_SOLVER_H
#define HERMITIAN_EIGEN_SOLVER_H

#include <armadillo>

template <typename MatrixType>
class HermitianEigenSolver
{
public:
    typedef arma::uword Index;
    typedef typename MatrixType::elem_type Scalar;
    typedef typename arma::get_pod_type<Scalar>::result RealScalar;
    typedef arma::Col<Scalar> VectorType;
    typedef arma::Col<RealScalar> RealVectorType;
    HermitianEigenSolver& compute(const MatrixType& matrix, bool compute_eigenvectors = true);

    const MatrixType& eigenvectors() const {
        return m_eivectors;
    }

    const RealVectorType& eigenvalues() const {
        return m_eivalues;
    }

    static const int m_maxIterations = 30;
protected:
    MatrixType m_eivectors;
    VectorType m_workspace;
    RealVectorType m_eivalues;

};

template <typename MatrixType>
HermitianEigenSolver<MatrixType>& HermitianEigenSolver<MatrixType>::compute(const MatrixType& a_matrix, bool compute_eigenvectors) {
    // TODO
}

template <typename RealScalar, typename Scalar, typename Index>
static void tridiagonal_qr_step(RealScalar *diag, RealScalar *subdiag, Index start, Index end, Scalar *matrixQ, Index n)
{
    // TODO
}


#endif // HERMITIAN_EIGEN_SOLVER_H
