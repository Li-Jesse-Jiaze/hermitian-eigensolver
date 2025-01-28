#ifndef HERMITIAN_EIGEN_SOLVER_H
#define HERMITIAN_EIGEN_SOLVER_H

#include "tridiagonalization.h"
#include "givens.h"
#include "base.h"

/**
 * @brief One symmetric tridiagonal QR step with implicit Wilkinson shift
 *
 * @param diag the diagonal vector of the input tridiagonal matrix
 * @param sub_diag the sub-diagonal vector of the input tridiagonal matrix
 * @param start starting index to work on
 * @param end last+1 index to work on
 * @param Q pointer to the column-major matrix with eigenvectors, set to 0 if no eigenvectors need
 * @param n size of the input matrix
 */
template<typename RealScalar, typename Scalar>
static void tri_diag_qr_step(RealScalar *diag, RealScalar *sub_diag, Index start, Index end, Scalar *Q, Index n);

/**
 * @brief Solve EVP for a tridiagonal matrix
 *
 * @param[in,out] diag Input of the main diagonal, output of the eigenvalues
 * @param[in,out] sub_diag Input of the sub diagonal
 * @param[in] max_iter The maximum number of iterations
 * @param[in] do_vectors Whether the eigenvectors have to be computed or not
 * @param[out] eigenvectors The matrix to store the eigenvectors as column vectors
 */
template<typename MatrixType, typename RealVectorType>
void eigen_tri_diag(RealVectorType &diag, RealVectorType &sub_diag,
                    Index max_iter, bool do_vectors,
                    MatrixType &eigenvectors);

/**
 * @brief Computes eigenvalues and eigenvectors of Hermitian matrices
 * 
 * @tparam MatrixType 
 */
template<typename MatrixType>
class HermitianEigenSolver {
public:
    typedef typename MatrixType::elem_type Scalar;
    typedef typename arma::get_pod_type<Scalar>::result RealScalar;
    typedef arma::Col<Scalar> VectorType;
    typedef arma::Col<RealScalar> RealVectorType;

    /**
     * @brief Construct a new Hermitian Eigen Solver
     * 
     * @param[in] matrix A Hermitian matrix
     * @param[in] computeVectors true (default) for compute eigenvectors as well or false compute eigenvalues only
     */
    explicit HermitianEigenSolver(const MatrixType &matrix, bool computeVectors = true)
            : mEigenVectors(matrix.n_rows, matrix.n_cols),
              mWorkspace(2 * matrix.n_cols),
              mEigenValues(matrix.n_cols),
              mSubDiag(matrix.n_rows > 1 ? matrix.n_rows - 1 : 1) {
        compute(matrix, computeVectors);
    }

    HermitianEigenSolver &compute(const MatrixType &matrix, bool computeVectors = true);

    /**
     * @brief Return a const reference to the matrix whose columns are the eigenvectors
     * 
     * @return const MatrixType& 
     */
    const MatrixType &eigenvectors() const {
        return mEigenVectors;
    }

    /**
     * @brief Return a const reference to the column vector containing the eigenvalues
     * 
     * @return const RealVectorType&
     */
    const RealVectorType &eigenvalues() const {
        return mEigenValues;
    }

    static const Index mMaxIterations = 30; // full max iterations is 30n, from LAPACK

protected:
    // TODO: Can we use fixed size for some of these?
    RealVectorType mEigenValues;
    MatrixType mEigenVectors;
    VectorType mWorkspace;
    RealVectorType mSubDiag;
};

template<typename MatrixType>
HermitianEigenSolver<MatrixType> &
HermitianEigenSolver<MatrixType>::compute(const MatrixType &matrix, bool computeVectors) {
    Index n = matrix.n_cols;

    if (n == 1) {
        mEigenVectors = matrix;
        mEigenValues(0) = std::real(matrix(0, 0));
    }

    RealVectorType &diag = mEigenValues;
    MatrixType &mat = mEigenVectors;

    mat = matrix;
    RealScalar scale = arma::abs(mat).max();
    if (scale == RealScalar(0))
        scale = RealScalar(1);
    mat /= scale;
    tridiagonalization(mat, diag, mSubDiag, mWorkspace, computeVectors);
    eigen_tri_diag(diag, mSubDiag, mMaxIterations, computeVectors, mEigenVectors);
    mEigenValues *= scale;
    return *this;
}

template<typename MatrixType, typename RealVectorType>
void eigen_tri_diag(RealVectorType &diag, RealVectorType &sub_diag,
                    Index max_iter, bool do_vectors,
                    MatrixType &eigenvectors) {
    typedef typename MatrixType::elem_type Scalar;
    typedef typename arma::get_pod_type<Scalar>::result RealScalar;
    Index n = diag.size();
    Index end = n - 1;
    Index start = 0;
    Index iter = 0;

    const RealScalar eps = std::numeric_limits<RealScalar>::epsilon();
    while (end > 0) {
        // Deflation
        // Scan the sub-diagonal for "zero"
        for (Index i = start; i < end; ++i) {
            if (std::abs(sub_diag[i]) <= eps * (std::abs(diag[i]) + std::abs(diag[i + 1])))
                sub_diag[i] = RealScalar(0);
        }

        // find the largest unreduced block
        while (end > 0 && (sub_diag[end - 1] == RealScalar(0)))
            end--;
        if (end <= 0)
            break;
        start = end - 1;
        while (start > 0 && (sub_diag[start - 1] != RealScalar(0)))
            start--;

        tri_diag_qr_step(diag.memptr(), sub_diag.memptr(), start, end,
                         do_vectors ? eigenvectors.memptr() : nullptr, n);

        if (++iter > max_iter * n) {
            std::cerr << "Warning: The QR algorithm did not converge" << std::endl;
            break;
        }
    }
}

template<typename RealScalar, typename Scalar>
static void tri_diag_qr_step(RealScalar *diag, RealScalar *sub_diag, Index start, Index end, Scalar *Q, Index n) {
    // Wilkinson Shift
    RealScalar td = (diag[end - 1] - diag[end]) * RealScalar(0.5);
    RealScalar e = sub_diag[end - 1];
    // e^2 or td^2 can be underflow, so dont use mu = diag[end] - e*e / (td + (td>0 ? 1 : -1) * sqrt(td*td + e*e));
    RealScalar mu = diag[end];
    if (td == RealScalar(0)) {
        mu -= std::abs(e);
    } else if (e != RealScalar(0)) {
        const RealScalar e2 = e * e;
        const RealScalar h = std::hypot(td, e);
        if (e2 == RealScalar(0)) {
            mu -= e / ((td + (td > RealScalar(0) ? h : -h)) / e);
        } else {
            mu -= e2 / (td + (td > RealScalar(0) ? h : -h));
        }
    }

    RealScalar x = diag[start] - mu;
    RealScalar z = sub_diag[start];
    for (Index k = start; k < end && z != RealScalar(0); ++k) {
        RealScalar c, s;
        // If z is zero, Givens will make it remain zero.
        make_givens(x, z, c, s);

        // A = G A G^T
        RealScalar sdk = s * diag[k] + c * sub_diag[k];
        RealScalar dkp1 = s * sub_diag[k] + c * diag[k + 1];

        diag[k] = c * (c * diag[k] - s * sub_diag[k]) - s * (c * sub_diag[k] - s * diag[k + 1]);
        diag[k + 1] = s * sdk + c * dkp1;
        sub_diag[k] = c * sdk - s * dkp1;

        if (k > start)
            sub_diag[k - 1] = c * sub_diag[k - 1] - s * z;

        // Bulge chasing back to Hessenberg form
        x = sub_diag[k];
        if (k < end - 1) {
            z = -s * sub_diag[k + 1];
            sub_diag[k + 1] = c * sub_diag[k + 1];
        }

        // apply the givens rotation to the unitary matrix Q = Q * G
        if (Q) {
            apply_givens_right(Q, n, k, k + 1, c, s);
        }
    }
}

#endif // HERMITIAN_EIGEN_SOLVER_H
