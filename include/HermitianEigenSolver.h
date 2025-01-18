#ifndef HERMITIAN_EIGEN_SOLVER_H
#define HERMITIAN_EIGEN_SOLVER_H

#include <armadillo>
#include "Tridiagonalization.h"
#include "GivensRotation.h"
#include "base.h"

using namespace base;

template<typename RealScalar, typename Scalar>
static void tri_diag_qr_step(RealScalar *diag, RealScalar *subDiag, Index start, Index end, Scalar *matrixQ, Index n);

template<typename MatrixType, typename RealVectorType>
void compute_from_tri_diag(RealVectorType &diag, RealVectorType &subDiag,
                           Index maxIterations, bool computeEigenvectors,
                           MatrixType &eigenVectors);

template<typename MatrixType>
class HermitianEigenSolver {
public:
    typedef typename MatrixType::elem_type Scalar;
    typedef typename arma::get_pod_type<Scalar>::result RealScalar;
    typedef arma::Col<Scalar> VectorType;
    typedef arma::Col<RealScalar> RealVectorType;

    HermitianEigenSolver()
            : mEigenVectors(),
              mWorkspace(),
              mEigenValues(),
              mSubDiag(),
              mHouseholderCoefficients() {}

    explicit HermitianEigenSolver(const MatrixType &matrix, bool computeEigenvectors = true)
            : mEigenVectors(matrix.n_rows, matrix.n_cols),
              mWorkspace(matrix.n_cols),
              mEigenValues(matrix.n_cols),
              mSubDiag(matrix.n_rows > 1 ? matrix.n_rows - 1 : 1),
              mHouseholderCoefficients(matrix.n_cols > 1 ? matrix.n_cols - 1 : 1) {
        compute(matrix, computeEigenvectors);
    }

    HermitianEigenSolver &compute(const MatrixType &matrix, bool computeEigenvectors = true);


    const MatrixType &eigenvectors() const {
        return mEigenVectors;
    }

    const RealVectorType &eigenvalues() const {
        return mEigenValues;
    }

    static const Index mMaxIterations = 30;

protected:
    MatrixType mEigenVectors;
    VectorType mWorkspace;
    RealVectorType mEigenValues;
    RealVectorType mSubDiag;
    VectorType mHouseholderCoefficients;
};

template<typename MatrixType>
HermitianEigenSolver<MatrixType> &
HermitianEigenSolver<MatrixType>::compute(const MatrixType &matrix, bool computeEigenvectors) {
    Index n = matrix.n_cols;
    mEigenValues.resize(n);

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
    mSubDiag.resize(n - 1);
    mHouseholderCoefficients.resize(n - 1);
    tridiagonalization(mat, diag, mSubDiag, mHouseholderCoefficients, mWorkspace, computeEigenvectors);
    compute_from_tri_diag(diag, mSubDiag, mMaxIterations, computeEigenvectors, mEigenVectors);
    mEigenValues *= scale;
    return *this;
}

template<typename RealScalar, typename Scalar>
static void tri_diag_qr_step(RealScalar *diag, RealScalar *subDiag, Index start, Index end, Scalar *matrixQ, Index n) {
    // Wilkinson Shift
    RealScalar td = (diag[end - 1] - diag[end]) * RealScalar(0.5);
    RealScalar e = subDiag[end - 1];
    // Note that thanks to scaling, e^2 or td^2 cannot overflow, however they can still underflow thus leading to inf/NaN values
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
    RealScalar z = subDiag[start];
    // If z ever becomes zero, the Givens rotation will be the identity and z will stay zero for all future iterations.
    for (Index k = start; k < end && z != RealScalar(0); ++k) {
        GivensRotation<RealScalar> rot;
        rot.makeGivens(x, z);

        // do T = G' T G
        RealScalar sdk = rot.s() * diag[k] + rot.c() * subDiag[k];
        RealScalar dkp1 = rot.s() * subDiag[k] + rot.c() * diag[k + 1];

        diag[k] =
                rot.c() * (rot.c() * diag[k] - rot.s() * subDiag[k]) -
                rot.s() * (rot.c() * subDiag[k] - rot.s() * diag[k + 1]);
        diag[k + 1] = rot.s() * sdk + rot.c() * dkp1;
        subDiag[k] = rot.c() * sdk - rot.s() * dkp1;

        if (k > start)
            subDiag[k - 1] = rot.c() * subDiag[k - 1] - rot.s() * z;

        // "Chasing the bulge" to return to Hessenberg form.
        x = subDiag[k];
        if (k < end - 1) {
            z = -rot.s() * subDiag[k + 1];
            subDiag[k + 1] = rot.c() * subDiag[k + 1];
        }

        // apply the givens rotation to the unit matrix Q = Q * G
        if (matrixQ) {
            apply_givens_right(matrixQ, n, k, k + 1, rot);
        }
    }
}

template<typename MatrixType, typename RealVectorType>
void compute_from_tri_diag(RealVectorType &diag, RealVectorType &subDiag,
                           Index maxIterations, bool computeEigenvectors,
                           MatrixType &eigenVectors) {
    typedef typename MatrixType::elem_type Scalar;
    typedef typename arma::get_pod_type<Scalar>::result RealScalar;
    Index n = diag.size();
    Index end = n - 1;
    Index start = 0;
    Index iter = 0; // total number of iterations

    const RealScalar considerAsZero = (std::numeric_limits<RealScalar>::min)();
    const RealScalar precision_inv = RealScalar(1) / (std::numeric_limits<RealScalar>::epsilon)();
    while (end > 0) {
        for (Index i = start; i < end; ++i) {
            if (std::abs(subDiag[i]) < considerAsZero) {
                subDiag[i] = RealScalar(0);
            } else {
                // abs(subDiag[i]) <= epsilon * sqrt(abs(diag[i]) + abs(diag[i+1]))
                // Scaled to prevent underflow.
                const RealScalar scaled_sub_diag = precision_inv * subDiag[i];
                if (scaled_sub_diag * scaled_sub_diag <= (std::abs(diag[i]) + std::abs(diag[i + 1]))) {
                    subDiag[i] = RealScalar(0);
                }
            }
        }

        // find the largest unreduced block at the end of the matrix.
        while (end > 0 && (subDiag[end - 1] == RealScalar(0))) {
            end--;
        }
        if (end <= 0)
            break;

        // if we spent too many iterations, we give up
        iter++;
        if (iter > maxIterations * n)
            break;

        start = end - 1;
        while (start > 0 && (subDiag[start - 1] != RealScalar(0)))
            start--;

        tri_diag_qr_step(
                diag.memptr(), subDiag.memptr(), start, end,
                computeEigenvectors ? eigenVectors.memptr() : (Scalar *) 0, n);
    }
}

#endif // HERMITIAN_EIGEN_SOLVER_H
