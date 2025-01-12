#include <iostream>
#include <armadillo>
#include <Eigen/Dense>
#include "GivensRotation.h"

template <typename Scalar>
void testGivensRotationConsistency()
{
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    // Generate random test values for p and q
    Scalar p = static_cast<Scalar>(rand()) / RAND_MAX * 10 - 5; // Random value between -5 and 5
    Scalar q = static_cast<Scalar>(rand()) / RAND_MAX * 10 - 5; // Random value between -5 and 5

    // Compute Givens rotation using GivensRotation
    GivensRotation<Scalar> givens;
    Scalar r1;
    givens.makeGivens(p, q, &r1);

    // Compute Givens rotation using Eigen
    Eigen::JacobiRotation<Scalar> eigenGivens;
    Scalar r2;
    eigenGivens.makeGivens(p, q, &r2);

    // Compare the results
    std::cout << "Testing Givens Rotation for Scalar type " << typeid(Scalar).name() << ":\n";
    std::cout << "p: " << p << ", q: " << q << "\n";
    std::cout << "GivensRotation: c = " << givens.c() << ", s = " << givens.s() << ", r = " << r1 << "\n";
    std::cout << "Eigen:          c = " << eigenGivens.c() << ", s = " << eigenGivens.s() << ", r = " << r2 << "\n";

    // Validate consistency
    if (std::abs(givens.c() - eigenGivens.c()) < 1e-6 &&
        std::abs(givens.s() - eigenGivens.s()) < 1e-6 &&
        std::abs(r1 - r2) < 1e-6)
    {
        std::cout << "Results are consistent.\n";
    }
    else
    {
        std::cout << "Results are NOT consistent.\n";
    }
    std::cout << "---------------------------------------------\n";
}

template <typename Scalar>
void testApplyGivensRight()
{
    using MatrixType = arma::Mat<std::complex<Scalar>>;

    // Generate a random complex matrix
    MatrixType matrix = arma::randu<MatrixType>(4, 4) + arma::randu<MatrixType>(4, 4) * std::complex<Scalar>(0, 1);

    // Compare the matrices
    std::cout << "Testing applyGivensRight for complex Scalar type " << typeid(Scalar).name() << ":\n";
    std::cout << "Original Matrix (Armadillo):\n" << matrix << "\n";

    // Choose two random columns
    arma::uword p = rand() % matrix.n_cols;
    arma::uword q = (p + 1) % matrix.n_cols;

    // Generate random Givens rotation
    Scalar p_val = static_cast<Scalar>(rand()) / RAND_MAX * 10 - 5;
    Scalar q_val = static_cast<Scalar>(rand()) / RAND_MAX * 10 - 5;

    GivensRotation<Scalar> givens;
    givens.makeGivens(p_val, q_val);

    // Copy matrix for comparison with Eigen
    Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix(4, 4);
    for (arma::uword i = 0; i < 4; ++i)
        for (arma::uword j = 0; j < 4; ++j)
            eigenMatrix(i, j) = matrix(i, j);

    // Apply Givens rotation using custom implementation
    applyGivensRight(matrix.memptr(), matrix.n_cols, p, q, givens);

    // Apply Givens rotation using Eigen
    Eigen::JacobiRotation<Scalar> eigenGivens;
    eigenGivens.makeGivens(p_val, q_val, 0);
    eigenMatrix.applyOnTheRight(p, q, eigenGivens);

    std::cout << "Transformed Matrix (Armadillo):\n" << matrix << "\n";
    std::cout << "Transformed Matrix (Eigen):\n" << eigenMatrix << "\n";

    bool consistent = true;
    for (arma::uword i = 0; i < 4; ++i)
    {
        for (arma::uword j = 0; j < 4; ++j)
        {
            if (std::abs(matrix(i, j) - eigenMatrix(i, j)) > 1e-6)
            {
                consistent = false;
                break;
            }
        }
        if (!consistent)
            break;
    }

    if (consistent)
    {
        std::cout << "applyGivensRight results are consistent.\n";
    }
    else
    {
        std::cout << "applyGivensRight results are NOT consistent.\n";
    }
    std::cout << "---------------------------------------------\n";
}

int main()
{
    // Seed random number generator
    srand(static_cast<unsigned>(time(0)));

    // Test with float and double
    testGivensRotationConsistency<float>();
    testGivensRotationConsistency<double>();

    testApplyGivensRight<float>();
    testApplyGivensRight<double>();

    return 0;
}
