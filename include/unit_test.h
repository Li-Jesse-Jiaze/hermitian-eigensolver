#include <iostream>
#include <armadillo>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <HermitianEigenSolver.h>

void test_test() {
    // Matrix size
    constexpr arma::uword n = 5;

    // Generate a random Hermitian (complex symmetric) matrix
    arma::cx_mat A = arma::randu<arma::cx_mat>(n, n);
    A = 0.5 * (A + A.t());  // Make it Hermitian

    A.print("A:");

    // Backup the original matrix for comparison
    arma::cx_mat A_copy = A;

    HermitianEigenSolver<arma::cx_mat> hs(A);
    hs.eigenvalues().print();
    hs.eigenvectors().print();
    // Perform tridiagonalization using Eigen
    Eigen::MatrixXcd eigenA = Eigen::MatrixXcd::Map(reinterpret_cast<std::complex<double>*>(A_copy.memptr()), n, n);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es;
    es.compute(eigenA);
    std::cout << es.eigenvalues() << std::endl;
    std::cout << es.eigenvectors() << std::endl;

    arma::cx_vec arma_ev;
    arma::cx_mat arma_evs;
    arma::eig_gen(arma_ev, arma_evs, A_copy);
    arma_ev.print();
    arma_evs.print();
}