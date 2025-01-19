#include <iostream>
#include <armadillo>
#include <HermitianEigenSolver.h>

void test_test() {
    // Matrix size
    constexpr arma::uword n = 5;

    // Generate a random Hermitian (complex symmetric) matrix
    auto A = arma::randu<arma::cx_mat>(n, n);
    A = 0.5 * (A + A.t());  // Make it Hermitian

//    A.print("A:");

    // Backup the original matrix for comparison
    arma::cx_mat A_copy = A;

    arma::wall_clock timer;
    timer.tic();
    HermitianEigenSolver<arma::cx_mat> hs(A, false);
    std::cout << "Mine: " << timer.toc() << std::endl;
    hs.eigenvalues().print();

    arma::vec arma_ev;
    arma::cx_mat arma_evs;
    timer.tic();
    arma::eig_sym(arma_ev, A_copy);
    std::cout << "Arma: " << timer.toc() << std::endl;
    arma_ev.print();
}

void test_time() {
    // Matrix size
    constexpr arma::uword n = 100;
    constexpr int num_iterations = 10;

    double total_time_mine = 0.0;
    double total_time_arma = 0.0;

    for (int i = 0; i < num_iterations; ++i) {
        // Generate a random Hermitian (complex symmetric) matrix
        auto A = arma::randu<arma::cx_mat>(n, n);
        A = 0.5 * (A + A.t());  // Make it Hermitian

        // Backup the original matrix for comparison
        arma::cx_mat A_copy = A;

        arma::wall_clock timer;

        // Test custom HermitianEigenSolver
        timer.tic();
        HermitianEigenSolver<arma::cx_mat> hs(A);
        total_time_mine += timer.toc();

        // Test Armadillo's eig_gen
        arma::vec arma_ev;
        arma::cx_mat arma_evs;
        timer.tic();
        arma::eig_sym(arma_ev, arma_evs, A_copy);
        total_time_arma += timer.toc();
    }

    // Compute average times
    double avg_time_mine = total_time_mine / num_iterations;
    double avg_time_arma = total_time_arma / num_iterations;

    // Print results
    std::cout << "Average time (Mine): " << avg_time_mine << " seconds" << std::endl;
    std::cout << "Average time (Arma): " << avg_time_arma << " seconds" << std::endl;
}
