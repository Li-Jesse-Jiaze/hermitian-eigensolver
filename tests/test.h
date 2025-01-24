#include <iostream>
#include <typeinfo>
#include <armadillo>
#include "HermitianEigenSolver.h"

template<typename RealScalar, typename Scalar>
arma::Mat<Scalar> generate_hermitian(arma::Col<RealScalar> eigen_values) {
    arma::uword n = eigen_values.n_elem;
    arma::Mat<Scalar> D(n, n, arma::fill::zeros);
    D.diag() = arma::conv_to<arma::Col<Scalar>>::from(eigen_values);
    auto rand = arma::randn<arma::Mat<Scalar>>(n, n);
    arma::Mat<Scalar> Q, _;
    arma::qr(Q, _, rand);
    arma::Mat<Scalar> A = Q * D * Q.t();
    return A;
}

template<typename RealScalar>
RealScalar compute_eigenvalue_error(const arma::Col<RealScalar> &expected_eigenvalues,
                                    const arma::Col<RealScalar> &computed_eigenvalues) {
    arma::Col<RealScalar> sorted_expected = arma::sort(expected_eigenvalues);
    arma::Col<RealScalar> sorted_computed = arma::sort(computed_eigenvalues);
    return arma::max(arma::abs(sorted_expected - sorted_computed) / arma::abs(sorted_expected));
}

template<typename RealScalar, typename Scalar>
RealScalar compute_eigenvector_error(const arma::Mat<Scalar> &matrix,
                                     const arma::Col<RealScalar> &eigenvalues,
                                     const arma::Mat<Scalar> &eigenvectors) {
    Index n = matrix.n_cols;
    arma::Mat<Scalar> D(n, n, arma::fill::zeros);
    D.diag() = arma::conv_to<arma::Col<Scalar>>::from(eigenvalues);
    arma::Mat<Scalar> B = eigenvectors * D * eigenvectors.t();
    return arma::norm(matrix - B, "inf");
}

template<typename Scalar>
void test_hard() {
    typedef typename arma::get_pod_type<Scalar>::result RealScalar;
    // Arma Limit test
    {
        arma::Col<RealScalar> eigenvalues(10, arma::fill::ones);
        // eigenvalues.t().print("Expected Eigenvalues:");
        while (true) {
            eigenvalues(9) /= 10;
            auto A = generate_hermitian<RealScalar, Scalar>(eigenvalues);
            arma::Col<RealScalar> ev;
            arma::eig_sym(ev, A);
            // ev.t().print("Arma:");
            if (compute_eigenvalue_error(eigenvalues, ev) > 1e-3) {
                std::cout << "Armadillo Limit:" << eigenvalues(9) << std::endl;
                break;
            }
        }
    }
    // Custom Limit test
    {
        arma::Col<RealScalar> eigenvalues(10, arma::fill::ones);
        while (true) {
            eigenvalues(9) /= 10;
            auto A = generate_hermitian<RealScalar, Scalar>(eigenvalues);
            HermitianEigenSolver hes(A);
            // hes.eigenvalues().t().print("HES");
            if (compute_eigenvalue_error(eigenvalues, hes.eigenvalues()) > 1e-3) {
                std::cout << "Custom Limit:" << eigenvalues(9) << std::endl;
                break;
            }
        }
    }
}

template<typename Scalar>
void test_time_vectors(Index n) {
    typedef typename arma::get_pod_type<Scalar>::result RealScalar;
    constexpr int iterations = 5;

    arma::Col<RealScalar> eigenvalues(n, arma::fill::randn);
    auto A = generate_hermitian<RealScalar, Scalar>(eigenvalues);

    std::vector<double> arma_times, my_times;
    std::vector<double> arma_value_errors, arma_vector_errors;
    std::vector<double> my_value_errors, my_vector_errors;

    for (int i = 0; i < iterations; ++i) {
        arma::wall_clock timer;

        // Compute with Armadillo
        arma::Col<RealScalar> ev;
        arma::Mat<Scalar> evs;
        timer.tic();
        arma::eig_sym(ev, evs, A);
        arma_times.push_back(timer.toc());
        // arma_value_errors.push_back(compute_eigenvalue_error(eigenvalues, ev));
        arma_vector_errors.push_back(compute_eigenvector_error(A, ev, evs));

        // Compute with HermitianEigenSolver
        timer.tic();
        HermitianEigenSolver hes(A);
        my_times.push_back(timer.toc());
        // my_value_errors.push_back(compute_eigenvalue_error(eigenvalues, hes.eigenvalues()));
        my_vector_errors.push_back(compute_eigenvector_error(A, hes.eigenvalues(), hes.eigenvectors()));
    }

    auto compute_average = [](const std::vector<double>& values) {
        return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    };

    std::cout << "Armadillo time: " << compute_average(arma_times) << " seconds\n";
    // std::cout << "Armadillo eigenvalue error: " << compute_average(arma_value_errors) << "\n";
    std::cout << "Armadillo eigenvector error: " << compute_average(arma_vector_errors) << "\n";

    std::cout << "Custom solver time: " << compute_average(my_times) << " seconds\n";
    // std::cout << "Custom solver eigenvalue error: " << compute_average(my_value_errors) << "\n";
    std::cout << "Custom solver eigenvector error: " << compute_average(my_vector_errors) << "\n";
}

template<typename Scalar>
void test_time_values(Index n) {
    typedef typename arma::get_pod_type<Scalar>::result RealScalar;
    constexpr int iterations = 5;

    arma::Col<RealScalar> eigenvalues(n, arma::fill::randn);
    auto A = generate_hermitian<RealScalar, Scalar>(eigenvalues);

    std::vector<double> arma_times, my_times;
    std::vector<double> arma_value_errors;
    std::vector<double> my_value_errors;

    for (int i = 0; i < iterations; ++i) {
        arma::wall_clock timer;

        // Compute with Armadillo
        arma::Col<RealScalar> ev;
        timer.tic();
        arma::eig_sym(ev, A);
        arma_times.push_back(timer.toc());
        arma_value_errors.push_back(compute_eigenvalue_error(eigenvalues, ev));

        // Compute with HermitianEigenSolver
        timer.tic();
        HermitianEigenSolver hes(A, false);
        my_times.push_back(timer.toc());
        my_value_errors.push_back(compute_eigenvalue_error(eigenvalues, hes.eigenvalues()));
    }

    auto compute_average = [](const std::vector<double>& values) {
        return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    };

    std::cout << "Armadillo time: " << compute_average(arma_times) << " seconds\n";
    std::cout << "Armadillo eigenvalue error: " << compute_average(arma_value_errors) << "\n";

    std::cout << "Custom solver time: " << compute_average(my_times) << " seconds\n";
    std::cout << "Custom solver eigenvalue error: " << compute_average(my_value_errors) << "\n";
}