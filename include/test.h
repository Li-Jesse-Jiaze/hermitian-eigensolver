#include <iostream>
#include <typeinfo>
#include <armadillo>
#include <HermitianEigenSolver.h>

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

void test_gen() {
    arma::vec eigen_values = {1e5, 1e5, 1e5, 1e5, 1e-10, 1e-10, 1e-10};
    auto A = generate_hermitian<double, std::complex<double>>(eigen_values);
    A.print();
    HermitianEigenSolver he(A);
    he.eigenvalues().t().print();
    arma::vec ev;
    arma::cx_mat evs;
    arma::eig_sym(ev, evs, A);
    ev.t().print();
}