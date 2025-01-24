#include <armadillo>
#include "test.h"

int main() {
    arma::arma_rng::set_seed(123);
    int sizes[] = {10, 100, 1000};

    // std::vector<std::pair<std::string, void(*)(int)>> test_cases_vv = {
    //         {"[f]", [](int n) { test_time_vectors<float>(n); }},
    //         {"[d]", [](int n) { test_time_vectors<double>(n); }},
    //         {"[cf]", [](int n) { test_time_vectors<std::complex<float>>(n); }},
    //         {"[cd]", [](int n) { test_time_vectors<std::complex<double>>(n); }}
    // };

    // for (int n : sizes) {
    //     std::cout << "Testing with n = " << n << std::endl;
    //     for (const auto& [label, test_func] : test_cases_vv) {
    //         std::cout << label << std::endl;
    //         test_func(n);
    //     }
    // }

    // std::vector<std::pair<std::string, void(*)(int)>> test_cases_v = {
    //         {"[f]", [](int n) { test_time_values<float>(n); }},
    //         {"[d]", [](int n) { test_time_values<double>(n); }},
    //         {"[cf]", [](int n) { test_time_values<std::complex<float>>(n); }},
    //         {"[cd]", [](int n) { test_time_values<std::complex<double>>(n); }}
    // };

    // for (int n : sizes) {
    //     std::cout << "Testing with n = " << n << std::endl;
    //     for (const auto& [label, test_func] : test_cases_v) {
    //         std::cout << label << std::endl;
    //         test_func(n);
    //     }
    // }

    test_hard<float>();
    test_hard<double>();
    test_hard<std::complex<float>>();
    test_hard<std::complex<double>>();

    return 0;
}
