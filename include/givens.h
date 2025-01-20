#ifndef GIVENS_ROTATION_H
#define GIVENS_ROTATION_H

#include "base.h"

/**
 * @brief Make a Givens Rotation s.t. G^T (a, b)^T = (r, 0)^T
 * 
 * @tparam Scalar The type of the input values (should be real)
 * @param[in] a The first value
 * @param[in] b The second value
 * @param[in] c The cosine component of G
 * @param[in] s The sine component of G
 */
template<typename Scalar>
inline void make_givens(const Scalar &a, const Scalar &b, Scalar &c, Scalar &s) {
    using std::abs;
    using std::sqrt;
    if (b == Scalar(0)) {
        c = a < Scalar(0) ? Scalar(-1) : Scalar(1);
        s = Scalar(0);
    } else if (a == Scalar(0)) {
        c = Scalar(0);
        s = b < Scalar(0) ? Scalar(1) : Scalar(-1);
    } else if (abs(a) > abs(b)) {
        Scalar t = b / a;
        Scalar u = sqrt(Scalar(1) + t * t);
        if (a < Scalar(0))
            u = -u;
        c = Scalar(1) / u;
        s = -t * c;
    } else {
        Scalar t = a / b;
        Scalar u = sqrt(Scalar(1) + t * t);
        if (b < Scalar(0))
            u = -u;
        s = -Scalar(1) / u;
        c = -t * s;
    }
}

/**
 * @brief Apply the Givens rotation with c and s to the columns p and q of matrix, i.e. A = A * G
 * 
 * @tparam Scalar The type of the matrix values (can be complex)
 * @tparam OtherScalar The type of the rotation parameters (only real)
 * @param matrix_ptr Pointer to the first element of the matrix (Armadillo dense matrix is stored in column-major order)
 * @param n_rows Number of rows in the matrix
 * @param p Index of the first column
 * @param q Index of the second column
 * @param c The cosine component of G
 * @param s The sine component of G
 */
template<typename Scalar, typename OtherScalar>
void apply_givens_right(Scalar *matrix_ptr, Index n_rows, Index p, Index q, const OtherScalar &c, const OtherScalar &s) {
    if (c == Scalar(1) && s == Scalar(0)) return;

    for (Index i = 0; i < n_rows; ++i) {
        Scalar *row_ptr = matrix_ptr + i;
        Scalar x = row_ptr[p * n_rows];
        Scalar y = row_ptr[q * n_rows];

        row_ptr[p * n_rows] = c * x - s * y;
        row_ptr[q * n_rows] = s * x + c * y;
    }
}

#endif // GIVENS_ROTATION_H
