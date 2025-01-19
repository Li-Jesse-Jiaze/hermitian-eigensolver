#ifndef GIVENS_ROTATION_H
#define GIVENS_ROTATION_H

#include "base.h"

// Support reals only
template<typename Scalar>
inline void make_givens(const Scalar &p, const Scalar &q, Scalar &c, Scalar &s) {
    using std::abs;
    using std::sqrt;
    if (q == Scalar(0)) {
        c = p < Scalar(0) ? Scalar(-1) : Scalar(1);
        s = Scalar(0);
    } else if (p == Scalar(0)) {
        c = Scalar(0);
        s = q < Scalar(0) ? Scalar(1) : Scalar(-1);
    } else if (abs(p) > abs(q)) {
        Scalar t = q / p;
        Scalar u = sqrt(Scalar(1) + t * t);
        if (p < Scalar(0))
            u = -u;
        c = Scalar(1) / u;
        s = -t * c;
    } else {
        Scalar t = p / q;
        Scalar u = sqrt(Scalar(1) + t * t);
        if (q < Scalar(0))
            u = -u;
        s = -Scalar(1) / u;
        c = -t * s;
    }
}

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
