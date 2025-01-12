#ifndef GIVENS_ROTATION_H
#define GIVENS_ROTATION_H

#include <armadillo>

// Support reals only
template<typename Scalar>
class GivensRotation {
public:
    GivensRotation() = default;
    GivensRotation(const Scalar &c, const Scalar &s) : m_c(c), m_s(s) {}

    Scalar &c() { return m_c; }
    Scalar c() const { return m_c; }
    Scalar &s() { return m_s; }
    Scalar s() const { return m_s; }

    void makeGivens(const Scalar &p, const Scalar &q, Scalar *r = 0);

protected:
    Scalar m_c, m_s;
};

// specialization for reals
template<typename Scalar>
void GivensRotation<Scalar>::makeGivens(const Scalar &p, const Scalar &q, Scalar *r) {
    using std::abs;
    using std::sqrt;
    if (q == Scalar(0)) {
        m_c = p < Scalar(0) ? Scalar(-1) : Scalar(1);
        m_s = Scalar(0);
        if (r)
            *r = abs(p);
    } else if (p == Scalar(0)) {
        m_c = Scalar(0);
        m_s = q < Scalar(0) ? Scalar(1) : Scalar(-1);
        if (r)
            *r = abs(q);
    } else if (abs(p) > abs(q)) {
        Scalar t = q / p;
        Scalar u = sqrt(Scalar(1) + t * t);
        if (p < Scalar(0))
            u = -u;
        m_c = Scalar(1) / u;
        m_s = -t * m_c;
        if (r)
            *r = p * u;
    } else {
        Scalar t = p / q;
        Scalar u = sqrt(Scalar(1) + t * t);
        if (q < Scalar(0))
            u = -u;
        m_s = -Scalar(1) / u;
        m_c = -t * m_s;
        if (r)
            *r = q * u;
    }
}

template<typename Scalar, typename OtherScalar>
void applyGivensRight(Scalar* matrix_ptr, arma::uword nRows, arma::uword p, arma::uword q, const GivensRotation<OtherScalar>& g) {
    Scalar c = g.c();
    Scalar s = g.s();

    if (c == Scalar(1) && s == Scalar(0)) return;

    for (arma::uword i = 0; i < nRows; ++i) {
        Scalar* row_ptr = matrix_ptr + i;
        Scalar x = row_ptr[p * nRows];
        Scalar y = row_ptr[q * nRows];

        row_ptr[p * nRows] = c * x - s * y;
        row_ptr[q * nRows] = s * x + c * y;
    }
}


#endif // GIVENS_ROTATION_H
