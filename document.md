# Solving Hermitian Eigenvalue Problems

*Jiaze Li, 4742380, Scientific Computing*

## Method

Let $A$ be a Hermitian (or symmetric) $n \times n$ matrix. A scalar $\lambda$ is called an **eigenvalue** and a nonzero column vector $z$ the corresponding **eigenvector** if $Az = \lambda z$. $\lambda$ is always real in our case since $A$ is Hermitian. 

The basic task of the Hermitian eigenvalue problems routines is to compute values of $\lambda$ and, optionally, corresponding vectors $z$ for a given matrix $A$. Therefore we would like to reformulate the problem as a eigen decomposition $A = Z \Lambda Z^*$, where $\Lambda$ is a diagonal matrix with eigenvalues on the diagonal and $Z$ is a unitary matrix with each column as an eigenvector. This computation proceeds in the following stages:

1. Reduce the Hermitian matrix $A$ into (real) tridiagonal form $T$ i.e. $A = U T U^*$;
2. Solve the eigen decomposition of $T$ i.e. $T = Q \Lambda Q^\top$. $\Lambda$ contains the eigenvalues of $T$, which are also the eigenvalues of $A$, since the first step is a similarity transformation;
3. (Optional) The eigenvectors of $A$ are the columns of $Z = U Q$. (If there is no need to compute eigenvectors, there is no need to store $U$ and $Q$)

### Householder Tridiagonalization

If $A$ is Hermitian, we construct a unitary matrix $U$ s.t. 
$$
U^* A U = T
$$
is tridiagonal. Using Householder to construct $U$ is a very efficient method. Suppose that Householder matrices $H_1, ..., H_{k-1}$ has been determined s.t. if
$$
A_{k-1}=\left(H_1 \cdots H_{k-1}\right)^* A\left(H_1 \cdots H_{k-1}\right),
$$
then
$$
A_{k-1} =
\begin{bmatrix}
B_{11} & B_{12} & 0 \\
B_{21} & B_{22} & B_{23} \\
0 & B_{32} & B_{33}
\end{bmatrix}
\quad
\begin{array}{c}
_{k-1} \\
_{1} \\
_{n-k}
\end{array}
$$
is tridiagonal through its first $k-1$ columns. If $\tilde{H}_k$ is an $(n-k)$-order Householder matrix s,t, $\tilde{H}_k^*B_{32}$ has the form $[\beta, 0, ..., 0]^\top$, $\beta \in \mathbb{R}$ and if $H_k = \text{diag}(I_k, \tilde{H}_k)$, then
$$
A_k = H_k^* A_{k-1} H_k =
\begin{bmatrix}
B_{11} & B_{12} & 0 \\
B_{21} & B_{22} & B_{23} \tilde{H}_k \\
0 & \tilde{H}_k^* B_{32} & \tilde{H}_k^* B_{33} \tilde{H}_k
\end{bmatrix}
\quad
\begin{array}{c}
_{k-1} \\
_1 \\
_{n-k}
\end{array}
$$
is tridiagonal through its first $k$ columns. Therefore, if $U = H_1 \cdots H_{n-2}$, then $U^* A U = T$ is real tridiagonal.

During the processing, it is important to keep matrix $\tilde{H}_k^* B_{33} \tilde{H}_k$ still Hermitian. Although this is theoretically certain, for better numerical stability we update it in the following way (rank-2 update). To be specific, as a householder reflection, $\tilde{H}_k$ has the form
$$
\tilde{H}_k=I-\tau u u^*, \quad \tau=2 / u^* u, \quad 0 \neq u \in \mathbb{C}^{n-k} .
$$
Note that if $p=\tau^* B_{33} u$ and $w=p-\left(\tau p^* u / 2\right) u$, then

$$
\tilde{H}_k^* B_{33} \tilde{H}_k=B_{33}-u w^*-w u^* .
$$

### QR Iteration with Shifts and Deflation

Once the tridiagonal matrix $T$ is obtained, we construct a orthogonal matrix $Q$ s.t. 
$$
Q^\top T Q = \Lambda
$$
is diagonal. We use QR iteration to construct $Q$ and compute $\Lambda$, or in other words, Givens rotation
$$
T'=G_{\text{all}}^\top T G_{\text{all}},\quad G_{\text{all}}=G_{1, 2} \cdots G_{n-1, n}
$$
until it converge to $\Lambda$.

To ensure convergence, we use the (implicit) Wilkinson shift (2.5.11 Theorem).

To reduce the computational effort, we detect the largest unreduced part before each QR step, specifically the part of the sub-diagonal that is non-zero. In the program, it's hard to actually iterate until the sub-diagonal is exactly 0, so we set
$$
h_{i, i-1}=0, \quad \text { where } \quad\left|h_{i, i-1}\right| \leq \text{eps}\left(\left|h_{i, i}\right|+\left|h_{i-1, i-1}\right|\right)
$$

## Implementation Details

> "Computer programming is an art, because it applies accumulated knowledge to the world, because it requires skill and ingenuity, and especially because it produces objects of beauty."
> — Donald Knuth

The detailed implementation is shown in the code and its comments. Only some details for reducing computation and memory usage or improving numerical stability are documented here.

### Templates

The entire program is implemented through template classes or template functions. e.g.

```C++
template<typename MatrixType>
class HermitianEigenSolver.
```

The advantage of using classes is that all memory, including that needed for temporary variables, is requested at the beginning (see Workspace).

The advantage of using templates is that the entire eigenproblem solver can be applied to a wide variety of dense matrices, real symmetric or Hermitian matrices, both single and double precision. There is also the more important point that many subfunctions can take not only `arma::Mat<type>` and its references as arguments, but also `arma::subview<Scalar>` and its references as inputs, which reduces the need for a lot of copying and the use of temporary variables.

### Workspace

At instance initialization, we request a complex vector of size $2N$ as a workspace

```C++
mWorkspace(2 * matrix.n_cols)
```

In subsequent computations, many temporary variables may be needed, but these are stored on the memory already requested by the workspace to enable memory reuse as well as control of total memory usage.

### Householder Reflector

For a given vector $v$, construct a Householder Reflector $H$ s.t. $H v = [\beta, 0, ... , 0]^\top$, but instead of explicitly storing $H$, store $\tau$ and $u$ s.t. $H = I-\tau u u^*$

```C++
template<typename VectorType, typename Scalar, typename RealScalar>
void make_householder(VectorType &vector, Scalar &tau, RealScalar &beta);
```

Here it is possible to reuse `vector` both as input $v$ and as output $u$.

Note that here $u$ actually has the special structure $[1, ...] ^\top$, so the first position of the `vector` is not important and may used for $\tau$. However, we didn't do that because 1) it's not critical to use just one more scalar, the point is to reduce computation and memory usage of $N$ or $N^2$, and 2) when using `make_householder` the `vector` is a reference to $B_{32 }$, whose first bit should also correctly store the sub-diagonal elements. But this structure is used as below.

If we are required to solve for eigenvectors, we should also construct the matrix $Q = H_{n-2} \cdots H_1$, and since there is no display to store $H$, we need the following function s.t. $A = H A$

```C++
template<typename MatrixType, typename VectorType, typename Scalar>
void apply_householder_left(MatrixType &A, const VectorType &v, const Scalar &tau, Scalar *workspace);
```

Note that the `v` here is not the full $u$, but is missing the first element. Still it is a reference from $B_{32}[1:]$, whose first bit should correctly store the sub-diagonal elements. Also trying to add a element to the back of a vector is easy, but adding one to the front is very tricky.

### Givens Rotation

Constructing a Givens rotation s.t. $G^\top (a, b)^\top = (r, 0)^\top$. The case of complex numbers is not considered here, since the tridiagonal matrix for which the Givens rotations are to be computed are already real. But it needs to be able to be applied to complex matrices.

```C++
template<typename Scalar>
inline void make_givens(const Scalar &a, const Scalar &b, Scalar &c, Scalar &s);
```

This time, for computing the eigenvectors, we no longer need to use $G$ to left-multiply $A$, but instead let it right-multiply $Q = Q G$

```C++
template<typename Scalar, typename RealScalar>
void apply_givens_right(Scalar *matrix_ptr, Index n_rows, Index p, Index q, const RealScalar &c, const RealScalar &s);
```

Only column-major order pointers to dense matrices are accepted here, the purpose of using pointers is to allow further less passing of parameters compared to `arma::Mat` references.

### Tridiagonalization

Tridiagonalization is implemented by the following functions, where the main and sub diagonals are returned by `diag` and `sub_diag`, and by $Q$ is returned by `A` if it needs to be constructed. Instead of returning the tridiagonal matrix as `arma::Mat`, only the diagonal and sub-diagonal vectors are used here to conserve memory usage.

```C++
template<typename MatrixType, typename RealVectorType, typename VectorType>
void tridiagonalization(MatrixType &A, RealVectorType &diag, RealVectorType &sub_diag,
                        VectorType &workspace, bool withQ);
```

As mentioned before, we keep tridiagonalizing $B_{32}$ using Householder. When updating the matrix, we don't care about the result of $B_{23} \tilde{H}_k$, but only about $\tilde{H}_k^* B_{33} \tilde{H}_k$. There are some details in the exact implementation:

```c++
VectorType householders(workspace.memptr(), n - 1, false, false);
```

This variable is constructed on `workspace`, and at step $k$, its first $k-1$ bits store the previous $\tau$, while the remainder is used for the temporary variable $w$ (see rank-2 update).

```c++
// w = conj(tau) * A[i+1:, i+1:] * u
householders.tail(remain) = subA * (scalar_conj(tau) * tail);
```

Here you must first do the scalar product to vector, then matrix multiply, because scalar product to matrix is much more computationally cost than scalar product to vector. Writing directly following the formula will increase the computation due to the fact that `*` is left associative.

```
arma::Mat<Scalar> uw = tail * householders.tail(remain).t();
subA -= (uw + uw.t()).
```

For $\tilde{H}_k^* B_{33} \tilde{H}_k=B_{33}-u w^*-w u^*$, but don't compute $uw^*$ and $wu^*$ twice as they are conjugate transposed. Also here it is possible to update only the upper diagonal of $B_{33}$ in recovering the complete matrix, but I haven't found a good way.

### Scaling

Before starting, we scale the matrix to ensure no overflow.

```C++
mat = matrix;
RealScalar scale = arma::abs(mat).max();
if (scale == RealScalar(0))
    scale = RealScalar(1);
mat /= scale;
...
mEigenValues *= scale;
```

### Deflation

We perform each QR step only on the largest unreduced block, specifically, pointing out where the QR step starts and ends based on two indices.

```c++
for (Index i = start; i < end; ++i) {
    if (std::abs(sub_diag[i]) <= eps * (std::abs(diag[i]) + std::abs(diag[i + 1])))
        sub_diag[i] = RealScalar(0);
}
// find the largest unreduced block
while (end > 0 && (sub_diag[end - 1] == RealScalar(0)))
    end--;
if (end <= 0)
    break;
start = end - 1;
while (start > 0 && (sub_diag[start - 1] != RealScalar(0)))
    start--;
```

## QR Step

A complete QR step is implemented by the following function

```C++
template<typename RealScalar, typename Scalar>
static void tri_diag_qr_step(RealScalar *diag, RealScalar *sub_diag, Index start, Index end, Scalar *Q, Index n);
```

When performing the Wilkinson shift

```C++
RealScalar td = (diag[end - 1] - diag[end]) * RealScalar(0.5);
RealScalar e = sub_diag[end - 1];
RealScalar mu = diag[end]; if (td == RealScalar[end] - diag[end])
if (td == RealScalar(0)) {
    mu -= std::abs(e); } else if (e !
} else if (e ! = RealScalar(0)) {
    const RealScalar e2 = e * e;
    if (e2 == RealScalar(0)) {
        mu -= e / ((td + (td > RealScalar(0) ? h : -h)) / e);
    } else {
        mu -= e2 / (td + (td > RealScalar(0) ? h : -h)); } else { mu -= e2 / (td + (td > RealScalar(0) ? h : -h)); }
    }
}
```

Instead of directly using the formula given in 2.5.7 Lemma
$$
\sigma=a_n+d-\operatorname{sign}(d) \sqrt{d^2+b_{n-1}^2}, \quad d=\frac{a_{n-1}-a_n}{2}.
$$
It is because although after Scaling $d^2$ and $b_{n-1}^2$ will not overflow, it may cause underflow because it is too small.

## Verification and Testing



## Results



## Conclusion