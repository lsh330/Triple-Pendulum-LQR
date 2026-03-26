import numpy as np
from numba import njit


@njit(cache=True)
def _det3(a, b, c, d, e, f, g, h, i):
    """Determinant of a 3x3 matrix via Sarrus' rule."""
    return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)


@njit(cache=True)
def solve_acceleration(M, rhs):
    """Solve M * x = rhs for a 4x4 system using cofactor expansion.

    Computes x = adj(M) * rhs / det(M) to avoid LAPACK overhead.
    """
    m00 = M[0, 0]; m01 = M[0, 1]; m02 = M[0, 2]; m03 = M[0, 3]
    m10 = M[1, 0]; m11 = M[1, 1]; m12 = M[1, 2]; m13 = M[1, 3]
    m20 = M[2, 0]; m21 = M[2, 1]; m22 = M[2, 2]; m23 = M[2, 3]
    m30 = M[3, 0]; m31 = M[3, 1]; m32 = M[3, 2]; m33 = M[3, 3]

    # 3x3 cofactors for first-row expansion (also reused in adjugate)
    # Cofactor C_ij = (-1)^(i+j) * minor_ij
    # Minor for (0,0): rows 1,2,3; cols 1,2,3
    A00 = _det3(m11, m12, m13, m21, m22, m23, m31, m32, m33)
    # Minor for (0,1): rows 1,2,3; cols 0,2,3
    A01 = -_det3(m10, m12, m13, m20, m22, m23, m30, m32, m33)
    # Minor for (0,2): rows 1,2,3; cols 0,1,3
    A02 = _det3(m10, m11, m13, m20, m21, m23, m30, m31, m33)
    # Minor for (0,3): rows 1,2,3; cols 0,1,2
    A03 = -_det3(m10, m11, m12, m20, m21, m22, m30, m31, m32)

    det = m00*A00 + m01*A01 + m02*A02 + m03*A03

    # Remaining cofactors for the adjugate matrix
    # Row 1 cofactors
    A10 = -_det3(m01, m02, m03, m21, m22, m23, m31, m32, m33)
    A11 = _det3(m00, m02, m03, m20, m22, m23, m30, m32, m33)
    A12 = -_det3(m00, m01, m03, m20, m21, m23, m30, m31, m33)
    A13 = _det3(m00, m01, m02, m20, m21, m22, m30, m31, m32)

    # Row 2 cofactors
    A20 = _det3(m01, m02, m03, m11, m12, m13, m31, m32, m33)
    A21 = -_det3(m00, m02, m03, m10, m12, m13, m30, m32, m33)
    A22 = _det3(m00, m01, m03, m10, m11, m13, m30, m31, m33)
    A23 = -_det3(m00, m01, m02, m10, m11, m12, m30, m31, m32)

    # Row 3 cofactors
    A30 = -_det3(m01, m02, m03, m11, m12, m13, m21, m22, m23)
    A31 = _det3(m00, m02, m03, m10, m12, m13, m20, m22, m23)
    A32 = -_det3(m00, m01, m03, m10, m11, m13, m20, m21, m23)
    A33 = _det3(m00, m01, m02, m10, m11, m12, m20, m21, m22)

    # adj(M) = cofactor matrix transposed, so adj[i,j] = A[j,i]
    # x = adj(M) * rhs / det
    if abs(det) < 1e-30:
        # Mass matrix near-singular: return zeros to avoid NaN propagation
        x = np.zeros(4)
        return x

    inv_det = 1.0 / det
    r0 = rhs[0]; r1 = rhs[1]; r2 = rhs[2]; r3 = rhs[3]

    x = np.empty(4)
    x[0] = (A00*r0 + A10*r1 + A20*r2 + A30*r3) * inv_det
    x[1] = (A01*r0 + A11*r1 + A21*r2 + A31*r3) * inv_det
    x[2] = (A02*r0 + A12*r1 + A22*r2 + A32*r3) * inv_det
    x[3] = (A03*r0 + A13*r1 + A23*r2 + A33*r3) * inv_det
    return x
