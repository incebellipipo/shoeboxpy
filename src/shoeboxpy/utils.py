import numpy as np
import numpy.typing as npt

def skew(v: np.ndarray) -> np.ndarray:
    r"""
    Return the 3x3 skew-symmetric matrix S(v) such that S(v)*x = v x x.

    If v = [v1, v2, v3], then:

    .. math::
       S(v) = \begin{bmatrix}
              0 & -v_3 &  v_2 \\
              v_3 & 0 & -v_1 \\
              -v_2 & v_1 & 0
              \end{bmatrix}
    """
    return np.array([
        [0.0,    -v[2],   v[1]],
        [v[2],    0.0,   -v[0]],
        [-v[1],  v[0],    0.0]
    ])