"""The mathematical operations for matrix Lie groups and Lie algebras.

Implement here the Lie group functions your kinematics and controller need,
split into small elementary functions. The four functions below must keep
their names and signatures: the provided code imports `so3_exp`, `unskew` and
`se3_inverse`, and `se3_exp` is its natural counterpart for the kinematics.
Add any other functions you find useful.
"""

import numpy as np

# Below this norm an angular velocity counts as zero. The exponential of a
# twist divides by the squared norm of its angular part, which is what makes
# the distinction necessary.
ANGULAR_TOLERANCE = 1e-12


def unskew(omega_tilde) -> np.ndarray:
    """Coordinate vector of a skew-symmetric matrix: so(3) -> R^3.

    Args:
        omega_tilde: a skew-symmetric 3x3 matrix.

    Returns:
        The vector omega with skew(omega) = omega_tilde, an array of 3 numbers.
    """
    raise NotImplementedError("unskew is not implemented yet")


def so3_exp(omega) -> np.ndarray:
    """Exponential map of SO(3): R^3 -> SO(3), by Rodrigues' formula.

    Args:
        omega: the exponential coordinates of a rotation, 3 numbers.

    Returns:
        The rotation matrix exp(omega_tilde), 3x3.
    """
    raise NotImplementedError("so3_exp is not implemented yet")


def se3_exp(twist) -> np.ndarray:
    """Exponential map of SE(3): R^6 -> SE(3).

    Args:
        twist: T = (omega, v), 6 numbers.

    Returns:
        The homogeneous matrix exp(T_tilde), 4x4.
    """
    raise NotImplementedError("se3_exp is not implemented yet")


def se3_inverse(H) -> np.ndarray:
    """Inverse of a homogeneous matrix, computed in closed form.

    Args:
        H: a 4x4 homogeneous matrix.

    Returns:
        The inverse of H, 4x4.
    """
    raise NotImplementedError("se3_inverse is not implemented yet")
