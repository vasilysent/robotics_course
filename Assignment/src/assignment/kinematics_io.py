"""Reading and validating the kinematic description of a serial chain.

The description lives in a YAML file; see `student/ur5e_kinematics.yaml` for the
format and for the conventions it follows. In short, for a chain with links
0..n and joints 1..n, where joint i connects link i-1 to link i:

    H_0_w               pose of the base frame Psi_0 relative to the world frame
    H_ref[i]            H_i^{i-1}(0), adjacent links in the reference configuration
    unit_twists[i]      T_i^{i,i-1} = (w, v), unit twist of joint i 
                        expressed in the CHILD frame Psi_i
    H_ee_n              pose of the end-effector frame relative to the last link frame Psi_n

Joints are indexed 1..n in the mathematics and 0..n-1 in the arrays, so
`chain.H_ref[i - 1]` is H_i^{i-1}(0).
"""

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import yaml

# The UR5e robot description file.
UR5E_CHAIN = resources.files("assignment.student") / "ur5e_kinematics.yaml"

# Absolute tolerance for the orthonormality and unit-norm checks. The entries
# are hand-written, so this is deliberately tight: it catches a wrong number,
# not floating-point noise.
TOL = 1e-9


class ChainFormatError(ValueError):
    """The kinematic description is malformed or geometrically inconsistent."""


@dataclass(frozen=True)
class ChainModel:
    """A validated kinematic description of a serial chain."""

    name: str
    joint_names: list[str]
    H_0_w: np.ndarray  # (4, 4)
    H_ref: np.ndarray  # (n, 4, 4)
    unit_twists: np.ndarray  # (n, 6)
    H_ee_n: np.ndarray  # (4, 4)

    @property
    def n(self) -> int:
        """Number of joints."""
        return len(self.joint_names)


def _as_homogeneous(value, where: str) -> np.ndarray:
    """Convert a nested list to a 4x4 element of SE(3), checking that it is one."""
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4):
        raise ChainFormatError(f"{where}: expected a 4x4 matrix, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ChainFormatError(f"{where}: contains a non-finite entry")

    bottom = matrix[3]
    if not np.allclose(bottom, [0.0, 0.0, 0.0, 1.0], atol=TOL):
        raise ChainFormatError(f"{where}: bottom row must be [0, 0, 0, 1], got {bottom}")

    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=TOL):
        raise ChainFormatError(
            f"{where}: the rotation block is not orthogonal, R^T R =\n{rotation.T @ rotation}"
        )
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > TOL:
        raise ChainFormatError(
            f"{where}: the rotation block has determinant {determinant:+.6f}, expected +1 "
            "(a determinant of -1 means the frame is left-handed, check the column signs)"
        )
    return matrix


def _as_unit_twist(value, where: str) -> np.ndarray:
    """Convert a list to a 6-vector twist (w, v), checking that it is a unit twist."""
    twist = np.asarray(value, dtype=float)
    if twist.shape != (6,):
        raise ChainFormatError(f"{where}: expected 6 numbers (w, v), got shape {twist.shape}")
    if not np.all(np.isfinite(twist)):
        raise ChainFormatError(f"{where}: contains a non-finite entry")

    omega, v = twist[:3], twist[3:]
    omega_norm = float(np.linalg.norm(omega))
    if omega_norm > TOL:
        # Rotational or screw joint: the angular part carries the unit.
        if abs(omega_norm - 1.0) > TOL:
            raise ChainFormatError(
                f"{where}: |w| = {omega_norm:.6f}, expected 1 for a rotational or screw joint"
            )
    else:
        # Translational joint: no rotation, so the linear part carries the unit.
        v_norm = float(np.linalg.norm(v))
        if abs(v_norm - 1.0) > TOL:
            raise ChainFormatError(
                f"{where}: w = 0 and |v| = {v_norm:.6f}; a twist with no angular part "
                "describes a translational joint and must have |v| = 1"
            )
    return twist


def load_chain_model(path: str | Path) -> ChainModel:
    """Load and validate a kinematic description from a YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        description = yaml.safe_load(stream)

    if not isinstance(description, dict):
        raise ChainFormatError(f"{path}: expected a mapping at the top level")

    missing = {"H_0_w", "joints", "H_ee_n"} - description.keys()
    if missing:
        raise ChainFormatError(f"{path}: missing top-level key(s) {sorted(missing)}")

    joints = description["joints"]
    if not isinstance(joints, list) or not joints:
        raise ChainFormatError(f"{path}: 'joints' must be a non-empty list, ordered base to tip")

    joint_names: list[str] = []
    H_ref = np.empty((len(joints), 4, 4))
    unit_twists = np.empty((len(joints), 6))

    for index, joint in enumerate(joints):
        number = index + 1  # joints are numbered 1..n
        if not isinstance(joint, dict):
            raise ChainFormatError(f"{path}: joint {number} must be a mapping")
        joint_missing = {"H_ref", "unit_twist"} - joint.keys()
        if joint_missing:
            raise ChainFormatError(f"{path}: joint {number} is missing {sorted(joint_missing)}")

        name = str(joint.get("name", f"joint_{number}"))
        joint_names.append(name)
        H_ref[index] = _as_homogeneous(joint["H_ref"], f"{path}: joint {number} ({name}) H_ref")
        unit_twists[index] = _as_unit_twist(
            joint["unit_twist"], f"{path}: joint {number} ({name}) unit_twist"
        )

    if len(set(joint_names)) != len(joint_names):
        raise ChainFormatError(f"{path}: joint names must be unique, got {joint_names}")

    return ChainModel(
        name=str(description.get("name", path.stem)),
        joint_names=joint_names,
        H_0_w=_as_homogeneous(description["H_0_w"], f"{path}: H_0_w"),
        H_ref=H_ref,
        unit_twists=unit_twists,
        H_ee_n=_as_homogeneous(description["H_ee_n"], f"{path}: H_ee_n"),
    )
