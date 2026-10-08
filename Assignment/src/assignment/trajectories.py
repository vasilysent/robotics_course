"""Paths for the end-effector to follow.

A trajectory says where the target frame is at a given time and how it is
moving, both analytically. Carrying the twist rather than leaving the
controller to difference the pose matters: the twist is what the damping term
of `control_wrench` subtracts, and a differenced one would be one step stale
and noisy, which shows up as a lag the arm never catches up with.

The twist is the one of the Lie-group convention used throughout, the body
twist read off the matrix H^{-1} H_dot:

    T_d^{d,w} = ( vee(R^T R_dot), R^T p_dot ),

so both halves come from the derivative of the pose, and a trajectory whose
target turns needs no different treatment from one whose target only
translates.

A trajectory is relative: `at` gives the pose of the target with respect to
wherever the path is anchored, and starts at the identity. A path is put
somewhere by multiplying from the left,

    H_d^w(t) = H_start^w H_rel(t),

usually by the pose the end-effector already has, so that the arm begins on the
path with no error and no run-up. The twist survives this untouched: for a
constant H_start,

    H_d^{-1} H_d_dot = H_rel^{-1} H_start^{-1} H_start H_rel_dot
                     = H_rel^{-1} H_rel_dot,

so the body twist of the target does not depend on where the path was put, and
`at` can return it as it stands.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from assignment.student.lie_groups import unskew


class Trajectory(ABC):
    """Where the target frame is, and how it moves, as a function of time."""

    @abstractmethod
    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Pose and twist of the target at time `t`.

        Args:
            t: time in seconds, measured from the start of the trajectory.

        Returns:
            H_rel(t), the pose of the target relative to where the path is
            anchored, as a 4x4 matrix, and T_d^{d,w}, its twist expressed in
            the target frame itself, as 6 numbers (omega, v). The pose at
            t = 0 is the identity; multiply it by the pose of the anchor to
            get the H_d^w that `TargetTracker.step` takes, and pass the twist
            on unchanged.
        """


def body_twist(R, R_dot, p_dot) -> np.ndarray:
    """Assemble the body twist of a frame from the derivatives of its pose.

    This is the second block column of H^{-1} H_dot written out, and it is the
    one place in the module where the convention lives, so that every
    trajectory produces its twist the same way.

    Args:
        R: the rotation of the frame, 3x3.
        R_dot: its time derivative, 3x3.
        p_dot: the velocity of its origin in the world frame, 3 numbers.

    Returns:
        The twist (omega, v), 6 numbers.
    """
    R = np.asarray(R, dtype=float)
    return np.concatenate([unskew(R.T @ np.asarray(R_dot, dtype=float)), R.T @ p_dot])


@dataclass(frozen=True)
class FigureEight(Trajectory):
    """A figure of eight traced by the target frame.

    The curve is a lemniscate of Gerono, written in the frame the path is
    anchored to as

        p(s) = A sin(s) x_hat + (B / 2) sin(2s) y_hat,   s = 2 pi t / T,

    which closes on itself once per period and crosses itself once. The
    crossing is at the origin of the anchor frame and is where the path starts,
    so the pose at t = 0 is the identity and the arm can begin from wherever it
    already is. The curve spans 2A along the long axis and B across, and its
    speed is largest at the crossing and vanishes nowhere, so the arm is kept
    moving throughout.

    The eight lies in the xy-plane of the anchor frame, which for an
    end-effector frame is the plane across the tool: anchored to a tool
    pointing down, the eight is traced flat as if drawn on a table, and to a
    tool pointing horizontally, it stands upright. The target keeps the
    orientation of the anchor throughout, so its angular velocity is zero.

    Attributes:
        long_amplitude: A, half the length of the eight, in metres.
        short_amplitude: B, its full width, in metres.
        period: T, the time for one full traversal, in seconds.
    """

    long_amplitude: float
    short_amplitude: float
    period: float

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        rate = 2.0 * np.pi / self.period
        s = rate * t

        # The curve and its velocity, in the frame the path is anchored to, so
        # x_hat and y_hat are the first two columns of the identity.
        half_width = self.short_amplitude / 2.0
        p = np.array(
            [self.long_amplitude * np.sin(s), half_width * np.sin(2.0 * s), 0.0]
        )
        p_dot = rate * np.array(
            [
                self.long_amplitude * np.cos(s),
                2.0 * half_width * np.cos(2.0 * s),
                0.0,
            ]
        )

        H = np.eye(4)
        H[:3, 3] = p
        return H, body_twist(np.eye(3), np.zeros((3, 3)), p_dot)
