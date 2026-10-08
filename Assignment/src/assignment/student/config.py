"""Settings of the task-space controller."""

import numpy as np

from assignment.student.lie_groups import so3_exp
from assignment.trajectories import FigureEight

INITIAL_CONFIGURATION = np.deg2rad([-90.0, -75.0, 75.0, 0.0, 90.0, 0.0])

# Pose the end-effector frame is pulled towards, H_d^w.
TARGET_POSITION = np.array([0.35, 0.25, 0.55])
TARGET_ORIENTATION = so3_exp(np.array([0.0, np.pi, 0.0]))

TARGET_POSE = np.eye(4)
TARGET_POSE[:3, :3] = TARGET_ORIENTATION
TARGET_POSE[:3, 3] = TARGET_POSITION

# A moving target: a figure of eight, anchored to the pose the end-effector starts in,
# so the arm begins on it. The period sets the speed.
FIGURE_EIGHT = FigureEight(
    long_amplitude=0.30,
    short_amplitude=0.22,
    period=10.0,
)

# Stiffness of the virtual spring in N/m for the translation and Nm/rad for the rotation.
K_TRANS = 600.0
K_ROT = 60.0
K_COUPLING = 0.0

# Damping of the virtual damper, opposing the motion of the end-effector
# relative to the target, in Ns/m and Nms/rad.
D_TRANS = 200.0
D_ROT = 16.0

# The limit norm of force and moment of the virtual spring, in N and Nm. Set either to
# None to remove the limit.
MAX_FORCE = 100.0
MAX_MOMENT = 15.0

# Joint speed guard. A real UR5e refuses to move a joint faster than
# JOINT_SPEED_SAFETY, so the controller brakes a joint that approaches it: the
# braking torque starts at zero at JOINT_SPEED_SOFT, grows with the square of
# the excess speed, and reaches the full torque of the motor at
# JOINT_SPEED_MAX.
JOINT_SPEED_SAFETY = np.pi
JOINT_SPEED_MAX = 0.8 * JOINT_SPEED_SAFETY
JOINT_SPEED_SOFT = 0.6 * JOINT_SPEED_SAFETY
