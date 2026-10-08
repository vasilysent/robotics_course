"""Record the arm following a target that traces a figure of eight.

The target trajectory starts at the end-effector pose at the initial configuration.

Rendering needs no window, so this script is run with plain `python` on every
platform:

    python scripts/06_record_figure_eight.py
"""

import mujoco
import numpy as np

from assignment.kinematics_io import UR5E_CHAIN, load_chain_model
from assignment.mujoco_scene import TORQUE_MODEL_PATH, load_model
from assignment.recording import (
    VideoRecorder,
    display_path,
    new_run_directory,
    save_error_plot,
)
from assignment.student import config
from assignment.student.control import TargetTracker
from assignment.student.kinematics import ChainData, update_kinematics
from assignment.visualisation import draw_frame

# How many times the eight is traced.
LAPS = 2

# Where the camera is put.
CAMERA_AZIMUTH = 300.0
CAMERA_ELEVATION = -15.0
CAMERA_DISTANCE = 1.4


def main() -> None:
    chain = load_chain_model(UR5E_CHAIN)
    trajectory = config.FIGURE_EIGHT
    duration = LAPS * trajectory.period

    model = load_model(gravity_compensation=False, path=TORQUE_MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = config.INITIAL_CONFIGURATION
    mujoco.mj_forward(model, data)

    start = ChainData(chain)
    update_kinematics(chain, start, config.INITIAL_CONFIGURATION)
    H_start_w = start.H_ee_w.copy()

    tracker = TargetTracker(model, data, chain)
    target_pose = H_start_w.copy()

    def draw(scene: mujoco.MjvScene) -> None:
        """Add the end-effector and target frames to the rendered scene."""
        draw_frame(scene, tracker.kinematics.H_ee_w)
        draw_frame(scene, target_pose)

    directory = new_run_directory("figure_eight")
    print(f"Recording {duration:.1f} s to {display_path(directory)}")

    peak_torque = np.zeros(model.nu)
    times = []
    errors = []

    with VideoRecorder(
        model,
        directory / "video.mp4",
        lookat=tuple(H_start_w[:3, 3]),
        distance=CAMERA_DISTANCE,
        azimuth=CAMERA_AZIMUTH,
        elevation=CAMERA_ELEVATION,
    ) as recorder:
        while data.time < duration:
            H_rel, target_twist = trajectory.at(data.time)
            target_pose = H_start_w @ H_rel
            tracker.step(target_pose, target_twist)

            peak_torque = np.maximum(peak_torque, np.abs(data.ctrl))
            times.append(data.time)
            errors.append(tracker.error(target_pose))

            if recorder.is_due(data.time):
                recorder.capture(data, draw)
            mujoco.mj_step(model, data)

    distance, angle = np.array(errors).T
    save_error_plot(
        directory / "error.png", times, distance, angle, "Figure-of-eight target"
    )

    print(f"Video saved to      {display_path(directory / 'video.mp4')}")
    print(f"Error plot saved to {display_path(directory / 'error.png')}")
    print(f"Tracking error, mean / peak:")
    print(
        f"  position:    {1e3 * np.sqrt(np.mean(distance**2)):.1f} / "
        f"{1e3 * distance.max():.1f} mm"
    )
    print(
        f"  orientation: {np.degrees(np.sqrt(np.mean(angle**2))):.2f} / "
        f"{np.degrees(angle.max()):.2f} deg"
    )
    print(f"Peak torque:   {np.array2string(peak_torque, precision=1)} Nm")
    print(f"Torque limits: {np.array2string(model.actuator_ctrlrange[:, 1], precision=1)} Nm")


if __name__ == "__main__":
    main()
