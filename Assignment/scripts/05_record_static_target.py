"""Record the run of script 04 to a video file instead of showing it live.

The control is exactly the one of `04_track_static_target.py`, the same
`TargetTracker.step` at every simulation step. Only the output differs: rather
than opening a window, the scene is rendered offscreen at a fixed frame rate
and written to an mp4 video file, together with a copy of the config that produced it.

Each run gets a directory of its own under `runs/`.

Rendering needs no window, so unlike the other scripts this one is run with
plain `python` on macOS as well:

    python scripts/05_record_static_target.py
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
from assignment.visualisation import draw_frame

# How long the arm is simulated and recorded, in seconds.
DURATION = 5.0

# Where the camera is put.
CAMERA_LOOKAT = (0.15, 0.15, 0.45)
CAMERA_DISTANCE = 1.4
CAMERA_AZIMUTH = 135.0
CAMERA_ELEVATION = -15.0


def main() -> None:
    chain = load_chain_model(UR5E_CHAIN)

    model = load_model(gravity_compensation=False, path=TORQUE_MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = config.INITIAL_CONFIGURATION
    mujoco.mj_forward(model, data)

    tracker = TargetTracker(model, data, chain)

    def draw(scene: mujoco.MjvScene) -> None:
        """Add the two frames the video is about to the rendered scene."""
        draw_frame(scene, tracker.kinematics.H_ee_w)
        draw_frame(scene, config.TARGET_POSE)

    directory = new_run_directory("static_target")
    print(f"Recording {DURATION:.1f} s to {display_path(directory)}")

    peak_torque = np.zeros(model.nu)
    times, errors = [], []

    with VideoRecorder(
        model,
        directory / "video.mp4",
        lookat=CAMERA_LOOKAT,
        distance=CAMERA_DISTANCE,
        azimuth=CAMERA_AZIMUTH,
        elevation=CAMERA_ELEVATION,
    ) as recorder:
        while data.time < DURATION:
            tracker.step(config.TARGET_POSE)

            peak_torque = np.maximum(peak_torque, np.abs(data.ctrl))
            times.append(data.time)
            errors.append(tracker.error(config.TARGET_POSE))

            if recorder.is_due(data.time):
                recorder.capture(data, draw)
            mujoco.mj_step(model, data)

    distances, angles = np.array(errors).T
    save_error_plot(
        directory / "error.png", times, distances, angles, "Static target"
    )

    print(f"Video saved to      {display_path(directory / 'video.mp4')}")
    print(f"Error plot saved to {display_path(directory / 'error.png')}")
    print(
        f"Final error:   {1e3 * distances[-1]:.2f} mm, "
        f"{np.degrees(angles[-1]):.2f} deg"
    )
    print(f"Peak torque:   {np.array2string(peak_torque, precision=1)} Nm")
    print(f"Torque limits: {np.array2string(model.actuator_ctrlrange[:, 1], precision=1)} Nm")


if __name__ == "__main__":
    main()
