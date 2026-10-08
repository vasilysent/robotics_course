"""Pull the end-effector towards a fixed target with a virtual spring and damper.

The arm is driven by torques rather than by position servos. At every step the
kinematics is evaluated at the current joint positions, the wrench of a virtual
spring and damper connecting the end-effector frame to the target frame is
computed, and the joint torques reproducing that wrench are commanded. That
step is `TargetTracker.step`, shared with the recording script.

The target and the gains are in `assignment.student.config`.

Run from the project root with:
    python scripts/04_track_static_target.py       (Linux / Windows)
    mjpython scripts/04_track_static_target.py     (macOS)
"""

import mujoco
import mujoco.viewer

from assignment.kinematics_io import UR5E_CHAIN, load_chain_model
from assignment.mujoco_scene import TORQUE_MODEL_PATH, load_model, run_viewer
from assignment.student import config
from assignment.student.control import TargetTracker
from assignment.visualisation import clear_frames, draw_frame


def main() -> None:
    chain = load_chain_model(UR5E_CHAIN)

    model = load_model(gravity_compensation=False, path=TORQUE_MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = config.INITIAL_CONFIGURATION
    mujoco.mj_forward(model, data)

    tracker = TargetTracker(model, data, chain)

    def control(viewer: mujoco.viewer.Handle) -> None:
        tracker.step(config.TARGET_POSE)

        clear_frames(viewer.user_scn)
        draw_frame(viewer.user_scn, tracker.kinematics.H_ee_w)
        draw_frame(viewer.user_scn, config.TARGET_POSE)

    run_viewer(model, data, on_step=control)


if __name__ == "__main__":
    main()
