"""Launch the MuJoCo viewer with the UR5e arm.

The end-effector frame, the model's `attachment_site`, is drawn at the flange
and follows the arm as it moves.

Run from the project root with:
    python scripts/01_view_robot.py       (Linux / Windows)
    mjpython scripts/01_view_robot.py     (macOS)
"""

import mujoco
import mujoco.viewer
import numpy as np

from assignment.mujoco_scene import load_model, run_viewer
from assignment.visualisation import clear_frames, draw_frame

GRAVITY_COMPENSATION = True
END_EFFECTOR_SITE = "attachment_site"


def main() -> None:
    model = load_model(GRAVITY_COMPENSATION)
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, END_EFFECTOR_SITE)

    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    H_ee_w = np.eye(4)

    def draw_end_effector(viewer: mujoco.viewer.Handle) -> None:
        H_ee_w[:3, :3] = data.site_xmat[site].reshape(3, 3)
        H_ee_w[:3, 3] = data.site_xpos[site]
        clear_frames(viewer.user_scn)
        draw_frame(viewer.user_scn, H_ee_w)

    run_viewer(model, data, on_step=draw_end_effector)


if __name__ == "__main__":
    main()
