"""Draw the link-fixed frames of the kinematic description in their reference poses.

The frames are read from the YAML robot description and drawn at the poses they
have in the reference configuration q = 0. They stay frozen there while the arm
itself is free to move.

Run from the project root with:
    python scripts/02_check_frames_reference.py       (Linux / Windows)
    mjpython scripts/02_check_frames_reference.py     (macOS)
"""

import mujoco
import mujoco.viewer
import numpy as np

from assignment.kinematics_io import UR5E_CHAIN, ChainModel, load_chain_model
from assignment.mujoco_scene import run_viewer
from assignment.visualisation import build_model_with_frames, draw_frame, show_site_frames


def reference_frame_poses(chain: ChainModel) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """World poses of every frame of the chain in the reference configuration.

    At q = 0 every joint exponential is the identity, so the pose of the link
    frame Psi_i in the world frame is simply the product

        H_{Psi_i}^w = H_0^w . H_1^0(0) . ... . H_i^{i-1}(0),

    and the end-effector frame follows from H_ee^n.

    Returns:
        The poses of the link frames keyed by name, and the pose of the
        end-effector frame, which is drawn separately from the others.
    """
    link_poses = {"Psi_0": chain.H_0_w}
    H = chain.H_0_w
    for index in range(chain.n):
        H = H @ chain.H_ref[index]
        link_poses[f"Psi_{index + 1}"] = H
    return link_poses, H @ chain.H_ee_n


def main() -> None:
    chain = load_chain_model(UR5E_CHAIN)
    link_poses, H_ee_w = reference_frame_poses(chain)

    model = build_model_with_frames(link_poses)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    def show_frames(viewer: mujoco.viewer.Handle) -> None:
        show_site_frames(viewer)
        draw_frame(viewer.user_scn, H_ee_w, label="Psi_ee")

    run_viewer(model, data, configure=show_frames)


if __name__ == "__main__":
    main()
