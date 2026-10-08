"""Check the forward kinematics by drawing the computed frames on the moving arm.

At every step the joint positions are read out of the simulation, the
kinematics of the chain is evaluated at that configuration, and the frames are
moved to the poses that were computed.

Run from the project root with:
    python scripts/03_check_forward_kinematics.py       (Linux / Windows)
    mjpython scripts/03_check_forward_kinematics.py     (macOS)
"""

import mujoco
import mujoco.viewer

from assignment.kinematics_io import UR5E_CHAIN, load_chain_model
from assignment.mujoco_scene import run_viewer
from assignment.student.kinematics import ChainData, update_kinematics
from assignment.visualisation import (
    build_model_with_frames,
    clear_frames,
    draw_frame,
    move_frames,
    show_site_frames,
    site_ids,
)


def main() -> None:
    chain = load_chain_model(UR5E_CHAIN)
    kinematics = ChainData(chain)

    names = [f"Psi_{i}" for i in range(chain.n + 1)]
    update_kinematics(chain, kinematics, [0.0] * chain.n)
    model = build_model_with_frames(dict(zip(names, list(kinematics.H_link_w))))
    frames = site_ids(model, names)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    print(
        f"Drawing {len(names) + 1} computed frames of '{chain.name}': "
        f"{', '.join(names)}, Psi_ee"
    )

    def follow_the_arm(viewer: mujoco.viewer.Handle) -> None:
        """Put every frame where the kinematic description says its link is."""
        update_kinematics(chain, kinematics, data.qpos)
        move_frames(model, frames, kinematics.H_link_w)
        clear_frames(viewer.user_scn)
        draw_frame(viewer.user_scn, kinematics.H_ee_w, label="Psi_ee")

    run_viewer(model, data, configure=show_site_frames, on_step=follow_the_arm)


if __name__ == "__main__":
    main()
