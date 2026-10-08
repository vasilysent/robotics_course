"""Loading the MuJoCo scene of the UR5e arm and running it in the viewer."""

import sys
import time
from collections.abc import Callable
from pathlib import Path

import mujoco
import mujoco.viewer

_MODELS = Path(__file__).resolve().parents[2] / "models" / "universal_robots_ur5e"

# The arm driven by position servos, whose controls are joint positions.
MODEL_PATH = _MODELS / "scene.xml"

# The same arm driven by motors, whose controls are joint torques.
TORQUE_MODEL_PATH = _MODELS / "scene_torque.xml"


def load_model(gravity_compensation: bool = True, path: Path = MODEL_PATH) -> mujoco.MjModel:
    """Compile a UR5e scene, optionally with gravity compensation enabled.

    Gravity compensation belongs in the model when the arm is driven by
    position servos, and in the control law when it is driven by torques.
    Enabling both would cancel the weight of the links twice.

    Gravity compensation cannot be switched on after compilation: the compiler
    counts the bodies with a non-zero `gravcomp` into the read-only field
    `mjModel.ngravcomp`, and the engine skips the whole gravity-compensation
    stage when that count is zero. Writing to `model.body_gravcomp` on a model
    compiled without it therefore has no effect. So the flag is set on the
    model spec, before it is compiled.
    """
    if not gravity_compensation:
        return mujoco.MjModel.from_xml_path(str(path))

    spec = mujoco.MjSpec.from_file(str(path))
    for body in spec.bodies:
        if body.name != "world":
            body.gravcomp = 1.0
    return spec.compile()


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    configure: Callable[[mujoco.viewer.Handle], None] | None = None,
    on_step: Callable[[mujoco.viewer.Handle], None] | None = None,
) -> None:
    """Show the model in the viewer and simulate it in real time until closed.

    `configure` is called once with the viewer handle before the simulation
    starts, which is where the visualisation options can be set.

    `on_step` is called once per iteration with the viewer handle, before the
    simulation is advanced, which is where a controller reads the state and
    writes `data.ctrl`, and where anything drawn into `viewer.user_scn` is
    refreshed.

    The viewer is the passive one, which hands control back to this function
    instead of running the simulation itself. It is used here rather than
    `mujoco.viewer.launch` because only the passive viewer exposes the
    visualisation options, and because every later script needs its own
    simulation loop anyway. On macOS it comes with a restriction: creating the
    window has to happen on the main thread of the process, which the `python`
    interpreter occupies with the script itself, so the `mjpython` launcher
    shipped with MuJoCo has to be used instead.
    """
    try:
        viewer = mujoco.viewer.launch_passive(model, data)
    except RuntimeError as error:
        if "mjpython" in str(error):
            sys.exit(f"On macOS, run this script with:  mjpython {sys.argv[0]}")
        raise

    with viewer:
        if configure is not None:
            configure(viewer)

        while viewer.is_running():
            step_start = time.perf_counter()
            if on_step is not None:
                on_step(viewer)
            mujoco.mj_step(model, data)
            # `sync` refreshes the viewer, and carries the control values set
            # with the sliders back into `data.ctrl`.
            viewer.sync()
            remaining = model.opt.timestep - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)
