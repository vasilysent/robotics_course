"""Drawing coordinate frames in the MuJoCo viewer.

A frame is drawn as a named site: the viewer renders a coordinate triad at
every site and labels it with the site's name. The sites are attached to the
world body, so their local pose in the model is also their pose in the world
and can be rewritten at any time to move a frame around.

All site triads share one size, set by `model.vis.scale`, so a frame that
should look different from the rest cannot be a site. `draw_frame` draws such a
frame as arrows of its own instead.
"""

import mujoco
import mujoco.viewer
import numpy as np

from pathlib import Path

from assignment.mujoco_scene import MODEL_PATH

# Length and thickness of the drawn axis triads, as a fraction of the size of
# the scene. The default framelength of 1.0 spans the whole robot.
FRAME_LENGTH = 0.25
FRAME_WIDTH = 0.02

# Radius of the small sphere marking the origin of each frame.
ORIGIN_RADIUS = 0.008

# The frames sit on the joint axes, which is inside the solid links. The arm is
# therefore drawn translucent, otherwise the triads are hidden by the meshes.
ROBOT_ALPHA = 0.4

# The end-effector frame is drawn longer and thinner than the link frames, to
# set it apart. It cannot be a site like the others: the triads of all sites
# share the one `model.vis.scale` setting, so a frame of a different size has
# to be drawn as geoms of its own. These are lengths in metres as drawn, which
# at FRAME_LENGTH = 0.25 makes the axes about two and a half times as long as
# the roughly 0.05 m of a site triad.
EE_FRAME_LENGTH = 0.13
EE_FRAME_WIDTH = 0.0035

# An arrow geom is drawn half as long as the size it is given, so the size has
# to be twice the length wanted.
_ARROW_SIZE_PER_LENGTH = 2.0

# Rotations taking the z-axis onto the x-, y- and z-axis, since an arrow geom
# points along the z-axis of its own frame.
_Z_ONTO_AXIS = (
    np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
    np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    np.eye(3),
)

# Colours of the x-, y- and z-axis, as MuJoCo draws its own frames.
_AXIS_COLOURS = (
    np.array([1.0, 0.0, 0.0, 1.0]),
    np.array([0.0, 1.0, 0.0, 1.0]),
    np.array([0.0, 0.0, 1.0, 1.0]),
)


def build_model_with_frames(
    poses: dict[str, np.ndarray],
    gravity_compensation: bool = True,
    path: Path = MODEL_PATH,
) -> mujoco.MjModel:
    """Compile the UR5e scene with one named site per frame.

    Args:
        poses: the pose of each frame in the world frame, keyed by the name the
            viewer will draw as its label. The names have to be plain ASCII.
        gravity_compensation: cancel the weight of the links, as in
            `assignment.mujoco_scene.load_model`.
        path: the scene to compile.

    Returns:
        The compiled model, with the sites in the order the poses were given.
    """
    spec = mujoco.MjSpec.from_file(str(path))

    if gravity_compensation:
        # See `assignment.mujoco_scene.load_model` for why this has to happen
        # before the model is compiled.
        for body in spec.bodies:
            if body.name != "world":
                body.gravcomp = 1.0

    quaternion = np.empty(4)
    for name, H in poses.items():
        mujoco.mju_mat2Quat(quaternion, np.asarray(H)[:3, :3].flatten())
        site = spec.worldbody.add_site()
        site.name = name
        site.pos = np.asarray(H)[:3, 3]
        site.quat = quaternion
        site.size = [ORIGIN_RADIUS, 0, 0]
        site.rgba = [1.0, 1.0, 1.0, 0.6]
        # Group 0 is visible by default, unlike the group 4 that the UR5e model
        # uses for its own attachment site.
        site.group = 0

    model = spec.compile()
    model.vis.scale.framelength = FRAME_LENGTH
    model.vis.scale.framewidth = FRAME_WIDTH
    fade_robot(model)
    return model


def fade_robot(model: mujoco.MjModel) -> None:
    """Make the arm translucent, leaving the rest of the scene untouched.

    The built-in `mjVIS_TRANSPARENT` visualisation flag only fades geoms that
    belong to a moving body, which would leave the frame of the base link
    hidden inside the base. Lowering the alpha of the geoms and materials of
    the arm itself covers the base as well.
    """
    robot_geoms = model.geom_bodyid > 0  # everything except the ground plane
    model.geom_rgba[robot_geoms, 3] = ROBOT_ALPHA
    for material in set(model.geom_matid[robot_geoms]):
        if material >= 0:
            model.mat_rgba[material, 3] = ROBOT_ALPHA


def show_site_frames(viewer: mujoco.viewer.Handle) -> None:
    """Draw a coordinate triad at every site and label it with the site name."""
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE


def site_ids(model: mujoco.MjModel, names) -> np.ndarray:
    """Ids of the named sites, for moving them later."""
    ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in names]
    )
    if np.any(ids < 0):
        missing = [name for name, i in zip(names, ids) if i < 0]
        raise ValueError(f"the model has no site(s) named {missing}")
    return ids


def move_frames(model: mujoco.MjModel, ids, poses) -> None:
    """Move the frames drawn by the given sites to new world poses.

    The sites belong to the world body, so writing their local pose places them
    directly in the world. The change reaches the viewer at the next simulation
    step, which recomputes the site positions from the model.

    Args:
        model: the model holding the sites.
        ids: the site ids, from `site_ids`.
        poses: one 4x4 homogeneous matrix per id, in the same order.
    """
    quaternion = np.empty(4)
    for site, H in zip(ids, poses):
        H = np.asarray(H)
        model.site_pos[site] = H[:3, 3]
        mujoco.mju_mat2Quat(quaternion, H[:3, :3].flatten())
        model.site_quat[site] = quaternion


def draw_frame(
    scene: mujoco.MjvScene,
    H,
    length: float = EE_FRAME_LENGTH,
    width: float = EE_FRAME_WIDTH,
    label: str = "",
) -> None:
    """Draw a coordinate triad as three arrows in a scene.

    Used for the end-effector frame, which is drawn separately from the link
    frames so that it can have a size of its own.

    The arrows are appended to whatever the scene already holds, so several
    frames can be drawn into it. A scene redrawn at every step has to be
    emptied first with `clear_frames`, or the arrows pile up.

    A label is drawn where its geom is, so the label of this frame is carried
    by a text-only geom at the tip of the z-axis rather than by an arrow at the
    origin. That keeps it clear of the label of a link frame sharing the same
    origin, and for the end-effector it puts the text out along the tool axis,
    away from the arm.

    Args:
        scene: the scene to draw into, the viewer's `user_scn`.
        H: the pose of the frame in the world frame, a 4x4 homogeneous matrix.
        length: length of the arrows as drawn.
        width: radius of the arrows.
        label: text to show at the tip of the z-axis, empty for none.
    """
    H = np.asarray(H, dtype=float)
    rotation, origin = H[:3, :3], H[:3, 3]
    size = np.array([width, width, _ARROW_SIZE_PER_LENGTH * length])

    for axis in range(3):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            size,
            origin,
            (rotation @ _Z_ONTO_AXIS[axis]).flatten(),
            _AXIS_COLOURS[axis],
        )
        geom.label = ""
        scene.ngeom += 1

    if label:
        # A label geom draws its text and nothing else, so the tip of the
        # z-axis keeps its arrowhead.
        text = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            text,
            mujoco.mjtGeom.mjGEOM_LABEL,
            np.zeros(3),
            origin + rotation[:, 2] * length,
            np.eye(3).flatten(),
            np.ones(4, dtype=float),
        )
        text.label = label
        scene.ngeom += 1


def clear_frames(scene: mujoco.MjvScene) -> None:
    """Remove everything previously drawn into a scene."""
    scene.ngeom = 0
