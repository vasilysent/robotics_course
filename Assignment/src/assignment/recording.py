"""Recording a run of the arm to a video file.

A run is written to its own directory under `runs/`, named after the moment it
was started, and holding the video and a plot of the tracking error together
with a verbatim copy of the config that produced it.

The copy is what makes the recording worth keeping. Every number the controller
was given lives in `config.py`, so the copy is a complete record of the run: two
of them can be compared with `diff`, and one can be put back over
`src/assignment/student/config.py` to reproduce the video exactly.

Unlike the viewer, rendering to a file needs no window and no main thread, so
scripts that only record are run with `python` on every platform.
"""

import shutil
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import mujoco
import numpy as np

from assignment.student import config

# These scripts render without a window, and so must matplotlib: the Agg
# backend writes to a file and never tries to open one. It has to be chosen
# before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Where the run directories are created, next to the scripts and the sources.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIRECTORY = PROJECT_ROOT / "runs"


def display_path(path: Path) -> Path:
    """A path as it is worth printing: relative to the project root if it is
    inside it, and unchanged otherwise."""
    try:
        return Path(path).relative_to(PROJECT_ROOT)
    except ValueError:
        return Path(path)

# Size and frame rate of the recorded video. 30 frames per second is plenty for
# an arm moving at a metre per second, and keeps the file small.
#
# The size is limited only by what the graphics driver will give as an
# offscreen buffer.
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30

# Quality passed to the encoder, from 0 to 10.
VIDEO_QUALITY = 8


def new_run_directory(name: str) -> Path:
    """Create a directory for one run and save the config used into it.

    Args:
        name: what the run is about, appended to the timestamp.

    Returns:
        The path of the created directory, which is empty apart from the copy
        of the config.
    """
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    directory = RUNS_DIRECTORY / f"{stamp}_{name}"
    directory.mkdir(parents=True)
    shutil.copy(config.__file__, directory / "config.py")
    return directory


def save_error_plot(path: Path, times, distances, angles, title: str = "") -> None:
    """Plot the tracking error of a run against time and write it to a file.

    Two panels sharing the time axis: how far the end-effector is from the
    target, and through what angle it is turned away from it. Together they are
    the two halves of the pose error the spring acts on.

    Args:
        path: the image file to write, by convention `error.png` of the run.
        times: the instants the error was sampled at, in seconds.
        distances: the distance to the target at each instant, in metres.
        angles: the angle to the target at each instant, in radians.
        title: heading for the figure, empty for none.
    """
    times = np.asarray(times, dtype=float)

    figure, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(8.0, 5.0))
    if title:
        figure.suptitle(title)

    top.plot(times, 1e3 * np.asarray(distances, dtype=float), linewidth=1.2)
    top.set_ylabel("position error [mm]")

    bottom.plot(
        times, np.degrees(np.asarray(angles, dtype=float)), linewidth=1.2, color="C1"
    )
    bottom.set_ylabel("orientation error [deg]")
    bottom.set_xlabel("time [s]")

    for panel in (top, bottom):
        panel.grid(True, alpha=0.3)
        panel.set_xlim(times[0], times[-1])
        panel.set_ylim(bottom=0.0)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


class VideoRecorder:
    """An offscreen camera writing the frames of a run to an mp4 file.

    Used as a context manager, so that the file is closed and playable even if
    the run is interrupted:

        with VideoRecorder(model, path) as recorder:
            while data.time < duration:
                mujoco.mj_step(model, data)
                if recorder.is_due(data.time):
                    recorder.capture(data, frames)

    Attributes:
        path: the file being written.
        frame_count: how many frames have been written so far.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        path: Path,
        lookat=(0.0, 0.0, 0.4),
        distance: float = 2.2,
        azimuth: float = 135.0,
        elevation: float = -20.0,
        width: int = VIDEO_WIDTH,
        height: int = VIDEO_HEIGHT,
        fps: int = VIDEO_FPS,
    ):
        """
        Args:
            model: the compiled model to render. Its offscreen framebuffer is
                enlarged to the size asked for, which the model file otherwise
                leaves at a small default.
            path: the mp4 file to write.
            lookat: the point the camera is aimed at, in the world frame.
            distance: how far the camera sits from that point, in metres.
            azimuth: angle around the vertical axis, in degrees.
            elevation: angle above the horizontal, in degrees, negative to look
                down at the arm.
            width, height: size of the video in pixels.
            fps: frames per second, both of the file and of the sampling.
        """
        # The offscreen buffer is sized by the model, not by the renderer, and
        # the UR5e scene leaves it at 640x480. Asking for more than it holds is
        # an error, so it is raised here before the renderer is created.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)

        self.path = Path(path)
        self.fps = fps
        self.frame_count = 0

        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._writer = imageio.get_writer(
            str(self.path), fps=fps, quality=VIDEO_QUALITY, macro_block_size=None
        )

        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self._camera)
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.lookat = np.asarray(lookat, dtype=float)
        self._camera.distance = distance
        self._camera.azimuth = azimuth
        self._camera.elevation = elevation

    @property
    def scene(self) -> mujoco.MjvScene:
        """The scene rendered into, for drawing frames with `draw_frame`.

        It is rebuilt from the model at every `capture`, so anything drawn into
        it has to be drawn again for each frame, and never has to be cleared.
        """
        return self._renderer.scene

    def is_due(self, simulation_time: float) -> bool:
        """Whether the next video frame falls at or before this instant.

        The simulation runs at its own timestep, much finer than the frame
        rate, so only some of the steps are rendered.
        """
        return simulation_time * self.fps >= self.frame_count

    def capture(self, data: mujoco.MjData, draw=None) -> None:
        """Render the current state and append it to the video.

        Args:
            data: the state to render.
            draw: called with the scene once it has been built from the model
                and before it is rendered, to add anything that is not part of
                the model, such as the frames of `assignment.visualisation`.
        """
        self._renderer.update_scene(data, camera=self._camera)
        if draw is not None:
            draw(self.scene)
        self._writer.append_data(self._renderer.render())
        self.frame_count += 1

    def close(self) -> None:
        """Finish the file and release the renderer."""
        self._writer.close()
        self._renderer.close()

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(self, *exception) -> None:
        self.close()
