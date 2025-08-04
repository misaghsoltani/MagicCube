from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from numpy.typing import NDArray

from magiccube.projection import Quaternion, project_points


class PolyView3D(Axes):
    """A 3D polygon viewing axes for interactive 3D visualization.

    This class extends matplotlib Axes to provide 3D polygon rendering
    with interactive rotation capabilities using mouse and keyboard.

    Attributes
    ----------
        view : ndarray
            The viewing position in 3D space.
        start_rot : Quaternion
            The initial rotation quaternion.
    """

    def __init__(
        self,
        view: tuple[float, float, float] = (0, 0, 10),
        fig: Figure | None = None,
        rect: tuple[float, float, float, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if rect is None:
            rect = (0.0, 0.0, 1.0, 1.0)
        if fig is None:
            fig = plt.gcf()

        self.view: NDArray[np.float32] = np.asarray(view, dtype=np.float32)
        self.start_rot: Quaternion = Quaternion.from_v_theta(np.array([1, -1, 0], dtype=np.float32), -np.pi / 6)

        # Define movement for up/down arrows or up/down mouse movement
        self._ax_UD: NDArray[np.float32] = np.array([1, 0, 0], dtype=np.float32)
        self._step_UD: float = 0.01

        # Define movement for left/right arrows or left/right mouse movement
        self._ax_LR: NDArray[np.float32] = np.array([0, -1, 0], dtype=np.float32)
        self._step_LR: float = 0.01

        # Internal state variable
        self._button1: bool = False
        self._button2: bool = False
        self._event_xy: tuple[float, float] | None = None
        self._current_rot: Quaternion = self.start_rot
        self._npts: list[int] = [1]
        self._xyzs: NDArray[np.float32] = np.array([[0, 0, 0]], dtype=np.float32)
        self._xys: NDArray[np.float32] = np.array([[0, 0]], dtype=np.float32)
        self._polys: list[Polygon] = []

        # initialize the axes.  We'll set some keywords by default
        kwargs.update({
            "aspect": "equal",
            "xlim": (-2.5, 2.5),
            "ylim": (-2.5, 2.5),
            "frameon": False,
            "xticks": [],
            "yticks": [],
        })
        super().__init__(fig, rect, **kwargs)
        self.xaxis.set_major_formatter(NullFormatter())
        self.yaxis.set_major_formatter(NullFormatter())

        # connect some GUI events
        self.figure.canvas.mpl_connect("button_press_event", self._mouse_press)
        self.figure.canvas.mpl_connect("button_release_event", self._mouse_release)
        self.figure.canvas.mpl_connect("motion_notify_event", self._mouse_motion)
        self.figure.canvas.mpl_connect("key_press_event", self._key_press)
        self.figure.canvas.mpl_connect("key_release_event", self._key_release)

    def poly_3d(self, xyz: NDArray[np.float32], **kwargs: Any) -> None:
        """Add a 3D polygon to the axes.

        Parameters
        ----------
        xyz : array_like
            an array of vertices, shape is (Npts, 3)
        **kwargs :
            additional arguments are passed to Polygon
        """
        xyz = np.asarray(xyz, dtype=np.float32)
        self._npts.append(self._npts[-1] + xyz.shape[0])
        self._xyzs = np.vstack([self._xyzs, xyz])

        self._polys.append(Polygon(xyz[:, :2], **kwargs))
        self.add_patch(self._polys[-1])
        self._update_projection()

    def poly_3d_batch(self, xyzs: list[NDArray[np.float32]], **kwargs: Any) -> None:
        """Add multiple 3D polygons to the axes.

        This is equivalent to

            for i in range(len(xyzs)):
                kwargs_i = dict([(key, kwargs[key][i]) for key in keys])
                ax.poly_3d(xyzs[i], **kwargs_i)

        But it is much more efficient (it avoids redrawing each time).

        Parameters
        ----------
        xyzs : list
            each item of xyzs is an array of shape (Npts, 3) where Npts may
            be different for each item
        **kwargs :
            additional arguments should be lists of the same length as xyzs,
            and each item will be passed to the ``Polygon`` constructor.
        """
        # Convert all arrays to float32 for consistency
        xyzs_float32 = [np.asarray(xyz, dtype=np.float32) for xyz in xyzs]
        n = len(xyzs_float32)
        kwds = [{key: kwargs[key][i] for key in kwargs} for i in range(n)]
        polys = [Polygon(xyz[:, :2], **kwd) for (xyz, kwd) in zip(xyzs_float32, kwds, strict=False)]
        npts = self._npts[-1] + np.cumsum([len(xyz) for xyz in xyzs_float32])
        self._polys += polys
        self._npts += list(npts)
        self._xyzs = np.vstack([self._xyzs] + xyzs_float32)
        self._xys = np.array(self._xyzs[:, :2], dtype=np.float32)

        [self.add_patch(p) for p in polys]
        self._update_projection()

    def rotate(self, rot: Quaternion) -> None:
        """Apply rotation to the current view.

        Parameters
        ----------
            rot : Quaternion
                The rotation quaternion to apply.
        """
        self._current_rot *= rot

    def _update_projection(self) -> None:
        proj: NDArray[np.float32] = project_points(self._xyzs, self._current_rot, self.view)
        for i in range(len(self._polys)):
            p: NDArray[np.float32] = proj[self._npts[i] : self._npts[i + 1]]
            self._polys[i].set_xy(p[:, :2])
            self._polys[i].set_zorder(-p[:-1, 2].mean())
        self.figure.canvas.draw()

    def _key_press(self, event: Any) -> None:
        """Handle key press events."""
        if event.key == "shift":
            self._ax_LR = np.array([0, 0, 1], dtype=np.float32)

        elif event.key == "right":
            self.rotate(Quaternion.from_v_theta(self._ax_LR, 5 * self._step_LR))
        elif event.key == "left":
            self.rotate(Quaternion.from_v_theta(self._ax_LR, -5 * self._step_LR))
        elif event.key == "up":
            self.rotate(Quaternion.from_v_theta(self._ax_UD, 5 * self._step_UD))
        elif event.key == "down":
            self.rotate(Quaternion.from_v_theta(self._ax_UD, -5 * self._step_UD))
        self._update_projection()

    def _key_release(self, event: Any) -> None:
        """Handle key release event."""
        if event.key == "shift":
            self._ax_LR = np.array([0, -1, 0], dtype=np.float32)

    def _mouse_press(self, event: Any) -> None:
        """Handle mouse button press."""
        self._event_xy = (event.x, event.y)
        if event.button == 1:
            self._button1 = True
        elif event.button == 3:
            self._button2 = True

    def _mouse_release(self, event: Any) -> None:
        """Handle mouse button release."""
        self._event_xy = None
        if event.button == 1:
            self._button1 = False
        elif event.button == 3:
            self._button2 = False

    def _mouse_motion(self, event: Any) -> None:
        """Handle mouse motion."""
        if (self._button1 or self._button2) and self._event_xy is not None:
            dx = event.x - self._event_xy[0]
            dy = event.y - self._event_xy[1]
            self._event_xy = (event.x, event.y)

            if self._button1:
                rot1 = Quaternion.from_v_theta(self._ax_UD, self._step_UD * dy)
                rot2 = Quaternion.from_v_theta(self._ax_LR, self._step_LR * dx)
                self.rotate(rot1 * rot2)

                self._update_projection()

            if self._button2:
                factor = 1 - 0.003 * (dx + dy)
                xlim = self.get_xlim()
                ylim = self.get_ylim()
                self.set_xlim(factor * xlim[0], factor * xlim[1])
                self.set_ylim(factor * ylim[0], factor * ylim[1])

                self.figure.canvas.draw()


def cube_axes(n: int = 1, **kwargs: Any) -> PolyView3D:
    """Create an n x n x n rubiks cube.

    kwargs are passed to the PolyView3D instance.
    """
    stickerwidth: float = 0.9
    small: float = 0.5 * (1.0 - stickerwidth)
    d1: float = 1 - small
    d2: float = 1 - 2 * small
    d3: float = 1.01
    base_sticker: NDArray[np.float32] = np.array(
        [
            [d1, d2, d3],
            [d2, d1, d3],
            [-d2, d1, d3],
            [-d1, d2, d3],
            [-d1, -d2, d3],
            [-d2, -d1, d3],
            [d2, -d1, d3],
            [d1, -d2, d3],
            [d1, d2, d3],
        ],
        dtype=np.float32,
    )

    base_face: NDArray[np.float32] = np.array(
        [[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, 1]], dtype=np.float32
    )

    x, y, _ = np.eye(3, dtype=np.float32)
    rots: list[Quaternion] = [Quaternion.from_v_theta(x, theta) for theta in (np.pi / 2, -np.pi / 2)]
    rots += [Quaternion.from_v_theta(y, theta) for theta in (np.pi / 2, -np.pi / 2, np.pi, 2 * np.pi)]

    cubie_width: float = 2.0 / n
    translations: NDArray[np.float32] = np.array(
        [[-1 + (i + 0.5) * cubie_width, -1 + (j + 0.5) * cubie_width, 0] for i in range(n) for j in range(n)],
        dtype=np.float32,
    )

    colors: list[str] = ["blue", "green", "white", "yellow", "orange", "red"]

    factor: NDArray[np.float32] = np.array([1.0 / n, 1.0 / n, 1], dtype=np.float32)

    ax: PolyView3D = PolyView3D(**kwargs)
    facecolor: list[str] = []
    polys: list[NDArray[np.float32]] = []

    for t in translations:
        base_face_trans: NDArray[np.float32] = (factor * base_face + t).astype(np.float32)
        base_sticker_trans: NDArray[np.float32] = (factor * base_sticker + t).astype(np.float32)
        for r, c in zip(rots, colors, strict=False):
            polys += [r.rotate(base_face_trans), r.rotate(base_sticker_trans)]
            facecolor += ["k", c]

    ax.poly_3d_batch(polys, facecolor=facecolor)

    ax.figure.text(
        0.05,
        0.05,
        ("Drag Mouse or use arrow keys to change perspective.\nHold shift to adjust z-axis rotation"),
        ha="left",
        va="bottom",
    )
    return ax


if __name__ == "__main__":
    fig = plt.figure(figsize=(5, 5))
    fig.add_axes(cube_axes(n=2, fig=fig))
    plt.show()
    plt.show()
