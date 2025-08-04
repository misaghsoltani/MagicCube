"""A Simple Interactive Cube.

This script plots a multi-color cube in three dimensions with perspective,
and allows the cube to be manipulated using either the mouse or the arrow
keys.

The rotations are based on quaternions: unfortunately there is no quaternion
algebra built-in to numpy or scipy, so we create a basic quaternion class to
accomplish this.

The cube is rendered using the zorder argument of any matplotlib object.  By
judiciously setting the zorder depending on the orientation, we can make the
cube appear to be solid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from matplotlib.axes import Axes
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.backend_bases import KeyEvent, MouseEvent


class Quaternion:
    """Quaternion Rotation.

    Class to aid in representing 3D rotations via quaternions.
    """

    @classmethod
    def from_v_theta(
        cls,
        v: NDArray[np.float32] | tuple[float, float, float] | tuple[int, int, int],
        theta: NDArray[np.float32] | float,
    ) -> Quaternion:
        """Construct quaternions from unit vectors v and rotation angles theta.

        Parameters
        ----------
        v : array_like
            array of vectors, last dimension 3. Vectors will be normalized.
        theta : array_like
            array of rotation angles in radians, shape = v.shape[:-1].

        Returns
        -------
        q : quaternion object
            quaternion representing the rotations
        """
        theta = np.asarray(theta, dtype=np.float32)
        v_array = np.asarray(v, dtype=np.float32)
        s = np.sin(0.5 * theta)
        c = np.cos(0.5 * theta)

        v_array = v_array * s / np.sqrt(np.sum(v_array * v_array, -1))
        x_shape = v_array.shape[:-1] + (4,)

        x = np.ones(x_shape, dtype=np.float32).reshape(-1, 4)
        x[:, 0] = c.ravel()
        x[:, 1:] = v_array.reshape(-1, 3)
        x = x.reshape(x_shape)

        return cls(x)

    def __init__(self, x: NDArray[np.float32] | list[float] | tuple[float, ...]) -> None:
        self.x = np.asarray(x, dtype=np.float32)

    def __repr__(self) -> str:
        """Return string representation of the quaternion."""
        return "Quaternion:\n" + self.x.__repr__()

    def __mul__(self, other: Quaternion) -> Quaternion:
        """Multiply two quaternions."""
        # multiplication of two quaternions.
        # we don't implement multiplication by a scalar
        sxr = self.x.reshape(self.x.shape[:-1] + (4, 1))
        oxr = other.x.reshape(other.x.shape[:-1] + (1, 4))

        prod = sxr * oxr
        return_shape = prod.shape[:-1]
        prod = prod.reshape((-1, 4, 4)).transpose((1, 2, 0))

        ret = np.array(
            [
                (prod[0, 0] - prod[1, 1] - prod[2, 2] - prod[3, 3]),
                (prod[0, 1] + prod[1, 0] + prod[2, 3] - prod[3, 2]),
                (prod[0, 2] - prod[1, 3] + prod[2, 0] + prod[3, 1]),
                (prod[0, 3] + prod[1, 2] - prod[2, 1] + prod[3, 0]),
            ],
            dtype=np.float32,
            order="F",
        ).T
        return self.__class__(ret.reshape(return_shape))

    def as_v_theta(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return the v, theta equivalent of the (normalized) quaternion."""
        x = self.x.reshape((-1, 4)).T

        # compute theta
        norm = np.sqrt((x**2).sum(0))
        theta = 2 * np.arccos(x[0] / norm)

        # compute the unit vector
        v = np.array(x[1:], order="F", copy=True, dtype=np.float32)
        v /= np.sqrt(np.sum(v**2, 0))

        # reshape the results
        v = v.T.reshape(self.x.shape[:-1] + (3,))
        theta = theta.reshape(self.x.shape[:-1])

        return v, theta

    def as_rotation_matrix(self) -> NDArray[np.float32]:
        """Return the rotation matrix of the (normalized) quaternion."""
        v, theta = self.as_v_theta()

        shape = theta.shape
        theta = theta.reshape(-1)
        v = v.reshape(-1, 3).T
        c = np.cos(theta)
        s = np.sin(theta)

        mat = np.array(
            [
                [
                    v[0] * v[0] * (1.0 - c) + c,
                    v[0] * v[1] * (1.0 - c) - v[2] * s,
                    v[0] * v[2] * (1.0 - c) + v[1] * s,
                ],
                [
                    v[1] * v[0] * (1.0 - c) + v[2] * s,
                    v[1] * v[1] * (1.0 - c) + c,
                    v[1] * v[2] * (1.0 - c) - v[0] * s,
                ],
                [
                    v[2] * v[0] * (1.0 - c) - v[1] * s,
                    v[2] * v[1] * (1.0 - c) + v[0] * s,
                    v[2] * v[2] * (1.0 - c) + c,
                ],
            ],
            order="F",
            dtype=np.float32,
        )
        reshaped_mat: NDArray[np.float32] = mat.T.reshape(shape + (3, 3)).astype(np.float32)
        return reshaped_mat


class CubeAxes(Axes):
    """Axes to show 3D cube.

    The cube orientation is represented by a quaternion.
    The cube has side-length 2, and the observer is a distance zloc away
    along the z-axis.
    """

    face: NDArray[np.float32] = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]], dtype=np.float32)

    # Define faces using the direct array to avoid scoping issues
    faces: NDArray[np.float32] = np.array(
        [
            np.hstack([
                np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]], dtype=np.float32)[:, :i],
                np.ones((5, 1), dtype=np.float32),
                np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]], dtype=np.float32)[:, i:],
            ])
            for i in range(3)
        ]
        + [
            np.hstack([
                np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]], dtype=np.float32)[:, :i],
                -np.ones((5, 1), dtype=np.float32),
                np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]], dtype=np.float32)[:, i:],
            ])
            for i in range(3)
        ],
        dtype=np.float32,
    )
    stickercolors: list[str] = ["#ffffff", "#00008f", "#ff6f00", "#ffcf00", "#009f0f", "#cf0000"]

    def __init__(self, fig: Any, rect: Any, *args: Any, **kwargs: Any) -> None:
        self.start_rot = Quaternion.from_v_theta(np.array([1, -1, 0], dtype=np.float32), -np.pi / 6)
        self.current_rot = self.start_rot

        self.start_zloc = 10.0
        self.current_zloc = 10.0

        # Define movement for up/down arrows or up/down mouse movement
        self._ax_UD: NDArray[np.float32] = np.array([1, 0, 0], dtype=np.float32)
        self._step_UD = 0.01

        # Define movement for left/right arrows or left/right mouse movement
        self._ax_LR: NDArray[np.float32] = np.array([0, -1, 0], dtype=np.float32)
        self._step_LR = 0.01

        # Internal variables.  These store states and data
        self._active = False
        self._xy: tuple[float, float] | None = None
        self._cube_poly: list[Polygon] | None = None
        self._shift_on = False

        # initialize the axes.  We'll set some keywords by default
        kwargs.update({
            "aspect": "equal",
            "xlim": (-1.5, 1.5),
            "ylim": (-1.5, 1.5),
            "frameon": False,
            "xticks": [],
            "yticks": [],
        })
        super().__init__(*args, **kwargs)

        # connect some GUI events
        self.figure.canvas.mpl_connect("button_press_event", self._mouse_press)  # type: ignore[arg-type]
        self.figure.canvas.mpl_connect("button_release_event", self._mouse_release)  # type: ignore[arg-type]
        self.figure.canvas.mpl_connect("motion_notify_event", self._mouse_motion)  # type: ignore[arg-type]
        self.figure.canvas.mpl_connect("key_press_event", self._key_press)  # type: ignore[arg-type]
        self.figure.canvas.mpl_connect("key_release_event", self._key_release)  # type: ignore[arg-type]

        self.draw_cube()

        self.figure.text(
            0.05,
            0.05,
            ("Drag Mouse or use arrow keys to change perspective.\nhold shift to rotate around z-axis"),
            ha="left",
            va="bottom",
        )

    @staticmethod
    def project_points(pts: NDArray[np.float32], rot: Quaternion, zloc: float) -> NDArray[np.float32]:
        """Project points to 2D given a rotation and a view.

        pts is an ndarray, last dimension 3
        rot is a Quaternion object, containing a single quaternion
        zloc is a distance along the z-axis from which the cube is being viewed
        """
        rotation_matrix = rot.as_rotation_matrix()
        rotated_pts = np.dot(pts, rotation_matrix.T)

        xdir = np.array([1.0, 0, 0], dtype=np.float32)
        ydir = np.array([0, 1.0, 0], dtype=np.float32)
        zdir = np.array([0, 0, 1.0], dtype=np.float32)

        view = zloc * zdir
        v2 = zloc**2

        result = []
        for p in rotated_pts.reshape((-1, 3)):
            dpoint = p - view
            dproj = 0.5 * dpoint * v2 / np.dot(dpoint, -1.0 * view)
            result += [
                np.array(
                    [
                        np.dot(xdir, dproj),
                        np.dot(ydir, dproj),
                        np.dot(zdir, dpoint / np.sqrt(v2)),
                    ],
                    dtype=np.float32,
                )
            ]
        return np.asarray(result, dtype=np.float32).reshape(pts.shape)

    def draw_cube(self, rot: Quaternion | None = None, zloc: float | None = None) -> None:
        """Draw a cube on the axes.

        The first time this is called, it will create a set of polygons
        representing the cube faces.  On initial calls, it will update
        these polygon faces with a given rotation and observer location.

        Parameters
        ----------
        rot : Quaternion object
            The quaternion representing the rotation
        zloc : float
            The location of the observer on the z-axis (adjusts perspective)
        """
        if rot is None:
            rot = self.current_rot
        if zloc is None:
            zloc = self.current_zloc

        self.current_rot = rot
        self.current_zloc = zloc

        if self._cube_poly is None:
            self._cube_poly = [
                Polygon(self.faces[i, :, :2], facecolor=self.stickercolors[i], alpha=0.9) for i in range(6)
            ]
            for i in range(6):
                self.add_patch(self._cube_poly[i])

        faces = self.project_points(self.faces, rot, zloc)
        zorder = np.argsort(np.argsort(faces[:, :4, 2].sum(1)))

        for i in range(6):
            self._cube_poly[i].set_zorder(10 * zorder[i])
            self._cube_poly[i].set_xy(faces[i, :, :2])

        self.figure.canvas.draw()

    def _key_press(self, event: KeyEvent) -> None:
        """Handle key press events."""
        if event.key == "shift":
            self._ax_LR = np.array([0, 0, 1], dtype=np.float32)
            self._shift_on = True

        elif event.key == "right":
            self.current_rot *= Quaternion.from_v_theta(self._ax_LR, self._step_LR)
        elif event.key == "left":
            self.current_rot *= Quaternion.from_v_theta(self._ax_LR, -self._step_LR)
        elif event.key == "up":
            self.current_rot *= Quaternion.from_v_theta(self._ax_UD, self._step_UD)
        elif event.key == "down":
            self.current_rot *= Quaternion.from_v_theta(self._ax_UD, -self._step_UD)
        self.draw_cube()

    def _key_release(self, event: KeyEvent) -> None:
        """Handle key release event."""
        if event.key == "shift":
            self._ax_LR = np.array([0, -1, 0], dtype=np.float32)

    def _mouse_press(self, event: MouseEvent) -> None:
        """Handle mouse button press."""
        if event.button == 1 and event.x is not None and event.y is not None:
            self._active = True
            self._xy = (float(event.x), float(event.y))

    def _mouse_release(self, event: MouseEvent) -> None:
        """Handle mouse button release."""
        if event.button == 1:
            self._active = False
            self._xy = None

    def _mouse_motion(self, event: MouseEvent) -> None:
        """Handle mouse motion."""
        if self._active and self._xy is not None and event.x is not None and event.y is not None:
            dx = float(event.x) - self._xy[0]
            dy = float(event.y) - self._xy[1]
            self._xy = (float(event.x), float(event.y))
            rot1 = Quaternion.from_v_theta(self._ax_UD, self._step_UD * dy)
            rot2 = Quaternion.from_v_theta(self._ax_LR, self._step_LR * dx)

            self.current_rot = self.current_rot * rot1 * rot2
            self.draw_cube()


if __name__ == "__main__":
    fig = plt.figure()
    ax = CubeAxes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    plt.show()
