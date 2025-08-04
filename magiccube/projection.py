from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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

        Returns:
        -------
        q : quaternion object
            quaternion representing the rotations
        """
        theta_array: NDArray[np.float32] = np.asarray(theta, dtype=np.float32)
        v_array: NDArray[np.float32] = np.asarray(v, dtype=np.float32)
        s = np.sin(0.5 * theta_array)
        c = np.cos(0.5 * theta_array)

        v_normalized = v_array * s / np.sqrt(np.sum(v_array * v_array, -1))
        x_shape = v_normalized.shape[:-1] + (4,)

        x = np.ones(x_shape, dtype=np.float32).reshape(-1, 4)
        x[:, 0] = c.ravel()
        x[:, 1:] = v_normalized.reshape(-1, 3)
        x = x.reshape(x_shape)

        return cls(x)

    def __init__(self, x: NDArray[np.float32] | list[float] | tuple[float, ...]) -> None:
        self.x = np.asarray(x, dtype=np.float32)

    def __repr__(self) -> str:
        """Return a string representation of the quaternion."""
        return "Quaternion:\n" + self.x.__repr__()

    def __mul__(self, other: Quaternion) -> Quaternion:
        """Multiply two quaternions.

        Parameters
        ----------
        other : Quaternion
            The quaternion to multiply with.

        Returns:
        -------
        Quaternion
            The product of the two quaternions.
        """
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
        ).T
        return mat.reshape(shape + (3, 3))

    def rotate(self, points: NDArray[np.float32]) -> NDArray[np.float32]:
        """Rotate points using the quaternion.

        Parameters
        ----------
        points : NDArray[np.float32]
            Array of points to rotate, with last dimension 3.

        Returns:
        -------
        NDArray[np.float32]
            Array of rotated points with the same shape as input.
        """
        rotation_matrix = self.as_rotation_matrix()
        return np.dot(points, rotation_matrix.T)


def project_points(
    points: NDArray[np.float32],
    q: Quaternion,
    view: NDArray[np.float32] | tuple[float, float, float],
    vertical: NDArray[np.float32] | list[int] | list[float] | None = None,
) -> NDArray[np.float32]:
    """Project points using a quaternion q and a view v.

    Parameters
    ----------
    points : array_like
        array of last-dimension 3
    q : Quaternion
        quaternion representation of the rotation
    view : array_like
        length-3 vector giving the point of view
    vertical : array_like
        direction of y-axis for view.  An error will be raised if it
        is parallel to the view.

    Returns:
    -------
    proj: array_like
        array of projected points: same shape as points.
    """
    if vertical is None:
        vertical = np.array([0, 1, 0], dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    view = np.asarray(view, dtype=np.float32)
    vertical = np.asarray(vertical, dtype=np.float32)

    xdir = np.cross(vertical, view).astype(np.float32)

    if np.all(xdir == 0):
        raise ValueError("vertical is parallel to v")

    xdir /= np.sqrt(np.dot(xdir, xdir))

    # get the unit vector corresponing to vertical
    ydir = np.cross(view, xdir)
    ydir /= np.sqrt(np.dot(ydir, ydir))

    # normalize the viewer location: this is the z-axis
    v2 = np.dot(view, view)
    zdir = view / np.sqrt(v2)

    # rotate the points
    rotation_matrix = q.as_rotation_matrix()
    rotated_points = np.dot(points, rotation_matrix.T)

    # project the points onto the view
    dpoint = rotated_points - view
    dpoint_view = np.dot(dpoint, view).reshape(dpoint.shape[:-1] + (1,))
    dproj = -dpoint * v2 / dpoint_view

    trans = list(range(1, dproj.ndim)) + [0]
    return np.array([np.dot(dproj, xdir), np.dot(dproj, ydir), -np.dot(dpoint, zdir)]).transpose(trans)
