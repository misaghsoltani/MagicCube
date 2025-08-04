"""This file is part of the Magic Cube project.

license
-------
Copyright 2012 David W. Hogg (NYU).

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
02110-1301 USA.

usage
-----
- initialize a solved cube with `c = Cube(N)` where `N` is the side length.
- randomize a cube with `c.randomize(32)` where `32` is the number of random moves to make.
- make cube moves with `c.move()` and turn the whole cube with `c.turn()`.
- make figures with `c.render().savefig(fn)` where `fn` is the filename.
- change sticker colors with, eg, `c.stickercolors[c.colordict["w"]] = "k"`.

conventions
-----------
- This is a model of where the stickers are, not where the solid cubies are.  That's a bug not a feature.
- Cubes are NxNxN in size.
- The faces have integers and one-letter names. The one-letter face names are given by the dictionary `Cube.facedict`.
- The layers of the cube have names that are composed of a face letter and a number,
  with 0 indicating the outermost face.
- Every layer has two layer names, for instance, (F, 1) and (B, 1) are the same layer
  of a 3x3x3 cube; (F, 1) and (B, 3) are the same layer of a 5x5x5.
- The colors have integers and one-letter names. The one-letter color names are given
  by the dictionary `Cube.colordict`.
- Convention is x before y in face arrays, plus an annoying baked-in left-handedness.
  Sue me.  Or fork, fix, pull-request.

to-do
-----
- Write translations to other move languages, so you can take a string of moves from
  some website (eg, <http://www.speedcubing.com/chris/3-permutations.html>) and execute it.
- Keep track of sticker ID numbers and orientations to show that seemingly unchanged
  parts of big cubes have had cubie swaps or stickers rotated.
- Figure out a physical "cubie" model to replace the "sticker" model.

"""  # noqa: D404

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class Cube:
    """Cube.

    Initialize with arguments:
    - `n`, the side length (the cube is `n`x`n`x`n`)
    - optional `whiteplastic=True` if you like white cubes
    """

    facedict: dict[str, int] = {"U": 0, "D": 1, "F": 2, "B": 3, "R": 4, "L": 5}
    dictface: dict[int, str] = {v: k for k, v in facedict.items()}
    normals: list[NDArray[np.float32]] = [
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, -1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    ]
    # this xdirs has to be synchronized with the self.move() function
    xdirs: list[NDArray[np.float32]] = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
        np.array([0, 0.0, 1.0], dtype=np.float32),
    ]
    colordict: dict[str, int] = {"w": 0, "y": 1, "b": 2, "g": 3, "o": 4, "r": 5}
    pltpos: list[tuple[float, float]] = [
        (0.0, 1.05),
        (0.0, -1.05),
        (0.0, 0.0),
        (2.10, 0.0),
        (1.05, 0.0),
        (-1.05, 0.0),
    ]
    labelcolor: str = "#7f00ff"

    def __init__(self, n: int, whiteplastic: bool = False) -> None:
        """(see above)."""
        self.N: int = n
        self.stickers: NDArray[np.int32] = np.array([np.tile(i, (self.N, self.N)) for i in range(6)], dtype=np.int32)
        self.stickercolors: list[str] = [
            "w",
            "#ffcf00",
            "#00008f",
            "#009f0f",
            "#ff6f00",
            "#cf0000",
        ]
        self.stickerthickness: float = 0.001  # sticker thickness in units of total cube size
        self.stickerwidth: float = 0.9  # sticker size relative to cubie size (must be < 1)
        self.plasticcolor: str = "#dfdfdf" if whiteplastic else "#1f1f1f"
        self.fontsize: float = 12.0 * (self.N / 5.0)

    def turn(self, f: str, d: int) -> None:
        """Turn whole cube (without making a layer move) around face `f`.

        `d` 90-degree turns in the clockwise direction.  Use `d=3` or
        `d=-1` for counter-clockwise.
        """
        for layer_num in range(self.N):
            self.move(f, layer_num, d)

    def move(self, f: str, layer: int, d: int) -> None:
        """Make a layer move of layer `layer` parallel to face `f` through `d` 90-degree turns.

        `d` 90-degree turns in the clockwise direction.  Layer `0` is
        the face itself, and higher `layer` values are for layers deeper
        into the cube.  Use `d=3` or `d=-1` for counter-clockwise
        moves, and `d=2` for a 180-degree move..
        """
        i = self.facedict[f]
        l2 = self.N - 1 - layer
        assert layer < self.N
        ds = range((d + 4) % 4)

        # Map each face to its opposite face for i2 calculation
        face_opposites = {"U": "D", "D": "U", "F": "B", "B": "F", "R": "L", "L": "R"}
        f2 = face_opposites[f]
        i2 = self.facedict[f2]

        if f == "U":
            for _ in ds:
                self._rotate([
                    (self.facedict["F"], range(self.N), l2),
                    (self.facedict["R"], range(self.N), l2),
                    (self.facedict["B"], range(self.N), l2),
                    (self.facedict["L"], range(self.N), l2),
                ])
        if f == "D":
            return self.move("U", l2, -d)
        if f == "F":
            for _ in ds:
                self._rotate([
                    (self.facedict["U"], range(self.N), layer),
                    (self.facedict["L"], l2, range(self.N)),
                    (self.facedict["D"], range(self.N)[::-1], l2),
                    (self.facedict["R"], layer, range(self.N)[::-1]),
                ])
        if f == "B":
            return self.move("F", l2, -d)
        if f == "R":
            for _ in ds:
                self._rotate([
                    (self.facedict["U"], l2, range(self.N)),
                    (self.facedict["F"], l2, range(self.N)),
                    (self.facedict["D"], l2, range(self.N)),
                    (self.facedict["B"], layer, range(self.N)[::-1]),
                ])
        if f == "L":
            return self.move("R", l2, -d)
        for _ in ds:
            if layer == 0:
                self.stickers[i] = np.rot90(self.stickers[i], 3)
            if layer == self.N - 1:
                self.stickers[i2] = np.rot90(self.stickers[i2], 1)

        print(f"moved {f} {layer} {len(ds)}")
        return None

    def _rotate(self, args: list[tuple[int, range, int] | tuple[int, int, range]]) -> None:
        """Perform internal rotation for the `move()` function."""
        a0 = args[0]
        foo = self.stickers[a0]
        a = a0
        for b in args[1:]:
            self.stickers[a] = self.stickers[b]
            a = b
        self.stickers[a] = foo

    def randomize(self, number: int) -> None:
        """Make `number` randomly chosen moves to scramble the cube."""
        for _t in range(number):
            f = self.dictface[np.random.randint(6)]
            layer_num = np.random.randint(self.N)
            d = 1 + np.random.randint(3)
            self.move(f, layer_num, d)

    @staticmethod
    def _render_points(points: list[NDArray[np.float32]], viewpoint: NDArray[np.float32]) -> list[NDArray[np.float32]]:
        """Perform internal function for the `render()` function.

        Clunky projection from 3-d to 2-d, but also return a zorder variable.
        """
        v2 = np.dot(viewpoint, viewpoint)
        zdir = viewpoint / np.sqrt(v2)
        xdir = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), zdir)
        xdir /= np.sqrt(np.dot(xdir, xdir))
        ydir = np.cross(zdir, xdir)
        result = []
        for p in points:
            dpoint = p - viewpoint
            dproj = 0.5 * dpoint * v2 / np.dot(dpoint, -1.0 * viewpoint)
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
        return result

    def render_views(self, ax: Axes) -> None:
        """Make three projected 3-dimensional views of the cube for the `render()` function.

        Because of zorder / occulting issues,
        this code is very brittle; it will not work for all viewpoints
        (the `np.dot(zdir, viewpoint)` test is not general; the corect
        test involves the "handedness" of the projected polygon).
        """
        csz = 2.0 / self.N
        x2 = 8.0
        x1 = 0.5 * x2
        for viewpoint, shift in [
            (np.array([-x1, -x1, x2], dtype=np.float32), np.array([-1.5, 3.0], dtype=np.float32)),
            (np.array([x1, x1, x2], dtype=np.float32), np.array([0.5, 3.0], dtype=np.float32)),
            (np.array([x2, x1, -x1], dtype=np.float32), np.array([2.5, 3.0], dtype=np.float32)),
        ]:
            for f, i in self.facedict.items():
                zdir = self.normals[i]
                if np.dot(zdir, viewpoint) < 0:
                    continue
                xdir = self.xdirs[i]
                ydir = np.cross(zdir, xdir)  # insanity: left-handed!
                psc = 1.0 - 2.0 * self.stickerthickness
                corners: list[NDArray[np.float32]] = [
                    (psc * zdir - psc * xdir - psc * ydir).astype(np.float32),
                    (psc * zdir + psc * xdir - psc * ydir).astype(np.float32),
                    (psc * zdir + psc * xdir + psc * ydir).astype(np.float32),
                    (psc * zdir - psc * xdir + psc * ydir).astype(np.float32),
                ]
                projects = self._render_points(corners, viewpoint)
                xys = [p[0:2] + shift for p in projects]
                zorder = np.mean([p[2] for p in projects])
                ax.add_artist(Polygon(xys, ec="none", fc=self.plasticcolor))
                for j in range(self.N):
                    for k in range(self.N):
                        corners = self._stickerpolygon(xdir, ydir, zdir, csz, j, k)
                        projects = self._render_points(corners, viewpoint)
                        xys = [p[0:2] + shift for p in projects]
                        ax.add_artist(
                            Polygon(
                                xys,
                                ec="none",
                                fc=self.stickercolors[self.stickers[i, j, k]],
                            )
                        )
                x0, y0, zorder = self._render_points(
                    [
                        1.5 * self.normals[i],
                    ],
                    viewpoint,
                )[0]
                ax.text(
                    x0 + shift[0],
                    y0 + shift[1],
                    f,
                    color=self.labelcolor,
                    ha="center",
                    va="center",
                    rotation=20,
                    fontsize=self.fontsize / (-zorder),
                )

    def _stickerpolygon(
        self,
        xdir: NDArray[np.float32],
        ydir: NDArray[np.float32],
        zdir: NDArray[np.float32],
        csz: float,
        j: int,
        k: int,
    ) -> list[NDArray[np.float32]]:
        small = 0.5 * (1.0 - self.stickerwidth)
        large = 1.0 - small
        return [
            (zdir - xdir + (j + small) * csz * xdir - ydir + (k + small + small) * csz * ydir).astype(np.float32),
            (zdir - xdir + (j + small + small) * csz * xdir - ydir + (k + small) * csz * ydir).astype(np.float32),
            (zdir - xdir + (j + large - small) * csz * xdir - ydir + (k + small) * csz * ydir).astype(np.float32),
            (zdir - xdir + (j + large) * csz * xdir - ydir + (k + small + small) * csz * ydir).astype(np.float32),
            (zdir - xdir + (j + large) * csz * xdir - ydir + (k + large - small) * csz * ydir).astype(np.float32),
            (zdir - xdir + (j + large - small) * csz * xdir - ydir + (k + large) * csz * ydir).astype(np.float32),
            (zdir - xdir + (j + small + small) * csz * xdir - ydir + (k + large) * csz * ydir).astype(np.float32),
            (zdir - xdir + (j + small) * csz * xdir - ydir + (k + large - small) * csz * ydir).astype(np.float32),
        ]

    def render_flat(self, ax: Axes) -> None:
        """Make an unwrapped, flat view of the cube for the `render()` function.

        This is a map, not a view really.  It does not
        properly render the plastic and stickers.
        """
        for f, i in self.facedict.items():
            x0, y0 = self.pltpos[i]
            cs = 1.0 / self.N
            for j in range(self.N):
                for k in range(self.N):
                    ax.add_artist(
                        Rectangle(
                            (x0 + j * cs, y0 + k * cs),
                            cs,
                            cs,
                            ec=self.plasticcolor,
                            fc=self.stickercolors[self.stickers[i, j, k]],
                        )
                    )
            ax.text(
                x0 + 0.5,
                y0 + 0.5,
                f,
                color=self.labelcolor,
                ha="center",
                va="center",
                rotation=20,
                fontsize=self.fontsize,
            )

    def render(self, flat: bool = True, views: bool = True) -> Figure:
        """Visualize the cube in a standard layout.

        Including a flat, unwrapped view and three perspective views.
        """
        assert flat or views
        xlim = (-2.4, 3.4)
        ylim = (-1.2, 4.0)
        if not flat:
            ylim = (2.0, 4.0)
        if not views:
            xlim = (-1.2, 3.2)
            ylim = (-1.2, 2.2)
        fig = plt.figure(
            figsize=(
                (xlim[1] - xlim[0]) * self.N / 5.0,
                (ylim[1] - ylim[0]) * self.N / 5.0,
            )
        )
        ax = fig.add_axes((0, 0, 1, 1), frameon=False, xticks=[], yticks=[])
        if views:
            self.render_views(ax)
        if flat:
            self.render_flat(ax)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        return fig


def adjacent_edge_flip(cube: Cube) -> None:
    """Do a standard edge-flipping algorithm.  Used for testing."""
    ls = range(cube.N)[1:-1]
    cube.move("R", 0, -1)
    for layer_num in ls:
        cube.move("U", layer_num, 1)
    cube.move("R", 0, 2)
    for layer_num in ls:
        cube.move("U", layer_num, 2)
    cube.move("R", 0, -1)
    cube.move("U", 0, -1)
    cube.move("R", 0, 1)
    for layer_num in ls:
        cube.move("U", layer_num, 2)
    cube.move("R", 0, 2)
    for layer_num in ls:
        cube.move("U", layer_num, -1)
    cube.move("R", 0, 1)
    cube.move("U", 0, 1)


def swap_off_diagonal(cube: Cube, f: str, l1: int, l2: int) -> None:
    """Perform a big-cube move that swaps three cubies (I think) but looks like two."""
    cube.move(f, l1, 1)
    cube.move(f, l2, 1)
    cube.move("U", 0, -1)
    cube.move(f, l2, -1)
    cube.move("U", 0, 1)
    cube.move(f, l1, -1)
    cube.move("U", 0, -1)
    cube.move(f, l2, 1)
    cube.move("U", 0, 1)
    cube.move(f, l2, -1)


def checkerboard(cube: Cube) -> None:
    """Dumbness."""
    ls = range(cube.N)[::2]
    for f in ["U", "F", "R"]:
        for layer_num in ls:
            cube.move(f, layer_num, 2)
    if cube.N % 2 == 0:
        for layer_num in ls:
            cube.move("F", layer_num, 2)


if __name__ == "__main__":
    """
    Functional testing.
    """
    np.random.seed(42)
    c = Cube(6, whiteplastic=False)
    #    c.turn("U", 1)
    #    c.move("U", 0, -1)
    #    swap_off_diagonal(c, "R", 2, 1)
    #    c.move("U", 0, 1)
    #    swap_off_diagonal(c, "R", 3, 2)
    #    checkerboard(c)
    for m in range(32):
        fig = c.render(flat=False)
        fig.savefig(f"test{m:02d}.png", dpi=865 / c.N)
        plt.close(fig)
        c.randomize(1)
