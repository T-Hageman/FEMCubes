"""Make the dimension drawings and sample meshes of the mesh generator pages.

Run from the repository root, in the FEniCSx environment::

    python docs/scripts/make_mesh_generator_figures.py            # all figures
    python docs/scripts/make_mesh_generator_figures.py rectangle  # selected cases

For every case ``<name>`` it writes, into ``docs/_static/mesh_generators/``,

* ``<name>_dimensions.svg``: the outline of the domain with its parameters,
  the origin and the named boundary and domain groups;
* ``<name>_mesh.png``: the mesh built by the generator from the parameter
  class of the case, with the boundary groups coloured.

The parameter classes below are included verbatim in the documentation
pages (``literalinclude`` with ``:pyobject:``), so the snippets on the pages
are exactly the parameters of the sample meshes. Option ``--preview DIR``
also writes PNG copies of the drawings to ``DIR`` for checking.
"""
import argparse
import io
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.collections  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Polygon  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from PIL import Image  # noqa: E402

from femcubes.core.params_template import ParamsTemplate  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "_static" / "mesh_generators"


# ----------------------------------------------------------------------
# Parameters of the sample meshes (shown on the documentation pages)
# ----------------------------------------------------------------------

class RectangleParams(ParamsTemplate):
    MeshType = "Rectangle"
    dim = 2
    Lx = 4.0
    Ly = 2.0
    dx = 0.25


class Rectangle3DParams(ParamsTemplate):
    MeshType = "Rectangle"
    dim = 3
    Lx = 2.0
    Ly = 1.0
    Lz = 1.0
    dx = 0.25


class RectangleNotchesParams(ParamsTemplate):
    MeshType = "Rectangle_Notches"
    dim = 2
    Lx = 4.0
    Ly = 2.0
    dx = 0.2
    nNotches = 3
    Notch_Width = 0.25
    Notch_Depth = 0.6


class RectangleNotches3DParams(ParamsTemplate):
    MeshType = "Rectangle_Notches"
    dim = 3
    Lx = 3.0
    Ly = 1.0
    Lz = 1.0
    dx = 0.15
    Notch_Positions = [1.0, 2.0]
    Notch_Width = 0.2
    Notch_Depth = 0.4


class RectangleCentreNotchParams(ParamsTemplate):
    MeshType = "Rectangle_CentreNotch"
    dim = 2
    Lx = 4.0
    Ly = 2.0
    dx = 0.25
    Notch_Width = 0.6
    Notch_Depth = 0.3
    dx_notch = 0.05


class RectangleRefinedTopParams(ParamsTemplate):
    MeshType = "Rectangle_RefinedTop"
    dim = 2
    Lx = 10.0
    Ly = 4.0
    Ly_refined = 1.0
    dx_min = 0.15
    dx_max = 0.6
    Lx_topleft = 1.0
    AlignWithRectangle = {"x_min": 4.0, "x_max": 7.0, "y_min": 1.5, "y_max": 2.5}


class PlateWithHoleParams(ParamsTemplate):
    MeshType = "PlateWithHole"
    dim = 2
    Lx = 2.0
    Ly = 1.5
    R = 0.5
    dx = 0.1


class PlateWithHole3DParams(ParamsTemplate):
    MeshType = "PlateWithHole"
    dim = 3
    Lx = 2.0
    Ly = 1.5
    Lz = 0.5
    R = 0.5
    dx = 0.2


class FullPlateWithHoleParams(ParamsTemplate):
    MeshType = "FullPlateWithHole"
    dim = 2
    Lx = 4.0
    Ly = 3.0
    R = 0.8
    dx = 0.25
    ellipticity = 2.0
    rotation_angle = 30.0
    dxmin = 0.05
    R_Refine = 0.2


class FullPlateWithHole3DParams(ParamsTemplate):
    MeshType = "FullPlateWithHole"
    dim = 3
    Lx = 4.0
    Ly = 3.0
    Lz = 1.0
    R = 0.8
    dx = 0.3
    ellipticity = 2.0
    rotation_angle = 30.0


class HalfPlateWithHoleParams(ParamsTemplate):
    MeshType = "HalfPlateWithHole"
    dim = 2
    Lx = 4.0
    Ly = 3.0
    R = 0.6
    dx = 0.25
    dxmin = 0.05
    R_Refine = 0.2


class SingleEdgeNotchedParams(ParamsTemplate):
    MeshType = "SingleEdgeNotched"
    dim = 2
    Lx = 1.0
    Ly = 1.0
    LNotch = 0.5
    NotchHeight = 0.04
    dx = 0.04


class SingleEdgeNotched3DParams(ParamsTemplate):
    MeshType = "SingleEdgeNotched"
    dim = 3
    Lx = 1.0
    Ly = 1.0
    Lz = 0.3
    LNotch = 0.5
    NotchHeight = 0.06
    dx = 0.08


class DamageChallengeParams(ParamsTemplate):
    MeshType = "DamageChallenge"
    dim = 2
    Lx = 4.0
    Ly = 2.0
    dx = 0.15
    Lx_BC = 0.6
    Lx_notch = 1.6
    Notch_Thickness = 0.1
    Ly_notch = 0.8
    Load_Width = 0.4
    BC_Width = 0.3


class DamageChallenge3DParams(ParamsTemplate):
    MeshType = "DamageChallenge"
    dim = 3
    Lx = 4.0
    Ly = 2.0
    Lz = 1.0
    dx = 0.25
    Lx_BC = 0.6
    Lx_notch = [1.6, 2.2]
    Notch_Thickness = 0.1
    Ly_notch = 0.8
    Load_Width = 0.4
    BC_Width = 0.3


class CylinderParams(ParamsTemplate):
    MeshType = "Cylinder"
    dim = 3
    R = 0.5
    Lz = 1.5
    dx = 0.12
    notch_r = 0.15
    notch_h = 0.75
    notch_dh = 0.2


class NotchedTensileBarParams(ParamsTemplate):
    MeshType = "NotchedTensileBar"
    dim = 2
    Lx = 4.0
    Ly = 1.6
    Lx_centre = 1.2
    Notch_Width = 0.1
    Notch_Depth = 0.3
    dx = 0.2
    dx_centre = 0.05


class TwoSquaredParams(ParamsTemplate):
    MeshType = "TwoSquared"
    dim = 2
    L = 1.0
    dx = 0.2
    dx_interface = 0.05


class CorrosionPitParams(ParamsTemplate):
    MeshType = "CorrosionPit"
    dim = 2
    W = 4.0
    H = 2.0
    R_pit = 0.5
    dx = 0.3
    dx_metal = 0.08


class PlateWithDefectParams(ParamsTemplate):
    MeshType = "PlateWithDefect"
    dim = 2
    W = 4.0
    H = 1.0
    R_defect = 0.5
    dx = 0.25
    dx_metal = 0.06
    defect_depth = 0.2
    weak_spots = [(1.4, 0.25)]
    voids = [(2.7, 0.35, 0.15)]


class PlateWithDefect3DParams(ParamsTemplate):
    MeshType = "PlateWithDefect"
    dim = 3
    W = 2.0
    H = 0.5
    R_defect = 0.5
    dx = 0.2
    dx_metal = 0.08


# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------

FONT_SIZE = 8.5
MONO = "DejaVu Sans Mono"
INK = "#1f1f1e"            # outlines and parameter names
INK_SOFT = "#52514e"       # dimension lines, notes
GHOST = "#a3a29c"          # hidden edges, removed parts
DOMAIN_FILL = "#ecebe7"
SIDE_FILL = "#d9d8d3"
EDGE_LW = 2.4              # boundary groups
OUTLINE_LW = 1.0
DIM_LW = 0.6
FIG_WIDTH = 6.4            # inch, sample meshes
DRAW_WIDTH = 7.5           # inch, drawings before cropping to their content
MAX_HEIGHT = 4.2           # inch
DPI_MESH = 130

# Categorical slots (validated order); groups keep their colour over all
# pages where possible: the preferred slot is used unless it is taken.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
         "#7a5c3e", "#8a8984"]
PREFERRED = {
    "bottom": 0, "right": 1, "top": 2, "left": 3, "hole": 4, "notch": 4, "symmetry": 5,
    "interface": 6, "front": 5, "back": 6, "outer": 1, "far": 1, "metal": 7,
    "metal:defect": 4, "metal:outer": 7, "metal:pit": 4, "top_load": 7,
    "bottom_left": 6, "bottom_right": 5, "top-left": 7, "metal:spot1": 6, "metal:void1": 5,
}
DOMAIN_TINTS = ["#dce8f7", "#fbe1d6", "#d5f0e5"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FONT_SIZE,
    "svg.fonttype": "none",
    "svg.hashsalt": "femcubes",
    "path.simplify": True,
})


def group_colours(names, given=None):
    """Colour per group name, unique within one figure; ``given`` colours are kept."""
    given = {k: v for k, v in (given or {}).items() if k in names}
    colours, used = {}, {SLOTS.index(c) for c in given.values()}
    for name in names:
        if name in given:
            colours[name] = given[name]
            continue
        slot = PREFERRED.get(name, PREFERRED.get(name.split(":")[0]))
        if slot is None or slot in used:
            slot = next(i for i in range(len(SLOTS)) if i not in used)
        used.add(slot)
        colours[name] = SLOTS[slot]
    return colours


# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------

def make_figure(panels, width=DRAW_WIDTH, gap=0.35, titles=None, vertical=False):
    """Axes with one common scale; ``panels`` holds (xlim, ylim) per axes, side by side or stacked."""
    widths = [x1 - x0 for (x0, x1), _ in panels]
    heights = [y1 - y0 for _, (y0, y1) in panels]
    title_space = 0.3 if titles else 0.0
    if vertical:
        scale = max(widths) / width
        total_h = sum(heights) / scale + (gap + title_space) * len(panels)
        fig = plt.figure(figsize=(width, total_h))
        axes, top = [], total_h
        for w, h in zip(widths, heights):
            top -= title_space
            ax = fig.add_axes([0.0, (top - h / scale) / total_h, w / scale / width, h / scale / total_h])
            top -= h / scale + gap
            axes.append(ax)
    else:
        usable = width - gap * (len(panels) - 1)
        scale = sum(widths) / usable                       # data units per inch
        if max(heights) / scale > MAX_HEIGHT:
            scale = max(heights) / MAX_HEIGHT
        height = max(heights) / scale
        total = sum(widths) / scale + gap * (len(panels) - 1)
        fig = plt.figure(figsize=(total, height + title_space))
        axes, left = [], 0.0
        for w, h in zip(widths, heights):
            axes.append(fig.add_axes([left / total, 0.0, w / scale / total, h / scale / (height + title_space)]))
            left += w / scale + gap
    for i, (ax, (xlim, ylim)) in enumerate(zip(axes, panels)):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.axis("off")
        if titles:
            ax.set_title(titles[i], fontsize=FONT_SIZE, color=INK_SOFT, loc="left")
    return fig, axes, scale


class Sketch:
    """Drawing primitives in data coordinates; ``u`` is a length of about 6 pt."""

    def __init__(self, ax, scale, colours):
        self.ax = ax
        self.u = scale * 6.0 / 72.0
        self.colours = colours
        ax.figure.group_colours = colours          # reused for the sample mesh

    # -- shapes ---------------------------------------------------------
    def fill(self, pts, color=DOMAIN_FILL, z=0):
        self.ax.add_patch(Polygon(np.asarray(pts), closed=True, fc=color, ec="none", zorder=z))

    def line(self, pts, color=INK, lw=OUTLINE_LW, ls="-", z=2):
        pts = np.asarray(pts)
        self.ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls, zorder=z,
                     solid_capstyle="round", dash_capstyle="butt", dashes=(4, 2.5) if ls == "--" else (None, None))

    def edge(self, pts, group):
        pts = np.asarray(pts)
        self.ax.plot(pts[:, 0], pts[:, 1], color=self.colours[group], lw=EDGE_LW, zorder=3,
                     solid_capstyle="butt")

    # -- text -----------------------------------------------------------
    def group(self, name, at, target=None, ha="center", va="center", text=None, dashed=False):
        """Group name in a box of the group colour, with an optional leader to ``target``."""
        if target is not None:
            self.line([at, target], color=INK_SOFT, lw=DIM_LW, ls="--" if dashed else "-", z=4)
            self.ax.plot(*target, marker="o", ms=2.2, color=INK_SOFT, zorder=4)
        self.ax.text(*at, text or name, family=MONO, color=INK, ha=ha, va=va, zorder=6,
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=self.colours[name], lw=1.4))

    def domain(self, text, at, ha="center", va="center"):
        self.ax.text(*at, text, family=MONO, color=INK_SOFT, style="italic", ha=ha, va=va, zorder=5)

    def label(self, text, at, ha="center", va="center", color=INK, mono=True, box=True, size=None):
        self.ax.text(*at, text, family=MONO if mono else None, color=color, ha=ha, va=va, zorder=6,
                     fontsize=size or FONT_SIZE,
                     bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none") if box else None)

    def note(self, text, at, ha="left", va="center"):
        self.ax.text(*at, text, color=INK_SOFT, ha=ha, va=va, zorder=6, fontsize=FONT_SIZE - 0.5)

    # -- dimensions -----------------------------------------------------
    def _arrow(self, a, b, style="<|-|>"):
        self.ax.add_patch(FancyArrowPatch(a, b, arrowstyle=f"{style},head_length=4,head_width=1.6",
                                          color=INK_SOFT, lw=DIM_LW, shrinkA=0, shrinkB=0, zorder=4))

    def dim(self, a, b, off, label, outside=False, shift=0.0, ext=True):
        """Dimension between ``a`` and ``b``, drawn offset by ``off`` (in units of u, signed normal).

        ``shift`` moves the label along the dimension line (units of u).
        """
        a, b = np.asarray(a, float), np.asarray(b, float)
        t = (b - a) / np.linalg.norm(b - a)
        n = np.array([-t[1], t[0]])
        a2, b2 = a + off * self.u * n, b + off * self.u * n
        if ext and abs(off) > 0:
            s = np.sign(off)
            for p, p2 in ((a, a2), (b, b2)):
                self.line([p + s * 0.3 * self.u * n, p2 + s * 0.5 * self.u * n], color=INK_SOFT, lw=DIM_LW, z=4)
        if outside:
            self._arrow(a2 - 2.2 * self.u * t, a2, "-|>")
            self._arrow(b2 + 2.2 * self.u * t, b2, "-|>")
            self.line([a2, b2], color=INK_SOFT, lw=DIM_LW, z=4)
        else:
            self._arrow(a2, b2)
        mid = 0.5 * (a2 + b2) + shift * self.u * t
        side = n * (np.sign(off) if off != 0 else 1.0)
        self._text_beside(label, mid, side)

    def _text_beside(self, label, at, direction, gap=0.6):
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        pos = at + gap * self.u * d
        if abs(d[0]) > abs(d[1]):
            ha, va = ("left" if d[0] > 0 else "right"), "center"
        else:
            ha, va = "center", ("bottom" if d[1] > 0 else "top")
        self.label(label, pos, ha=ha, va=va)

    def radius(self, centre, end, label, direction=None):
        """Arrow from ``centre`` to ``end`` with ``label`` beside its middle."""
        c, e = np.asarray(centre, float), np.asarray(end, float)
        self._arrow(c, e, "-|>")
        self.ax.plot(*c, marker="+", ms=5, color=INK_SOFT, mew=DIM_LW, zorder=4)
        t = (e - c) / np.linalg.norm(e - c)
        self._text_beside(label, 0.5 * (c + e), direction if direction is not None else [-t[1], t[0]])

    def leader(self, text, at, target):
        """Parameter name with a thin leader to ``target``."""
        self.line([at, target], color=INK_SOFT, lw=DIM_LW, z=4)
        self.ax.plot(*target, marker="o", ms=2.2, color=INK_SOFT, zorder=4)
        dx = target[0] - at[0]
        self.label(text, at, ha="right" if dx > 0 else "left")

    # -- frame ----------------------------------------------------------
    def origin(self, at=(0.0, 0.0), text="(0, 0)", direction=(-1.0, -1.0)):
        self.ax.plot(*at, marker="o", ms=4.5, mfc="white", mec=INK, mew=1.1, zorder=7)
        d = np.asarray(direction, float)
        pos = np.asarray(at) + 0.9 * self.u * d / np.linalg.norm(d)
        self.ax.text(*pos, text, color=INK_SOFT, ha="right" if d[0] < 0 else "left",
                     va="top" if d[1] < 0 else "bottom", zorder=7, fontsize=FONT_SIZE - 0.5)

    def triad(self, at, labels=("x", "y"), dirs=((1, 0), (0, 1))):
        at = np.asarray(at, float)
        for lab, d in zip(labels, dirs):
            d = np.asarray(d, float)
            d = d / np.linalg.norm(d)
            d = d / np.linalg.norm(d)
            tip = at + 3.0 * self.u * d
            self.ax.add_patch(FancyArrowPatch(at, tip, arrowstyle="-|>,head_length=4,head_width=2",
                                              color=INK, lw=0.9, shrinkA=0, shrinkB=0, zorder=5))
            self.ax.text(*(tip + 0.9 * self.u * d), lab, ha="center", va="center", style="italic",
                         color=INK, zorder=5)


def arc(centre, rx, ry, a0, a1, rot=0.0, n=80):
    """Points of an elliptical arc (angles in degrees, ``rot`` the rotation of the ellipse)."""
    t = np.radians(np.linspace(a0, a1, n))
    x, y = rx * np.cos(t), ry * np.sin(t)
    r = np.radians(rot)
    return np.column_stack([centre[0] + x * np.cos(r) - y * np.sin(r), centre[1] + x * np.sin(r) + y * np.cos(r)])


def rect(x0, y0, x1, y1):
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


# -- oblique prisms ----------------------------------------------------

def depth_vector(depth, k=0.5, angle=35.0):
    return depth * k * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])


def _inside(pt, outer, holes):
    if not MplPath(outer).contains_point(pt, radius=0.0):
        return False
    return not any(MplPath(h).contains_point(pt, radius=0.0) for h in holes)


def prism(sk, outer, D, holes=(), back_outer=None, back_holes=None, extra=()):
    """Oblique drawing of the extrusion of a plane face (``outer`` with ``holes``) by ``D``.

    Hidden edges are dashed. ``back_outer`` / ``back_holes`` replace the
    shifted front outline for the back face (for slanted features);
    ``extra`` lists additional connector pairs (front point, back point).
    """
    outer = np.asarray(outer, float)
    holes = [np.asarray(h, float) for h in holes]
    back_outer = outer + D if back_outer is None else np.asarray(back_outer, float)
    back_holes = [h + D for h in holes] if back_holes is None else [np.asarray(h, float) for h in back_holes]

    def hidden(p, frac):
        # A point at depth fraction ``frac`` is hidden if a nearer part of the solid covers it.
        return any(_inside(p - s * D, outer, holes) for s in np.linspace(0.01, frac - 0.01, 30)) if frac > 0.02 else False

    def draw_polyline(pts, fracs, closed):
        pts = np.asarray(pts, float)
        fracs = np.asarray(fracs, float)
        if closed:
            pts, fracs = np.vstack([pts, pts[:1]]), np.append(fracs, fracs[0])
        dense, dfrac = [], []
        for a, b, fa, fb in zip(pts[:-1], pts[1:], fracs[:-1], fracs[1:]):
            m = max(2, int(np.linalg.norm(b - a) / (0.4 * sk.u)))
            for s in np.linspace(0, 1, m, endpoint=False):
                dense.append(a + (b - a) * s)
                dfrac.append(fa + (fb - fa) * s)
        dense.append(pts[-1])
        dfrac.append(fracs[-1])
        dense = np.array(dense)
        flags = np.array([hidden(p, f) for p, f in zip(dense, dfrac)])
        start = 0
        for i in range(1, len(dense) + 1):
            if i == len(dense) or flags[i] != flags[start]:
                seg = dense[start:min(i + 1, len(dense))]
                if len(seg) > 1:
                    sk.line(seg, color=GHOST if flags[start] else INK, ls="--" if flags[start] else "-",
                            lw=0.8 if flags[start] else OUTLINE_LW)
                start = i

    # side walls, then the front face over them
    for loop, back in [(outer, back_outer)] + list(zip(holes, back_holes)):
        for i in range(len(loop)):
            j = (i + 1) % len(loop)
            sk.fill([loop[i], loop[j], back[j], back[i]], color=SIDE_FILL, z=0)
    sk.fill(outer, z=1)
    for h in holes:
        sk.fill(h, color=SIDE_FILL, z=1)
    draw_polyline(back_outer, np.ones(len(back_outer)), True)
    for h in back_holes:
        draw_polyline(h, np.ones(len(h)), True)
    connectors = [(p, q) for p, q in zip(outer, back_outer) if _is_corner(outer, p)]
    for h, hb in zip(holes, back_holes):
        n = np.array([-D[1], D[0]])
        proj = h @ n
        for k in (int(np.argmax(proj)), int(np.argmin(proj))):
            connectors.append((h[k], hb[k]))
    connectors += list(extra)
    for p, q in connectors:
        draw_polyline(np.array([p, q]), [0.0, 1.0], False)
    sk.line(np.vstack([outer, outer[:1]]))
    for h in holes:
        sk.line(np.vstack([h, h[:1]]))


def _is_corner(loop, p, tol=1e-9):
    """True for vertices where the outline turns (not for sampled curve points)."""
    i = int(np.argmin(np.linalg.norm(loop - p, axis=1)))
    a, b = loop[i - 1], loop[(i + 1) % len(loop)]
    t1, t2 = p - a, b - p
    cross = t1[0] * t2[1] - t1[1] * t2[0]
    return abs(cross) > 0.05 * np.linalg.norm(t1) * np.linalg.norm(t2)


# ----------------------------------------------------------------------
# Dimension drawings
# ----------------------------------------------------------------------

def _pad(x0, x1, y0, y1, l=1.0, r=1.0, b=1.0, t=1.0):
    return (x0 - l, x1 + r), (y0 - b, y1 + t)


def draw_rectangle(p):
    Lx, Ly = p.Lx, p.Ly
    names = ["bottom", "right", "top", "left"]
    fig, (ax,), scale = make_figure([_pad(0, Lx, 0, Ly, 1.3, 0.9, 0.8, 0.4)])
    sk = Sketch(ax, scale, group_colours(names))
    u = sk.u
    sk.fill(rect(0, 0, Lx, Ly))
    for g, pts in zip(names, [[(0, 0), (Lx, 0)], [(Lx, 0), (Lx, Ly)], [(Lx, Ly), (0, Ly)], [(0, Ly), (0, 0)]]):
        sk.edge(pts, g)
    sk.group("bottom", (Lx / 2, 1.6 * u))
    sk.group("top", (Lx / 2, Ly - 1.6 * u))
    sk.group("left", (1.2 * u, Ly / 2), ha="left")
    sk.group("right", (Lx - 1.2 * u, Ly / 2), ha="right")
    sk.domain("domain", (Lx / 2, Ly / 2))
    sk.dim((0, 0), (Lx, 0), -3.0, "Lx")
    sk.dim((0, 0), (0, Ly), 3.0, "Ly")
    sk.origin()
    sk.triad((-8 * u, Ly - 5 * u))
    return fig


def draw_rectangle_3d(p):
    Lx, Ly, Lz = p.Lx, p.Ly, p.Lz
    D = depth_vector(Ly)
    names = ["left", "top", "back", "front", "right", "bottom"]
    xlim = (-1.35, Lx + D[0] + 1.6)
    ylim = (-0.5, Lz + D[1] + 0.35)
    fig, (ax,), scale = make_figure([(xlim, ylim)])
    sk = Sketch(ax, scale, group_colours(names))
    prism(sk, rect(0, 0, Lx, Lz), D)
    sk.group("left", (Lx * 0.35, Lz * 0.45), text="left (y = 0)")
    sk.group("top", (Lx * 0.45 + 0.5 * D[0], Lz + 0.5 * D[1]), text="top (z = Lz)")
    sk.group("back", (Lx + 0.5 * D[0], Lz * 0.5 + 0.5 * D[1]), text="back (x = Lx)")
    sk.group("front", (-0.3, Lz * 0.75), target=(0.5 * D[0], Lz * 0.5 + 0.5 * D[1]),
             text="front (x = 0)", ha="right", dashed=True)
    sk.group("right", (Lx + D[0] + 0.6, Lz + D[1] - 0.1), target=(Lx * 0.75 + D[0], Lz * 0.6 + D[1]),
             text="right (y = Ly)", ha="left", dashed=True)
    sk.group("bottom", (Lx + 0.9, -0.2), target=(Lx * 0.7 + 0.5 * D[0], 0.5 * D[1]),
             text="bottom (z = 0)", ha="left", dashed=True)
    sk.dim((0, 0), (Lx, 0), -2.2, "Lx")
    sk.dim((0, 0), (0, Lz), 2.2, "Lz")
    sk.dim((Lx, 0), (Lx + D[0], D[1]), -2.2, "Ly")
    sk.origin(text="(0, 0, 0)")
    sk.triad((Lx + D[0] + 0.25, 0.05), labels=("x", "y", "z"), dirs=((1, 0), D / np.linalg.norm(D), (0, 1)))
    sk.domain("domain", (Lx * 0.35, Lz * 0.2))
    return fig


def _notch_outline(Lx, Ly, intervals, depth):
    pts = [(0, 0), (Lx, 0), (Lx, Ly)]
    for x0, x1 in sorted(intervals, reverse=True):
        pts += [(x1, Ly), (x1, Ly - depth), (x0, Ly - depth), (x0, Ly)]
    pts.append((0, Ly))
    return np.array(pts, float)


def _notch_intervals(p):
    pos = getattr(p, "Notch_Positions", None)
    if pos is None:
        pos = [p.Lx * (i + 0.5) / p.nNotches for i in range(p.nNotches)]
    w = p.Notch_Width
    return [(x - w / 2, x + w / 2) for x in pos], list(pos)


def draw_rectangle_notches(p):
    Lx, Ly, d = p.Lx, p.Ly, p.Notch_Depth
    intervals, pos = _notch_intervals(p)
    names = ["bottom", "right", "top", "left", "notch"]
    fig, (ax,), scale = make_figure([_pad(0, Lx, 0, Ly, 1.3, 0.9, 0.8, 1.5)])
    sk = Sketch(ax, scale, group_colours(names))
    u = sk.u
    outline = _notch_outline(Lx, Ly, intervals, d)
    sk.fill(outline)
    sk.edge([(0, 0), (Lx, 0)], "bottom")
    sk.edge([(Lx, 0), (Lx, Ly)], "right")
    sk.edge([(0, Ly), (0, 0)], "left")
    xs = [0] + [x for iv in intervals for x in iv] + [Lx]
    for a, b in zip(xs[0::2], xs[1::2]):
        sk.edge([(a, Ly), (b, Ly)], "top")
    for x0, x1 in intervals:
        sk.edge([(x0, Ly), (x0, Ly - d), (x1, Ly - d), (x1, Ly)], "notch")
    sk.group("bottom", (Lx / 2, 1.6 * u))
    sk.group("top", (0.5 * intervals[0][0], Ly - 1.6 * u))
    sk.group("left", (1.2 * u, Ly / 2), ha="left")
    sk.group("right", (Lx - 1.2 * u, Ly / 2), ha="right")
    sk.group("notch", (pos[1] + 2.5 * u, Ly - d - 2.0 * u), target=(intervals[1][1], Ly - d))
    sk.domain("domain", (Lx / 2, Ly * 0.3))
    sk.dim((0, 0), (Lx, 0), -3.0, "Lx")
    sk.dim((0, 0), (0, Ly), 3.0, "Ly")
    x0, x1 = intervals[-1]
    sk.dim((x0, Ly), (x1, Ly), 2.0, "Notch_Width", outside=True)
    sk.dim((x1, Ly - d), (x1, Ly), -1.2, "Notch_Depth", ext=False)
    sk.dim((0, Ly), (pos[0], Ly), 4.6, "Notch_Positions[0]", ext=False)
    sk.line([(pos[0], Ly - d), (pos[0], Ly + 5.1 * u)], color=INK_SOFT, lw=DIM_LW, ls="-.", z=4)
    sk.line([(0, Ly + 0.3 * u), (0, Ly + 5.1 * u)], color=INK_SOFT, lw=DIM_LW, z=4)
    sk.note("default centres: Lx (i + 1/2) / nNotches, i = 0 ... nNotches - 1", (Lx * 0.3, Ly + 6.8 * u))
    sk.origin()
    sk.triad((-8 * u, Ly - 5 * u))
    return fig


def draw_rectangle_notches_3d(p):
    Lx, Ly, Lz, d = p.Lx, p.Ly, p.Lz, p.Notch_Depth
    intervals, pos = _notch_intervals(p)
    D = depth_vector(Ly)
    names = ["left", "top", "back", "notch", "front", "right", "bottom"]
    xlim = (-1.0, Lx + D[0] + 1.6)
    ylim = (-0.45, Lz + D[1] + 0.55)
    fig, (ax,), scale = make_figure([(xlim, ylim)])
    sk = Sketch(ax, scale, group_colours(names))
    u = sk.u
    prism(sk, _notch_outline(Lx, Lz, intervals, d), D)
    x0, x1 = intervals[-1]
    sk.group("left", (Lx * 0.72, Lz * 0.12), text="left (y = 0)")
    sk.group("top", (Lx * 0.35, Lz + D[1] + 0.3), target=(0.5 * intervals[0][0] + 0.5 * D[0], Lz + 0.5 * D[1]),
             text="top (z = Lz)", ha="right")
    sk.group("notch", (Lx * 0.6, Lz + D[1] + 0.3), target=(0.5 * (x0 + x1) + 0.5 * D[0], Lz - d + 0.5 * D[1]))
    sk.group("back", (Lx + D[0] + 0.15, Lz * 0.35 + 0.5 * D[1]), target=(Lx + 0.5 * D[0], Lz * 0.5 + 0.5 * D[1]),
             text="back (x = Lx)", ha="left")
    sk.group("front", (-0.3, Lz * 0.85), target=(0.5 * D[0], Lz * 0.5 + 0.5 * D[1]),
             text="front (x = 0)", ha="right", dashed=True)
    sk.group("right", (Lx + D[0] + 0.35, Lz + D[1] - 0.05), target=(Lx * 0.88 + D[0], Lz * 0.6 + D[1]),
             text="right (y = Ly)", ha="left", dashed=True)
    sk.group("bottom", (Lx + 0.9, -0.2), target=(Lx * 0.7 + 0.5 * D[0], 0.5 * D[1]),
             text="bottom (z = 0)", ha="left", dashed=True)
    sk.dim((0, 0), (Lx, 0), -2.2, "Lx")
    sk.dim((0, 0), (0, Lz), 2.2, "Lz")
    sk.dim((Lx, 0), (Lx + D[0], D[1]), -2.2, "Ly")
    xa, xb = intervals[1]
    sk.dim((xa, Lz - d), (xb, Lz - d), -1.6, "Notch_Width", outside=True, ext=False)
    sk.dim((xb, Lz - d), (xb, Lz), -1.4, "Notch_Depth", ext=False)
    yr = Lz - d - 4.0 * u
    sk.dim((0, yr), (pos[0], yr), 0.0, "Notch_Positions[0]", ext=False)
    sk.line([(pos[0], yr - 0.8 * u), (pos[0], Lz - d)], color=INK_SOFT, lw=DIM_LW, ls="-.", z=4)
    sk.origin(text="(0, 0, 0)")
    sk.triad((Lx + D[0] + 0.25, 0.05), labels=("x", "y", "z"), dirs=((1, 0), D / np.linalg.norm(D), (0, 1)))
    sk.note("notches cut from z = Lz through the full y extent", (xlim[0] + 0.05, ylim[0] + 0.05))
    return fig


def draw_rectangle_centre_notch(p):
    Lx, Ly, w, h = p.Lx, p.Ly, p.Notch_Width, p.Notch_Depth
    names = ["bottom", "right", "top", "left"]
    fig, (ax,), scale = make_figure([_pad(0, Lx, 0, Ly, 1.3, 0.9, 0.8, 0.4)])
    colours = group_colours(names)
    sk = Sketch(ax, scale, colours)
    u = sk.u
    sk.fill(rect(0, 0, Lx, Ly))
    x0, x1, y0, y1 = Lx / 2 - w / 2, Lx / 2 + w / 2, Ly / 2 - h / 2, Ly / 2 + h / 2
    sk.fill(rect(x0, y0, x1, y1), color="white", z=1)
    pad = 5 * p.dx_notch
    sk.line(np.vstack([rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad)] * 1)[[0, 1, 2, 3, 0]], color=GHOST, ls="--", lw=0.8)
    sk.line(rect(x0, y0, x1, y1)[[0, 1, 2, 3, 0]], color=INK, lw=OUTLINE_LW)
    for g, pts in zip(names, [[(0, 0), (Lx, 0)], [(Lx, 0), (Lx, Ly)], [(Lx, Ly), (0, Ly)], [(0, Ly), (0, 0)]]):
        sk.edge(pts, g)
    sk.group("bottom", (Lx / 2, 1.6 * u))
    sk.group("top", (Lx / 2, Ly - 1.6 * u))
    sk.group("left", (1.2 * u, Ly / 2), ha="left")
    sk.group("right", (Lx - 1.2 * u, Ly / 2), ha="right")
    sk.domain("domain", (Lx * 0.25, Ly * 0.7))
    sk.dim((0, 0), (Lx, 0), -3.0, "Lx")
    sk.dim((0, 0), (0, Ly), 3.0, "Ly")
    sk.dim((x0, y1 + pad), (x1, y1 + pad), 1.2, "Notch_Width", ext=False)
    sk.dim((x1 + pad, y0), (x1 + pad, y1), -1.2, "Notch_Depth", ext=False)
    sk.line([(x1, y0), (x1 + pad + 1.5 * u, y0)], color=INK_SOFT, lw=DIM_LW)
    sk.line([(x1, y1), (x1 + pad + 1.5 * u, y1)], color=INK_SOFT, lw=DIM_LW)
    sk.note("centred at (Lx/2, Ly/2)\nedge not tagged", (x0 - pad - 0.5 * u, y0 - pad - 2.2 * u), ha="right")
    sk.note("dx_notch inside the dashed box\n(5 dx_notch around the notch)", (x1 + pad + 1.0 * u, y0 - pad - 2.6 * u))
    sk.origin()
    sk.triad((-8 * u, Ly - 5 * u))
    return fig


def draw_rectangle_refined_top(p):
    Lx, Ly, Lr, xt = p.Lx, p.Ly, p.Ly_refined, p.Lx_topleft
    a = p.AlignWithRectangle
    names = ["bottom", "right", "top", "left", "top-left"]
    fig, (ax,), scale = make_figure([_pad(0, Lx, 0, Ly, 2.8, 2.8, 1.5, 1.6)])
    sk = Sketch(ax, scale, group_colours(names))
    u = sk.u
    yi = Ly - Lr
    sk.fill(rect(0, 0, Lx, Ly))
    sk.fill(rect(0, yi, Lx, Ly), color=SIDE_FILL, z=0.5)
    sk.line([(0, yi), (Lx, yi)], color=INK_SOFT, lw=0.8, ls="--")
    sk.line([(xt, yi), (xt, Ly)], color=INK_SOFT, lw=0.8, ls="--")
    sk.line(rect(a["x_min"], a["y_min"], a["x_max"], a["y_max"])[[0, 1, 2, 3, 0]], color=INK_SOFT, lw=0.8, ls="--")
    sk.edge([(0, 0), (Lx, 0)], "bottom")
    sk.edge([(Lx, 0), (Lx, Ly)], "right")
    sk.edge([(Lx, Ly), (xt, Ly)], "top")
    sk.edge([(xt, Ly), (0, Ly)], "top-left")
    sk.edge([(0, Ly), (0, 0)], "left")
    sk.group("bottom", (Lx * 0.2, 1.6 * u))
    sk.group("top", (Lx * 0.6, Ly + 1.8 * u))
    sk.group("top-left", (xt / 2, Ly + 1.8 * u))
    sk.group("left", (1.2 * u, yi * 0.3), ha="left")
    sk.group("right", (Lx - 1.2 * u, yi * 0.3), ha="right")
    sk.domain("domain", (Lx * 0.2, yi * 0.75))
    sk.note("dx_min", (Lx * 0.82, Ly - Lr / 2), ha="center")
    sk.note("dx_max", (Lx * 0.45, yi * 0.2), ha="center")
    ax0, ax1, ay0, ay1 = a["x_min"], a["x_max"], a["y_min"], a["y_max"]
    sk.note("AlignWithRectangle", (0.5 * (ax0 + ax1), 0.5 * (ay0 + ay1)), ha="center")
    sk.dim((0, 0), (Lx, 0), -3.0, "Lx")
    sk.dim((0, 0), (0, Ly), 3.0, "Ly")
    sk.dim((Lx, yi), (Lx, Ly), -3.0, "Ly_refined")
    sk.dim((0, Ly), (xt, Ly), 4.6, "Lx_topleft", outside=True)
    sk.dim((0, ay1 + 1.6 * u), (ax0, ay1 + 1.6 * u), 0.0, "x_min", ext=False)
    sk.dim((0, ay0 - 1.6 * u), (ax1, ay0 - 1.6 * u), 0.0, "x_max", ext=False, shift=-6)
    sk.dim((ax1 + 1.6 * u, 0), (ax1 + 1.6 * u, ay0), 0.0, "y_min", ext=False)
    sk.dim((ax1 + 7.0 * u, 0), (ax1 + 7.0 * u, ay1), 0.0, "y_max", ext=False)
    for pts in ([(ax0, ay1), (ax0, ay1 + 2.2 * u)], [(ax1, ay0), (ax1, ay0 - 2.2 * u)],
                [(ax1, ay0), (ax1 + 2.2 * u, ay0)], [(ax1, ay1), (ax1 + 7.6 * u, ay1)]):
        sk.line(pts, color=INK_SOFT, lw=DIM_LW, z=4)
    sk.origin()
    sk.triad((-10 * u, Ly - 3 * u))
    return fig


def _plate_quarter(Lx, Ly, R):
    return np.vstack([[(R, 0), (Lx, 0), (Lx, Ly), (0, Ly), (0, R)], arc((0, 0), R, R, 90, 0, n=40)[1:-1]])


def draw_plate_with_hole(p):
    Lx, Ly, R = p.Lx, p.Ly, p.R
    names = ["bottom", "right", "top", "left", "hole"]
    fig, (ax,), scale = make_figure([_pad(0, Lx, 0, Ly, 0.6, 0.45, 0.4, 0.15)])
    sk = Sketch(ax, scale, group_colours(names))
    u = sk.u
    sk.fill(_plate_quarter(Lx, Ly, R))
    sk.edge([(R, 0), (Lx, 0)], "bottom")
    sk.edge([(Lx, 0), (Lx, Ly)], "right")
    sk.edge([(Lx, Ly), (0, Ly)], "top")
    sk.edge([(0, Ly), (0, R)], "left")
    sk.edge(arc((0, 0), R, R, 0, 90), "hole")
    sk.group("bottom", ((R + Lx) / 2, 1.6 * u))
    sk.group("top", (Lx / 2, Ly - 1.6 * u))
    sk.group("left", (1.2 * u, (R + Ly) / 2), ha="left")
    sk.group("right", (Lx - 1.2 * u, Ly / 2), ha="right")
    sk.group("hole", (R + 3 * u, R + 2.5 * u), target=tuple(arc((0, 0), R, R, 45, 45, n=1)[0]), ha="left")
    sk.domain("domain", (Lx * 0.6, Ly * 0.45))
    sk.radius((0, 0), arc((0, 0), R, R, 20, 20, n=1)[0], "R", direction=(0.3, -1))
    sk.dim((0, 0), (Lx, 0), -3.0, "Lx")
    sk.dim((0, 0), (0, Ly), 3.0, "Ly")
    sk.origin(text="(0, 0), hole centre")
    sk.triad((-5.5 * u, Ly - 5 * u))
    return fig


def _faces_3d_extruded(sk, Lx, Ly, Lz, D, positions):
    """Face labels for plates extruded along z: x right, y up, z towards the viewer.

    The undistorted near face is ``z = Lz``; ``D`` points from it to the far face ``z = 0``.
    """
    pos = positions
    sk.group("top", pos["near"], text="top (z = Lz)")
    sk.group("right", pos["right"], text="right (y = Ly)")
    sk.group("back", pos["back"], text="back (x = Lx)", ha="left", target=(Lx + 0.5 * D[0], Ly * 0.5 + 0.5 * D[1]))
    sk.group("bottom", pos["far"], target=(Lx * 0.8 + D[0], Ly * 0.55 + D[1]), text="bottom (z = 0)", ha="left", dashed=True)
    sk.group("front", pos["front"], target=(0.5 * D[0], Ly * 0.7 + 0.5 * D[1]), text="front (x = 0)", ha="right", dashed=True)
    sk.group("left", pos["left"], target=(Lx * 0.75 + 0.5 * D[0], 0.5 * D[1]), text="left (y = 0)", ha="left", dashed=True)


def draw_plate_with_hole_3d(p):
    Lx, Ly, Lz, R = p.Lx, p.Ly, p.Lz, p.R
    D = depth_vector(Lz)
    names = ["bottom", "right", "back", "top", "front", "left", "hole"]
    xlim = (-0.75, Lx + D[0] + 1.35)
    ylim = (-0.45, Ly + D[1] + 0.45)
    fig, (ax,), scale = make_figure([(xlim, ylim)])
    sk = Sketch(ax, scale, group_colours(names))
    prism(sk, _plate_quarter(Lx, Ly, R), D)
    _faces_3d_extruded(sk, Lx, Ly, Lz, D, {
        "near": (Lx * 0.62, Ly * 0.72), "right": (Lx * 0.45 + 0.5 * D[0], Ly + 0.5 * D[1]),
        "back": (Lx + D[0] + 0.12, Ly * 0.25), "far": (Lx + D[0] + 0.12, Ly + D[1] + 0.25),
        "front": (-0.12, Ly * 0.9 + D[1]), "left": (Lx + 0.25, -0.3)})
    sk.group("hole", (R + 0.35, R + 0.35), target=tuple(arc((0, 0), R, R, 75, 75, n=1)[0] + 0.5 * D), ha="left")
    sk.radius((0, 0), arc((0, 0), R, R, 40, 40, n=1)[0], "R", direction=(-0.6, 1))
    sk.dim((0, 0), (Lx, 0), -2.4, "Lx")
    sk.dim((0, 0), (0, Ly), 2.4, "Ly")
    sk.dim((Lx, 0), (Lx + D[0], D[1]), -2.4, "Lz")
    sk.origin(at=tuple(D), text="(0, 0, 0), behind", direction=(1, 0.05))
    sk.triad((-0.6, Ly * 0.3), labels=("x", "y", "z"), dirs=((1, 0), (0, 1), -D))
    sk.note("hole axis: x = y = 0", (Lx * 0.4, -0.38))
    return fig


def _ellipse(p, n=160):
    c = (p.Lx / 2, p.Ly / 2)
    return arc(c, p.R, p.R / getattr(p, "ellipticity", 1.0), 0, 360, getattr(p, "rotation_angle", 0.0), n=n)[:-1]


def _offset_curve(pts, dist):
    """Closed curve ``pts`` (counter-clockwise) offset outwards by ``dist``."""
    t = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    n = np.column_stack([t[:, 1], -t[:, 0]])
    n /= np.linalg.norm(n, axis=1)[:, None]
    return pts + dist * n


def _hole_dims(sk, p, with_refine):
    c = np.array([p.Lx / 2, p.Ly / 2])
    rot = np.radians(p.rotation_angle)
    e = np.array([np.cos(rot), np.sin(rot)])
    f = np.array([-np.sin(rot), np.cos(rot)])
    rx, ry = p.R, p.R / p.ellipticity
    sk.line([c - 1.3 * rx * e, c + 1.3 * rx * e], color=INK_SOFT, lw=DIM_LW, ls="-.")
    sk.line([c, c + 1.3 * rx * np.array([1.0, 0.0])], color=INK_SOFT, lw=DIM_LW, ls="-.")
    sk._arrow(c, c + rx * e, "-|>")
    sk._arrow(c, c + ry * f, "-|>")
    sk.ax.plot(*c, marker="+", ms=5, color=INK_SOFT, mew=DIM_LW, zorder=4)
    sk._text_beside("R", c + 0.6 * rx * e, -f)
    sk.leader("R / ellipticity", tuple(c + np.array([-1.5 * rx, 1.1 * rx])), tuple(c + 0.6 * ry * f))
    ra = 0.55 * rx
    pts = arc(c, ra, ra, 0, p.rotation_angle, n=30)
    sk.line(pts[:-2], color=INK_SOFT, lw=DIM_LW)
    sk._arrow(pts[-3], pts[-1], "-|>")
    sk.leader("rotation_angle", tuple(c + np.array([1.4 * rx, -0.45 * rx])), tuple(pts[12]))
    if with_refine:
        ring = _offset_curve(_ellipse(p), p.R_Refine)
        sk.line(np.vstack([ring, ring[:1]]), color=INK_SOFT, lw=0.8, ls="--")
        ring2 = _offset_curve(_ellipse(p), 2 * p.R_Refine)
        sk.line(np.vstack([ring2, ring2[:1]]), color=GHOST, lw=0.8, ls="--")


def draw_full_plate_with_hole(p):
    Lx, Ly = p.Lx, p.Ly
    names = ["bottom", "right", "top", "left", "hole"]
    fig, (ax,), scale = make_figure([_pad(0, Lx, 0, Ly, 1.0, 2.0, 0.5, 0.2)])
    sk = Sketch(ax, scale, group_colours(names))
    u = sk.u
    sk.fill(rect(0, 0, Lx, Ly))
    hole = _ellipse(p)
    sk.fill(hole, color="white", z=0.5)
    for g, pts in zip(names, [[(0, 0), (Lx, 0)], [(Lx, 0), (Lx, Ly)], [(Lx, Ly), (0, Ly)], [(0, Ly), (0, 0)]]):
        sk.edge(pts, g)
    sk.edge(np.vstack([hole, hole[:1]]), "hole")
    _hole_dims(sk, p, True)
    sk.group("bottom", (Lx / 2, 1.6 * u))
    sk.group("top", (Lx * 0.3, Ly - 1.6 * u))
    sk.group("left", (1.2 * u, Ly / 2), ha="left")
    sk.group("right", (Lx - 1.2 * u, Ly * 0.2), ha="right")
    sk.group("hole", (Lx * 0.62, Ly * 0.18), target=tuple(hole[100]), ha="left")
    sk.domain("domain", (Lx * 0.18, Ly * 0.2))
    sk.dim((0, 0), (Lx, 0), -3.0, "Lx")
    sk.dim((0, 0), (0, Ly), 3.0, "Ly")
    for dist, text, direction, y in ((p.R_Refine, "R_Refine", (1.0, 1.0), 0.88), (2 * p.R_Refine, "2 R_Refine", (1.0, 0.3), 0.74)):
        ring = _offset_curve(hole, dist)
        target = ring[np.argmax(ring @ np.array(direction))]
        sk.leader(text, (Lx + 1.2 * u, Ly * y), tuple(target))
    sk.note("dxmin within R_Refine,\ndx beyond 2 R_Refine", (Lx + 1.2 * u, Ly * 0.42))
    sk.note("hole centre (Lx/2, Ly/2)", (Lx + 1.2 * u, Ly * 0.28))
    sk.origin()
    sk.triad((-8 * u, Ly - 5 * u))
    return fig


def draw_full_plate_with_hole_3d(p):
    Lx, Ly, Lz = p.Lx, p.Ly, p.Lz
    D = depth_vector(Lz)
    names = ["bottom", "right", "back", "top", "front", "left", "hole"]
    xlim = (-1.2, Lx + D[0] + 1.6)
    ylim = (-0.65, Ly + D[1] + 0.45)
    fig, (ax,), scale = make_figure([(xlim, ylim)])
    sk = Sketch(ax, scale, group_colours(names))
    hole = _ellipse(p)
    prism(sk, rect(0, 0, Lx, Ly), D, holes=[hole[::-1]])
    _hole_dims(sk, p, False)
    _faces_3d_extruded(sk, Lx, Ly, Lz, D, {
        "near": (Lx * 0.75, Ly * 0.85), "right": (Lx * 0.3 + 0.5 * D[0], Ly + 0.5 * D[1]),
        "back": (Lx + D[0] + 0.12, Ly * 0.25), "far": (Lx + D[0] + 0.12, Ly + D[1] + 0.25),
        "front": (-0.15, Ly * 0.9 + D[1]), "left": (Lx + 0.3, -0.45)})
    sk.group("hole", (Lx * 0.62, Ly * 0.18), target=tuple(hole[100] + 0.3 * D), ha="left")
    sk.dim((0, 0), (Lx, 0), -2.4, "Lx")
    sk.dim((0, 0), (0, Ly), 2.4, "Ly")
    sk.dim((Lx, 0), (Lx + D[0], D[1]), -2.4, "Lz")
    sk.origin(at=tuple(D), text="(0, 0, 0), behind", direction=(1, 1))
    sk.triad((-0.85, Ly * 0.4), labels=("x", "y", "z"), dirs=((1, 0), (0, 1), -D))
    return fig


def draw_half_plate_with_hole(p):
    Lx, Ly = p.Lx, p.Ly
    names = ["bottom", "right", "top", "symmetry", "hole"]
    fig, (ax,), scale = make_figure([_pad(0, Lx, 0, Ly, 0.5, 2.0, 0.85, 0.2)])
    sk = Sketch(ax, scale, group_colours(names))
    u = sk.u
    c = np.array([Lx / 2, Ly / 2])
    R, Ry = p.R, p.R / getattr(p, "ellipticity", 1.0)
    half = np.vstack([[(Lx / 2, 0), (Lx, 0), (Lx, Ly), (Lx / 2, Ly)], arc(c, R, Ry, 90, -90, n=60)])
    sk.line(rect(0, 0, Lx / 2, Ly)[[1, 0, 3, 2]], color=GHOST, ls="--", lw=0.8)
    sk.line(arc(c, R, Ry, 90, 270, n=60), color=GHOST, ls="--", lw=0.8)
    sk.note("not meshed", (Lx * 0.08, Ly * 0.88))
    sk.fill(half)
    sk.edge([(Lx / 2, 0), (Lx, 0)], "bottom")
    sk.edge([(Lx, 0), (Lx, Ly)], "right")
    sk.edge([(Lx, Ly), (Lx / 2, Ly)], "top")
    sk.edge([(Lx / 2, Ly), (Lx / 2, Ly / 2 + Ry)], "symmetry")
    sk.edge([(Lx / 2, Ly / 2 - Ry), (Lx / 2, 0)], "symmetry")
    sk.edge(arc(c, R, Ry, -90, 90), "hole")
    sk.group("bottom", (Lx * 0.75, 1.6 * u))
    sk.group("top", (Lx * 0.75, Ly - 1.6 * u))
    sk.group("symmetry", (Lx / 2 + 1.2 * u, Ly * 0.82), ha="left")
    sk.group("symmetry", (Lx / 2 + 1.2 * u, Ly * 0.18), ha="left")
    sk.group("right", (Lx - 1.2 * u, Ly * 0.4), ha="right")
    sk.group("hole", (Lx / 2 + R + 4 * u, Ly / 2 - 0.5 * u), target=tuple(arc(c, R, Ry, 15, 15, n=1)[0]), ha="left")
    sk.domain("domain", (Lx * 0.85, Ly * 0.62))
    sk.radius(c, c + np.array([R * np.cos(np.radians(-30)), R * np.sin(np.radians(-30))]), "R", direction=(0.4, -1))
    ring = np.vstack([[(Lx / 2, Ly / 2 + R + p.R_Refine)], arc(c, R + p.R_Refine, R + p.R_Refine, 90, -90, n=60)])
    sk.line(ring, color=INK_SOFT, lw=0.8, ls="--")
    ring2 = arc(c, R + 2 * p.R_Refine, R + 2 * p.R_Refine, 90, -90, n=60)
    sk.line(ring2, color=GHOST, lw=0.8, ls="--")
    sk.leader("R_Refine", (Lx + 1.2 * u, Ly * 0.8), tuple(arc(c, R + p.R_Refine, R + p.R_Refine, 60, 60, n=1)[0]))
    sk.leader("2 R_Refine", (Lx + 1.2 * u, Ly * 0.66), tuple(arc(c, R + 2 * p.R_Refine, R + 2 * p.R_Refine, 35, 35, n=1)[0]))
    sk.note("dxmin within R_Refine,\ndx beyond 2 R_Refine", (Lx + 1.2 * u, Ly * 0.5))
    sk.note("hole centre (Lx/2, Ly/2)", (Lx + 1.2 * u, Ly * 0.36))
    sk.note("ellipticity, rotation_angle:\nas FullPlateWithHole", (Lx + 1.2 * u, Ly * 0.2))
    sk.dim((0, 0), (Lx, 0), -4.6, "Lx (full plate)")
    sk.dim((Lx / 2, 0), (Lx, 0), -1.2, "Lx / 2", ext=False)
    sk.dim((0, 0), (0, Ly), 3.0, "Ly")
    sk.origin()
    sk.triad((Lx * 0.2, Ly * 0.4))
    return fig


def _sen_outline(Lx, Ly, a, h):
    y0, y1 = 0.5 * (Ly - h), 0.5 * (Ly + h)
    return np.array([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly), (0, y1), (a, y1), (a, y0), (0, y0)], float)


def draw_single_edge_notched(p):
    Lx, Ly, a, h = p.Lx, p.Ly, p.LNotch, p.NotchHeight
    names = ["bottom", "right", "top", "left", "notch"]
    fig, (ax,), scale = make_figure([_pad(0, Lx, 0, Ly, 0.42, 0.12, 0.2, 0.06)], width=5.4)
    sk = Sketch(ax, scale, group_colours(names))
    u = sk.u
    y0, y1 = 0.5 * (Ly - h), 0.5 * (Ly + h)
    sk.fill(_sen_outline(Lx, Ly, a, h))
    sk.edge([(0, 0), (Lx, 0)], "bottom")
    sk.edge([(Lx, 0), (Lx, Ly)], "right")
    sk.edge([(Lx, Ly), (0, Ly)], "top")
    sk.edge([(0, Ly), (0, y1)], "left")
    sk.edge([(0, y0), (0, 0)], "left")
    sk.edge([(0, y1), (a, y1), (a, y0), (0, y0)], "notch")
    sk.group("bottom", (Lx / 2, 1.6 * u))
    sk.group("top", (Lx / 2, Ly - 1.6 * u))
    sk.group("left", (1.2 * u, Ly * 0.25), ha="left")
    sk.group("right", (Lx - 1.2 * u, Ly * 0.75), ha="right")
    sk.group("notch", (a * 0.5, Ly / 2 + 3.0 * u), target=(a * 0.5, y1))
    sk.domain("domain", (Lx * 0.7, Ly * 0.25))
    sk.dim((0, 0), (Lx, 0), -3.0, "Lx")
    sk.dim((0, 0), (0, Ly), 3.0, "Ly")
    sk.dim((0, y0), (a, y0), -2.5, "LNotch", ext=False)
    sk.dim((a, y0), (a, y1), -6.0, "NotchHeight", outside=True)
    sk.line([(a, y0), (a + 6.5 * u, y0)], color=INK_SOFT, lw=DIM_LW)
    sk.line([(a, y1), (a + 6.5 * u, y1)], color=INK_SOFT, lw=DIM_LW)
    sk.note("centred at y = Ly/2", (a + 3.0 * u, Ly / 2 - 3.6 * u))
    sk.origin()
    sk.triad((-7.5 * u, Ly - 5 * u))
    return fig


def draw_single_edge_notched_3d(p):
    Lx, Ly, Lz, a, h = p.Lx, p.Ly, p.Lz, p.LNotch, p.NotchHeight
    D = depth_vector(Lz)
    names = ["bottom", "right", "back", "top", "front", "left", "notch"]
    xlim = (-0.42, Lx + D[0] + 0.55)
    ylim = (-0.22, Ly + D[1] + 0.2)
    fig, (ax,), scale = make_figure([(xlim, ylim)], width=6.3)
    sk = Sketch(ax, scale, group_colours(names))
    u = sk.u
    y0, y1 = 0.5 * (Ly - h), 0.5 * (Ly + h)
    prism(sk, _sen_outline(Lx, Ly, a, h), D)
    _faces_3d_extruded(sk, Lx, Ly, Lz, D, {
        "near": (Lx * 0.6, Ly * 0.25), "right": (Lx * 0.45 + 0.5 * D[0], Ly + 0.5 * D[1]),
        "back": (Lx + D[0] + 0.05, Ly * 0.25), "far": (Lx + D[0] + 0.05, Ly + D[1] + 0.1),
        "front": (-0.05, Ly * 0.9 + D[1]), "left": (Lx + 0.12, -0.14)})
    sk.group("notch", (a + 0.2, Ly / 2 + 0.17), target=(a + 0.5 * D[0], y1 + 0.5 * D[1]), ha="left")
    sk.dim((0, 0), (Lx, 0), -2.4, "Lx")
    sk.dim((0, 0), (0, Ly), 2.4, "Ly")
    sk.dim((Lx, 0), (Lx + D[0], D[1]), -2.4, "Lz")
    sk.dim((0, y0), (a, y0), -2.5, "LNotch", ext=False)
    sk.dim((a, y0), (a, y1), -6.0, "NotchHeight", outside=True)
    sk.line([(a, y0), (a + 6.5 * u, y0)], color=INK_SOFT, lw=DIM_LW)
    sk.line([(a, y1), (a + 6.5 * u, y1)], color=INK_SOFT, lw=DIM_LW)
    sk.origin(at=tuple(D), text="(0, 0, 0), behind", direction=(1, 1))
    sk.triad((-0.3, Ly * 0.3), labels=("x", "y", "z"), dirs=((1, 0), (0, 1), -D))
    return fig


def _dc_outline(Lx, Ly, xn, t, hn):
    return np.array([(0, 0), (xn - t / 2, 0), (xn - t / 2, hn), (xn + t / 2, hn), (xn + t / 2, 0),
                     (Lx, 0), (Lx, Ly), (0, Ly)], float)


def draw_damage_challenge(p):
    Lx, Ly, xn, t, hn, xb = p.Lx, p.Ly, p.Lx_notch, p.Notch_Thickness, p.Ly_notch, p.Lx_BC
    lw, bw = p.Load_Width, p.BC_Width
    names = ["bottom", "right", "top", "left", "notch", "top_load", "bottom_left", "bottom_right"]
    fig, (ax,), scale = make_figure([_pad(0, Lx, 0, Ly, 1.0, 0.65, 1.3, 1.05)])
    sk = Sketch(ax, scale, group_colours(names))
    u = sk.u
    sk.fill(_dc_outline(Lx, Ly, xn, t, hn))
    bottom_pts = sorted([0, xn - t / 2, xn + t / 2, Lx, xb - bw / 2, xb + bw / 2, Lx - xb - bw / 2, Lx - xb + bw / 2])
    for a, b in zip(bottom_pts[:-1], bottom_pts[1:]):
        if abs(a - (xn - t / 2)) < 1e-12:
            continue
        mid = 0.5 * (a + b)
        g = "bottom_left" if abs(mid - xb) < bw / 2 else "bottom_right" if abs(mid - (Lx - xb)) < bw / 2 else "bottom"
        sk.edge([(a, 0), (b, 0)], g)
    sk.edge([(Lx, 0), (Lx, Ly)], "right")
    sk.edge([(Lx, Ly), (Lx / 2 + lw / 2, Ly)], "top")
    sk.edge([(Lx / 2 + lw / 2, Ly), (Lx / 2 - lw / 2, Ly)], "top_load")
    sk.edge([(Lx / 2 - lw / 2, Ly), (0, Ly)], "top")
    sk.edge([(0, Ly), (0, 0)], "left")
    sk.edge([(xn - t / 2, 0), (xn - t / 2, hn), (xn + t / 2, hn), (xn + t / 2, 0)], "notch")
    sk.group("bottom", (0.5 * (xn + Lx - xb), 1.6 * u))
    sk.group("top", (Lx * 0.2, Ly - 1.6 * u))
    sk.group("left", (1.2 * u, Ly / 2), ha="left")
    sk.group("right", (Lx - 1.2 * u, Ly / 2), ha="right")
    sk.group("notch", (xn + 3 * u, hn * 0.55), target=(xn + t / 2, hn * 0.55), ha="left")
    sk.group("top_load", (Lx / 2, Ly - 1.8 * u))
    sk.group("bottom_left", (xb, 2.0 * u))
    sk.group("bottom_right", (Lx - xb, 2.0 * u))
    sk.domain("domain", (Lx * 0.7, Ly * 0.55))
    sk.dim((0, 0), (Lx, 0), -5.4, "Lx")
    sk.dim((0, 0), (0, Ly), 3.0, "Ly")
    sk.dim((0, 0), (xb, 0), -2.6, "Lx_BC", ext=False)
    sk.dim((Lx - xb, 0), (Lx, 0), -2.6, "Lx_BC", ext=False)
    sk.line([(xb, 0), (xb, -3.1 * u)], color=INK_SOFT, lw=DIM_LW)
    sk.line([(Lx - xb, 0), (Lx - xb, -3.1 * u)], color=INK_SOFT, lw=DIM_LW)
    sk.dim((Lx - xb - bw / 2, 0), (Lx - xb + bw / 2, 0), 4.2, "BC_Width")
    sk.dim((xn - t / 2, hn), (xn + t / 2, hn), 1.6, "Notch_Thickness", outside=True, ext=False)
    sk.dim((xn - t / 2, 0), (xn - t / 2, hn), 2.2, "Ly_notch", ext=False)
    sk.dim((0, Ly), (xn, Ly), 2.0, "Lx_notch", ext=False, shift=-4)
    sk.line([(xn, hn + 4.0 * u), (xn, Ly + 2.5 * u)], color=INK_SOFT, lw=DIM_LW, ls="-.")
    sk.dim((Lx / 2 - lw / 2, Ly), (Lx / 2 + lw / 2, Ly), 2.0, "Load_Width", ext=True)
    sk.origin()
    sk.triad((-7 * u, Ly - 5 * u))
    return fig


def draw_damage_challenge_3d(p):
    Lx, Ly, Lz, t, hn, xb = p.Lx, p.Ly, p.Lz, p.Notch_Thickness, p.Ly_notch, p.Lx_BC
    x0, x1 = p.Lx_notch
    lw, bw = p.Load_Width, p.BC_Width
    D = depth_vector(Lz)
    tx = t * np.hypot(x1 - x0, Lz) / Lz          # notch width along x
    names = ["bottom", "right", "back", "top", "front", "left", "notch", "top_load", "bottom_left", "bottom_right"]
    colours = group_colours(names)
    panel_b = ((-1.3, Lx + D[0] + 1.9), (-0.75, Ly + D[1] + 0.6))
    panel_a = ((-1.3, Lx + 0.6), (-0.55, Lz + 0.45))
    fig, (ax2, ax1), scale = make_figure([panel_b, panel_a], titles=["(a) oblique view", "(b) notch in the x-z plane, viewed in the +y direction"],
                                         vertical=True)
    # (b) the notch trace in the x-z plane
    sa = Sketch(ax1, scale, colours)
    u = sa.u
    sa.fill(rect(0, 0, Lx, Lz))
    for xc, w, g in ((xb, bw, "bottom_left"), (Lx - xb, bw, "bottom_right"), (Lx / 2, lw, "top_load")):
        sa.fill(rect(xc - w / 2, 0, xc + w / 2, Lz), color=colours[g] + "55", z=0.6)
    sa.fill([(x0 - tx / 2, 0), (x0 + tx / 2, 0), (x1 + tx / 2, Lz), (x1 - tx / 2, Lz)], color="white", z=0.7)
    sa.line(rect(0, 0, Lx, Lz)[[0, 1, 2, 3, 0]])
    sa.edge([(x0 - tx / 2, 0), (x1 - tx / 2, Lz)], "notch")
    sa.edge([(x0 + tx / 2, 0), (x1 + tx / 2, Lz)], "notch")
    sa.dim((0, Lz), (x1, Lz), 2.2, "Lx_notch[1]", ext=False, shift=-3)
    sa.dim((0, 0), (x0, 0), -2.2, "Lx_notch[0]", ext=False, shift=-3)
    sa.dim((Lx, 0), (Lx, Lz), -2.0, "Lz")
    sa.line([(x0, 0), (x0, -2.7 * u)], color=INK_SOFT, lw=DIM_LW)
    sa.line([(x1, Lz), (x1, Lz + 2.7 * u)], color=INK_SOFT, lw=DIM_LW)
    sa.line([(0, 0), (0, -2.7 * u)], color=INK_SOFT, lw=DIM_LW)
    sa.line([(0, Lz), (0, Lz + 2.7 * u)], color=INK_SOFT, lw=DIM_LW)
    sa.line([(x0, 0), (x1, Lz)], color=INK_SOFT, lw=DIM_LW, ls="-.")
    sa.leader("Notch_Thickness (normal to the notch)", (Lx + 0.9, Lz * 0.5), (x1 + 0.3 * (x0 - x1) + tx / 2, 0.7 * Lz))
    sa.origin(text="(0, 0, 0)")
    sa.triad((-1.0, Lz * 0.15), labels=("x", "z"))
    # (a) oblique view
    sk = Sketch(ax2, scale, colours)
    front = _dc_outline(Lx, Ly, x1, tx, hn)          # near face z = Lz
    back = _dc_outline(Lx, Ly, x0, tx, hn) + D       # far face z = 0
    extra = [(front[k], back[k]) for k in (1, 2, 3, 4)]
    prism(sk, front, D, back_outer=back, extra=extra)
    for xc, w, g, y in ((Lx / 2, lw, "top_load", Ly), (xb, bw, "bottom_left", 0.0), (Lx - xb, bw, "bottom_right", 0.0)):
        strip = np.array([(xc - w / 2, y), (xc + w / 2, y), (xc + w / 2 + D[0], y + D[1]), (xc - w / 2 + D[0], y + D[1])])
        sk.ax.add_patch(Polygon(strip, closed=True, fc=colours[g] + ("90" if y > 0 else "30"), ec=colours[g],
                                lw=0.8, ls="-" if y > 0 else "--", zorder=2.5))
    sk.group("top", (Lx * 0.75, Ly * 0.6), text="top (z = Lz)")
    sk.group("right", (Lx * 0.2 + 0.5 * D[0], Ly + 0.5 * D[1]), text="right (y = Ly)")
    sk.group("back", (Lx + D[0] + 0.15, Ly * 0.55), text="back (x = Lx)", ha="left", target=(Lx + 0.5 * D[0], Ly * 0.5 + 0.5 * D[1]))
    sk.group("bottom", (Lx + D[0] + 0.15, Ly + D[1] + 0.25), target=(Lx * 0.85 + D[0], Ly * 0.5 + D[1]), text="bottom (z = 0)", ha="left", dashed=True)
    sk.group("front", (-0.15, Ly * 0.9 + D[1]), target=(0.5 * D[0], Ly * 0.7 + 0.5 * D[1]), text="front (x = 0)", ha="right", dashed=True)
    sk.group("left", (Lx + 0.6, -0.45), target=(Lx * 0.88 + 0.5 * D[0], 0.5 * D[1]), text="left (y = 0)", ha="left", dashed=True)
    sk.group("top_load", (Lx / 2 + 0.5 * D[0] + 0.6, Ly + D[1] + 0.35), target=(Lx / 2 + 0.6 * D[0], Ly + 0.6 * D[1]), ha="left")
    sk.group("bottom_left", (xb, -0.5), target=(xb + 0.5 * D[0], 0.5 * D[1]), dashed=True)
    sk.group("bottom_right", (Lx - xb - 0.4, Ly * 0.3), target=(Lx - xb + 0.5 * D[0], 0.5 * D[1]), dashed=True, ha="right")
    sk.group("notch", (x1 - 0.35, hn + 0.45), target=(x1 + 0.3 * D[0] + 0.3 * (x0 - x1), hn + 0.3 * D[1]), ha="right")
    sk.dim((0, 0), (Lx, 0), -2.6, "Lx")
    sk.dim((0, 0), (0, Ly), 2.4, "Ly")
    sk.dim((Lx, 0), (Lx + D[0], D[1]), -2.4, "Lz")
    sk.dim((x1 + tx / 2, 0), (x1 + tx / 2, hn), -1.2, "Ly_notch", ext=False)
    sk.origin(at=tuple(D), text="(0, 0, 0), behind", direction=(1, 1))
    sk.triad((-1.0, Ly * 0.35), labels=("x", "y", "z"), dirs=((1, 0), (0, 1), -D))
    sk.note("Lx_BC, BC_Width and Load_Width as in 2-D", (-1.25, Ly + D[1] + 0.5))
    return fig


def draw_cylinder(p):
    R, Lz, nr, nh, ndh = p.R, p.Lz, p.notch_r, p.notch_h, p.notch_dh
    names = ["bottom", "top", "outer", "notch"]
    fig, (ax,), scale = make_figure([((-R - 0.75, R + 0.9), (-0.35, Lz + 0.35))], width=4.7)
    sk = Sketch(ax, scale, group_colours(names))
    u = sk.u
    z0, z1 = nh - ndh / 2, nh + ndh / 2
    half = [(0, 0), (R, 0), (R, z0), (R - nr, z0), (R - nr, z1), (R, z1), (R, Lz), (0, Lz)]
    sec = np.array(half + [(-x, z) for x, z in half[::-1]], float)
    sk.fill(sec)
    sk.edge([(-R, 0), (R, 0)], "bottom")
    sk.edge([(-R, Lz), (R, Lz)], "top")
    for sgn in (1, -1):
        sk.edge([(sgn * R, 0), (sgn * R, z0)], "outer")
        sk.edge([(sgn * R, z1), (sgn * R, Lz)], "outer")
        sk.edge([(sgn * R, z0), (sgn * (R - nr), z0), (sgn * (R - nr), z1), (sgn * R, z1)], "notch")
    sk.line([(0, -0.25), (0, Lz + 0.25)], color=INK_SOFT, lw=DIM_LW, ls="-.")
    sk.note("axis x = y = 0", (-0.05, Lz + 0.25), ha="right")
    sk.group("bottom", (R / 2, 1.6 * u))
    sk.group("top", (R / 2, Lz - 1.6 * u))
    sk.group("outer", (R - 1.2 * u, Lz * 0.25), ha="right")
    sk.group("notch", (R - nr - 1.2 * u, nh), ha="right")
    sk.domain("domain", (R / 2, Lz * 0.75))
    sk.dim((0, Lz), (R, Lz), 2.2, "R", ext=False)
    sk.line([(R, Lz), (R, Lz + 2.7 * u)], color=INK_SOFT, lw=DIM_LW)
    sk.dim((R, 0), (R, Lz), -3.0, "Lz")
    sk.dim((-R, z1), (-R + nr, z1), 1.6, "notch_r", outside=True, ext=False, shift=3)
    sk.dim((-R, z0), (-R, z1), 4.0, "notch_dh", outside=True)
    sk.dim((-R * 0.45, 0), (-R * 0.45, nh), 0.0, "notch_h", ext=False)
    sk.line([(-R * 0.45 - 0.8 * u, nh), (-(R - nr), nh)], color=INK_SOFT, lw=DIM_LW, ls="-.")
    sk.origin(text="(0, 0, 0)", direction=(1, -1))
    sk.triad((-R - 0.6, Lz * 0.8), labels=("x", "z"))
    sk.note("section through the axis; the groove runs round", (-R - 0.7, -0.25), ha="left")
    return fig


def draw_notched_tensile_bar(p):
    Lx, Ly, Lc, w, d = p.Lx, p.Ly, p.Lx_centre, p.Notch_Width, p.Notch_Depth
    names = ["bottom", "right", "top", "left", "notch", "interface"]
    fig, (ax,), scale = make_figure([_pad(0, Lx, 0, Ly, 0.55, 0.15, 0.4, 0.5)])
    colours = group_colours(names)
    sk = Sketch(ax, scale, colours)
    u = sk.u
    xc0, xc1 = (Lx - Lc) / 2, (Lx + Lc) / 2
    n0, n1 = (Lx - w) / 2, (Lx + w) / 2
    sk.fill(rect(0, 0, xc0, Ly), color=DOMAIN_TINTS[0])
    sk.fill(np.array([(xc0, 0), (n0, 0), (n0, d), (n1, d), (n1, 0), (xc1, 0), (xc1, Ly), (n1, Ly),
                      (n1, Ly - d), (n0, Ly - d), (n0, Ly), (xc0, Ly)]), color=DOMAIN_TINTS[1])
    sk.fill(rect(xc1, 0, Lx, Ly), color=DOMAIN_TINTS[2])
    sk.edge([(0, 0), (n0, 0)], "bottom")
    sk.edge([(n1, 0), (Lx, 0)], "bottom")
    sk.edge([(0, Ly), (n0, Ly)], "top")
    sk.edge([(n1, Ly), (Lx, Ly)], "top")
    sk.edge([(0, 0), (0, Ly)], "left")
    sk.edge([(Lx, 0), (Lx, Ly)], "right")
    sk.edge([(n0, 0), (n0, d), (n1, d), (n1, 0)], "notch")
    sk.edge([(n0, Ly), (n0, Ly - d), (n1, Ly - d), (n1, Ly)], "notch")
    sk.edge([(xc0, 0), (xc0, Ly)], "interface")
    sk.edge([(xc1, 0), (xc1, Ly)], "interface")
    sk.domain("domain:left", (xc0 / 2, Ly * 0.5))
    sk.domain("domain:centre", (Lx / 2, Ly * 0.5))
    sk.domain("domain:right", (Lx - xc0 / 2, Ly * 0.5))
    sk.group("bottom", (xc0 / 2, 1.6 * u), text="bottom:left")
    sk.group("bottom", (Lx - xc0 / 2, 1.6 * u), text="bottom:right")
    sk.group("top", (xc0 / 2, Ly - 1.6 * u), text="top:left")
    sk.group("top", (Lx - xc0 / 2, Ly - 1.6 * u), text="top:right")
    sk.group("bottom", ((xc0 + n0) / 2, -2.4 * u), target=((xc0 + n0) / 2, 0), text="bottom:centre")
    sk.group("top", ((xc0 + n0) / 2, Ly + 2.4 * u), target=((xc0 + n0) / 2, Ly), text="top:centre")
    sk.group("left", (1.2 * u, Ly * 0.25), ha="left", text="left:left")
    sk.group("right", (Lx - 1.2 * u, Ly * 0.25), ha="right", text="right:right")
    sk.group("notch", (Lx / 2, Ly * 0.3), target=(n0, d * 0.6), text="notch:centre")
    sk.group("interface", (xc0 - 1.0 * u, Ly * 0.78), ha="right", text="interface:left")
    sk.group("interface", (xc1 + 1.0 * u, Ly * 0.78), ha="left", text="interface:right")
    sk.dim((0, 0), (Lx, 0), -5.2, "Lx")
    sk.dim((0, 0), (0, Ly), 3.0, "Ly")
    sk.dim((xc0, Ly), (xc1, Ly), 5.6, "Lx_centre")
    sk.dim((n0, Ly), (n1, Ly), 2.4, "Notch_Width", outside=True, ext=False, shift=8)
    sk.line([(n1, Ly), (n1, Ly + 2.9 * u)], color=INK_SOFT, lw=DIM_LW)
    sk.line([(n0, Ly), (n0, Ly + 2.9 * u)], color=INK_SOFT, lw=DIM_LW)
    sk.dim((n1, 0), (n1, d), -3.0, "Notch_Depth", ext=False)
    sk.line([(n1, d), (n1 + 3.5 * u, d)], color=INK_SOFT, lw=DIM_LW)
    sk.origin()
    sk.triad((-5.5 * u, Ly - 5 * u))
    return fig


def draw_two_squared(p):
    L = p.L
    Ly = getattr(p, "Ly", L)
    names = ["bottom", "right", "top", "left", "interface"]
    fig, (ax,), scale = make_figure([_pad(0, 2 * L, 0, Ly, 0.35, 0.12, 0.25, 0.12)], width=5.8)
    sk = Sketch(ax, scale, group_colours(names))
    u = sk.u
    sk.fill(rect(0, 0, L, Ly), color=DOMAIN_TINTS[0])
    sk.fill(rect(L, 0, 2 * L, Ly), color=DOMAIN_TINTS[1])
    sk.edge([(0, 0), (2 * L, 0)], "bottom")
    sk.edge([(0, Ly), (2 * L, Ly)], "top")
    sk.edge([(0, 0), (0, Ly)], "left")
    sk.edge([(2 * L, 0), (2 * L, Ly)], "right")
    sk.edge([(L, 0), (L, Ly)], "interface")
    sk.domain("domain:left", (L / 2, Ly * 0.5))
    sk.domain("domain:right", (1.5 * L, Ly * 0.5))
    sk.group("bottom", (L / 2, 1.6 * u), text="bottom:left")
    sk.group("bottom", (1.5 * L, 1.6 * u), text="bottom:right")
    sk.group("top", (L / 2, Ly - 1.6 * u), text="top:left")
    sk.group("top", (1.5 * L, Ly - 1.6 * u), text="top:right")
    sk.group("left", (1.2 * u, Ly * 0.3), ha="left", text="left:left")
    sk.group("right", (2 * L - 1.2 * u, Ly * 0.3), ha="right", text="right:right")
    sk.group("interface", (L, Ly * 0.7))
    sk.dim((0, 0), (L, 0), -3.0, "L")
    sk.dim((L, 0), (2 * L, 0), -3.0, "L")
    sk.dim((0, 0), (0, Ly), 3.0, "Ly (default L)")
    sk.origin()
    sk.triad((-6 * u, Ly - 5 * u))
    return fig


def draw_corrosion_pit(p):
    W, H, R = p.W, p.H, p.R_pit
    names = ["top", "left", "right", "metal:outer", "metal:pit"]
    fig, (ax,), scale = make_figure([_pad(0, W, -R, H, 0.9, 0.4, 0.7, 0.3)])
    sk = Sketch(ax, scale, group_colours(names))
    u = sk.u
    pit = arc((W / 2, 0), R, R, 180, 360, n=60)
    sk.fill(np.vstack([[(0, 0)], pit, [(W, 0), (W, H), (0, H)]]))
    sk.fill(np.vstack([[(0, 0)], pit, [(W, 0), (W, -R - 0.2), (0, -R - 0.2)]]), color="#f6f5f2", z=0.2)
    sk.note("metal (not meshed)", (0.1, -R + 0.05))
    sk.edge([(0, H), (W, H)], "top")
    sk.edge([(0, 0), (0, H)], "left")
    sk.edge([(W, 0), (W, H)], "right")
    sk.edge([(0, 0), (W / 2 - R, 0)], "metal:outer")
    sk.edge([(W / 2 + R, 0), (W, 0)], "metal:outer")
    sk.edge(pit, "metal:pit")
    sk.group("top", (W / 2, H - 1.6 * u))
    sk.group("left", (1.2 * u, H / 2), ha="left")
    sk.group("right", (W - 1.2 * u, H / 2), ha="right")
    sk.group("metal:outer", (W * 0.2, 1.6 * u))
    sk.group("metal:outer", (W * 0.8, 1.6 * u))
    sk.group("metal:pit", (W / 2 + R + 3 * u, -R * 0.75), target=tuple(arc((W / 2, 0), R, R, 300, 300, n=1)[0]), ha="left")
    sk.domain("domain", (W / 2, H * 0.55))
    sk.radius((W / 2, 0), arc((W / 2, 0), R, R, 235, 235, n=1)[0], "R_pit", direction=(-1, 0.2))
    sk.line([(W / 2, 0), (W / 2, -R - 2 * u)], color=INK_SOFT, lw=DIM_LW, ls="-.")
    sk.dim((0, -R), (W / 2, -R), -1.2, "W / 2", ext=False)
    sk.dim((0, H), (W, H), 2.0, "W")
    sk.dim((0, 0), (0, H), 3.0, "H")
    sk.origin()
    sk.triad((-6 * u, H - 5 * u))
    return fig


def draw_plate_with_defect(p):
    W, H, R = p.W, p.H, p.R_defect
    dd = getattr(p, "defect_depth", 0.0)
    (sc, sh), = p.weak_spots
    (vc, vr, vm), = p.voids
    names = ["top", "far", "symmetry", "metal:defect", "metal:outer", "metal:spot1", "metal:void1"]
    fig, (ax,), scale = make_figure([_pad(0, W, -0.85, H, 0.7, 0.4, 0.25, 0.75)])
    sk = Sketch(ax, scale, group_colours(names))
    u = sk.u
    zc = np.sqrt(vr ** 2 - vm ** 2)
    t0 = 180 - np.degrees(np.arctan2(zc, vm))                 # left mouth, seen from the void centre
    t1 = np.degrees(np.arctan2(zc, vm)) + 360                 # right mouth, after passing the bottom
    void = arc((vc, -zc), vr, vr, t0, t1, n=100)
    outline = np.vstack([[(0, -dd), (R, -dd), (R, 0), (vc - vm, 0)], void, [(W, 0), (W, H), (0, H)]])
    sk.fill(outline)
    sk.fill(np.vstack([[(0, -dd), (R, -dd), (R, 0), (vc - vm, 0)], void, [(W, 0), (W, -0.85), (0, -0.85)]]),
            color="#f6f5f2", z=0.2)
    sk.note("metal (not meshed)", (W * 0.9, -0.7), ha="center")
    sk.edge([(0, H), (W, H)], "top")
    sk.edge([(W, 0), (W, H)], "far")
    sk.edge([(0, -dd), (0, H)], "symmetry")
    sk.edge([(0, -dd), (R, -dd), (R, 0)], "metal:defect")
    sk.edge([(R, 0), (sc - sh, 0)], "metal:outer")
    sk.edge([(sc + sh, 0), (vc - vm, 0)], "metal:outer")
    sk.edge([(vc + vm, 0), (W, 0)], "metal:outer")
    sk.edge([(sc - sh, 0), (sc + sh, 0)], "metal:spot1")
    sk.edge(void, "metal:void1")
    sk.group("top", (W / 2, H - 1.6 * u))
    sk.group("far", (W - 1.2 * u, H / 2), ha="right")
    sk.group("symmetry", (1.2 * u, H * 0.65), ha="left")
    sk.group("metal:defect", (R * 0.5, -dd - 2.4 * u), target=(R * 0.5, -dd))
    sk.group("metal:outer", ((sc + sh + vc - vm) / 2, 1.8 * u))
    sk.group("metal:outer", ((vc + vm + W) / 2 + 0.3, 1.8 * u))
    sk.group("metal:spot1", (sc, 4.4 * u), target=(sc - 0.5 * sh, 0))
    sk.group("metal:void1", (vc + vr + 2 * u, -zc - 0.1), target=tuple(arc((vc, -zc), vr, vr, -30, -30, n=1)[0]), ha="left")
    sk.domain("domain", (W * 0.22, H * 0.7))
    sk.dim((0, 0), (R, 0), 1.8, "R_defect", ext=False)
    sk.dim((R, -dd), (R, 0), -1.0, "defect_depth", ext=False, outside=True)
    sk.dim((0, H), (W, H), 2.0, "W")
    sk.dim((W, 0), (W, H), -3.5, "H")
    sk.dim((0, H), (sc, H), 5.2, "weak_spots[0][0]", ext=False, shift=-2)
    sk.line([(sc, 0), (sc, H + 5.6 * u)], color=INK_SOFT, lw=DIM_LW, ls="-.")
    sk.dim((sc, 0), (sc + sh, 0), -1.4, "weak_spots[0][1]", ext=False, shift=4)
    sk.dim((0, H), (vc, H), 8.4, "voids[0][0]", ext=False, shift=4)
    sk.line([(vc, -zc), (vc, H + 8.8 * u)], color=INK_SOFT, lw=DIM_LW, ls="-.")
    sk.line([(0, H + 0.3 * u), (0, H + 8.8 * u)], color=INK_SOFT, lw=DIM_LW)
    rim = arc((vc, -zc), vr, vr, 225, 225, n=1)[0]
    sk._arrow(np.array([vc, -zc]), rim, "-|>")
    sk.leader("voids[0][1]", (vc - vr - 2.5 * u, -zc - 0.3 * vr), tuple(0.5 * (rim + np.array([vc, -zc]))))
    sk.dim((vc, 0), (vc + vm, 0), 1.6, "voids[0][2]", ext=False)
    sk.origin(direction=(-1, 0.4))
    sk.triad((-0.55, -0.6))
    return fig


def draw_plate_with_defect_3d(p):
    W, H, R = p.W, p.H, p.R_defect
    D = depth_vector(W)
    names = ["symmetry", "top", "far", "metal:defect", "metal:outer"]
    xlim = (-1.1, W + D[0] + 1.4)
    ylim = (-0.6, H + D[1] + 0.4)
    fig, (ax,), scale = make_figure([(xlim, ylim)], width=6.5)
    sk = Sketch(ax, scale, group_colours(names))
    prism(sk, rect(0, 0, W, H), D)
    q = arc((0, 0), R, R, 0, 90, n=50)
    qp = np.column_stack([q[:, 0] + q[:, 1] / W * D[0], q[:, 1] / W * D[1]])
    sk.line(qp, color=GHOST, ls="--", lw=0.8)
    sk.group("symmetry", (W * 0.6, H * 0.5), text="symmetry (y = 0)")
    sk.group("symmetry", (-0.2, H + 0.5 * D[1] + 0.15), target=(0.5 * D[0], H * 0.5 + 0.5 * D[1]),
             text="symmetry (x = 0)", ha="right", dashed=True)
    sk.group("top", (W * 0.4 + 0.5 * D[0], H + 0.5 * D[1]), text="top (z = H)")
    sk.group("far", (W + D[0] + 0.15, H * 0.2 + 0.3 * D[1]), target=(W + 0.5 * D[0], H * 0.5 + 0.5 * D[1]),
             text="far (x = W)", ha="left")
    sk.group("far", (W + D[0] + 0.15, H + D[1] + 0.2), target=(W * 0.7 + D[0], H * 0.6 + D[1]),
             text="far (y = W)", ha="left", dashed=True)
    sk.group("metal:defect", (-0.2, -0.35), target=tuple(qp[25] * 0.5), ha="right", dashed=True)
    sk.group("metal:outer", (W + 0.3, -0.35), target=(W * 0.6 + 0.4 * D[0], 0.4 * D[1]), ha="left", dashed=True)
    sk.note("metal: z = 0, defect centred at the origin", (W * 0.5, -0.62), ha="center")
    sk.dim((0, 0), (R, 0), -1.4, "R_defect")
    sk.dim((0, 0), (W, 0), -4.4, "W")
    sk.dim((W, 0), (W + D[0], D[1]), -2.4, "W")
    sk.dim((0, 0), (0, H), 2.4, "H")
    sk.origin(text="(0, 0, 0)", direction=(-1, -1))
    sk.triad((-1.0, 0.05), labels=("x", "y", "z"), dirs=((1, 0), D / np.linalg.norm(D), (0, 1)))
    return fig


# ----------------------------------------------------------------------
# Sample meshes
# ----------------------------------------------------------------------

def _generate(params):
    import femcubes.mesh_generators  # noqa: F401  (registers the generators)
    from femcubes.core.mesh.generator_base import MeshGeneratorBase
    return MeshGeneratorBase.get_generator(params.MeshType).generate(params)


def _group_of(name, level):
    return name.split(":")[0] if level == "top" else name


def _save_png(fig, path, dpi=DPI_MESH, colors=128):
    """Save as an indexed PNG; the palette always holds the group colours exactly."""
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    fixed = [int(c[i:i + 2], 16) for c in SLOTS + DOMAIN_TINTS + ["#ffffff", "#000000"] for i in (1, 3, 5)]
    rest = img.quantize(colors=colors - len(fixed) // 3, method=Image.Quantize.MEDIANCUT).getpalette()
    palette = np.array(fixed + rest[:3 * colors - len(fixed)], dtype=np.int32).reshape(-1, 3)
    pixels = np.asarray(img, dtype=np.int32).reshape(-1, 3)
    unique, inverse = np.unique(pixels, axis=0, return_inverse=True)
    nearest = np.argmin(((unique[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2), axis=1)
    out = Image.fromarray(nearest[inverse.ravel()].reshape(img.size[1], img.size[0]).astype(np.uint8), mode="P")
    out.putpalette(palette.astype(np.uint8).ravel().tolist())
    out.save(path, optimize=True)


def mesh_figure_2d(params, path, level="leaf", colours=None):
    from dolfinx.mesh import entities_to_geometry
    res = _generate(params)
    mesh = res.mesh
    x = mesh.geometry.x[:, :2]
    cells = mesh.geometry.dofmap
    names = {v: k for k, v in res.boundary_markers.items()}
    groups = list(dict.fromkeys(_group_of(names[v], level) for v in sorted(names)))
    colours = group_colours(groups, colours)
    xmin, ymin = x.min(axis=0)
    xmax, ymax = x.max(axis=0)
    w = FIG_WIDTH
    h = min(MAX_HEIGHT, w * (ymax - ymin) / (xmax - xmin))
    w = h * (xmax - xmin) / (ymax - ymin)
    fig = plt.figure(figsize=(max(w, 3.6), h + 0.55))
    ax = fig.add_axes([0.5 * (1 - w / max(w, 3.6)), 0.55 / (h + 0.55), w / max(w, 3.6), h / (h + 0.55)])
    ax.set_aspect("equal")
    ax.axis("off")
    pad = 0.01 * max(xmax - xmin, ymax - ymin)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    handles = []
    dom_names = {v: k for k, v in res.domain_markers.items()}
    if len(dom_names) > 1 and res.cell_tags is not None:
        for i, (tag, name) in enumerate(sorted(dom_names.items())):
            sel = res.cell_tags.indices[res.cell_tags.values == tag]
            ax.add_collection(matplotlib.collections.PolyCollection(x[cells[sel]], fc=DOMAIN_TINTS[i], ec="none", zorder=0))
            handles.append(Patch(fc=DOMAIN_TINTS[i], ec="none", label=name))
    ax.add_collection(matplotlib.collections.PolyCollection(x[cells], fc="none", ec="#8a8984", lw=0.3, zorder=1))
    mesh.topology.create_connectivity(1, 2)
    ft = res.facet_tags
    for g in groups:
        tags = [v for v, n in names.items() if _group_of(n, level) == g]
        facets = ft.indices[np.isin(ft.values, tags)]
        if len(facets) == 0:
            continue
        seg = x[entities_to_geometry(mesh, 1, facets)]
        ax.add_collection(matplotlib.collections.LineCollection(seg, colors=colours[g], lw=2.0, zorder=3,
                                                                capstyle="butt"))
        handles.append(Line2D([], [], color=colours[g], lw=2.4, label=g))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False,
               prop={"family": MONO, "size": FONT_SIZE}, handlelength=1.6, columnspacing=1.2)
    _save_png(fig, path)
    return mesh.topology.index_map(2).size_global


def mesh_figure_3d(params, path, views, up, colours=None):
    import pyvista as pv
    from dolfinx.mesh import entities_to_geometry
    res = _generate(params)
    mesh = res.mesh
    x = mesh.geometry.x
    names = {v: k for k, v in res.boundary_markers.items()}
    groups = [names[v] for v in sorted(names)]
    colours = group_colours(groups, colours)
    ft = res.facet_tags
    mesh.topology.create_connectivity(2, 3)
    images = []
    for view in views:
        plotter = pv.Plotter(off_screen=True, window_size=(640, 520))
        plotter.set_background("white")
        for g in groups:
            facets = ft.indices[ft.values == res.boundary_markers[g]]
            if len(facets) == 0:
                continue
            tri = entities_to_geometry(mesh, 2, facets)
            faces = np.hstack([np.full((len(tri), 1), 3), tri]).ravel()
            plotter.add_mesh(pv.PolyData(x, faces), color=colours[g], show_edges=True, edge_color="#3a3a38",
                             line_width=0.6, lighting=True, ambient=0.65, diffuse=0.35, specular=0.0)
        centre = 0.5 * (x.min(axis=0) + x.max(axis=0))
        size = np.linalg.norm(x.max(axis=0) - x.min(axis=0))
        direction = np.asarray(view, float) / np.linalg.norm(view)
        plotter.camera.position = tuple(centre + 3.0 * size * direction)
        plotter.camera.focal_point = tuple(centre)
        plotter.camera.up = tuple(up)
        plotter.reset_camera()
        plotter.camera.zoom(1.05)
        plotter.add_axes(line_width=2, labels_off=False, color="black")
        images.append(plotter.screenshot(return_img=True))
        plotter.close()
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH / len(images) * 520 / 640 + 0.45))
    for i, img in enumerate(images):
        ax = fig.add_axes([i / len(images), 0.45 / (fig.get_figheight()), 1 / len(images), 1 - 0.45 / fig.get_figheight()])
        ax.imshow(img)
        ax.axis("off")
    handles = [Patch(fc=colours[g], ec="#3a3a38", lw=0.5, label=g) for g in groups]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False,
               prop={"family": MONO, "size": FONT_SIZE}, handlelength=1.2, columnspacing=1.2)
    _save_png(fig, path, dpi=120, colors=200)
    return mesh.topology.index_map(3).size_global


# ----------------------------------------------------------------------
# Cases
# ----------------------------------------------------------------------

V_Z_DEPTH = [(1.0, 0.9, 1.6), (-1.0, -0.9, -1.6)]       # x right, y up, z towards the viewer
V_Z_UP = [(1.0, -1.6, 0.9), (-1.0, 1.6, -0.6)]          # x right, z up, y receding

CASES = {
    "rectangle": (RectangleParams, draw_rectangle, {}),
    "rectangle_3d": (Rectangle3DParams, draw_rectangle_3d, {"views": V_Z_UP, "up": (0, 0, 1)}),
    "rectangle_notches": (RectangleNotchesParams, draw_rectangle_notches, {}),
    "rectangle_notches_3d": (RectangleNotches3DParams, draw_rectangle_notches_3d, {"views": V_Z_UP, "up": (0, 0, 1)}),
    "rectangle_centre_notch": (RectangleCentreNotchParams, draw_rectangle_centre_notch, {}),
    "rectangle_refined_top": (RectangleRefinedTopParams, draw_rectangle_refined_top, {}),
    "plate_with_hole": (PlateWithHoleParams, draw_plate_with_hole, {}),
    "plate_with_hole_3d": (PlateWithHole3DParams, draw_plate_with_hole_3d, {"views": V_Z_DEPTH, "up": (0, 1, 0)}),
    "full_plate_with_hole": (FullPlateWithHoleParams, draw_full_plate_with_hole, {}),
    "full_plate_with_hole_3d": (FullPlateWithHole3DParams, draw_full_plate_with_hole_3d, {"views": V_Z_DEPTH, "up": (0, 1, 0)}),
    "half_plate_with_hole": (HalfPlateWithHoleParams, draw_half_plate_with_hole, {}),
    "single_edge_notched": (SingleEdgeNotchedParams, draw_single_edge_notched, {}),
    "single_edge_notched_3d": (SingleEdgeNotched3DParams, draw_single_edge_notched_3d, {"views": V_Z_DEPTH, "up": (0, 1, 0)}),
    "damage_challenge": (DamageChallengeParams, draw_damage_challenge, {}),
    "damage_challenge_3d": (DamageChallenge3DParams, draw_damage_challenge_3d, {"views": V_Z_DEPTH, "up": (0, 1, 0)}),
    "cylinder": (CylinderParams, draw_cylinder, {"views": [(1.0, -1.6, 0.9), (1.0, -1.6, -0.9)], "up": (0, 0, 1)}),
    "notched_tensile_bar": (NotchedTensileBarParams, draw_notched_tensile_bar, {"level": "top"}),
    "two_squared": (TwoSquaredParams, draw_two_squared, {"level": "top"}),
    "corrosion_pit": (CorrosionPitParams, draw_corrosion_pit, {}),
    "plate_with_defect": (PlateWithDefectParams, draw_plate_with_defect, {}),
    "plate_with_defect_3d": (PlateWithDefect3DParams, draw_plate_with_defect_3d, {"views": [(1.0, -1.6, 0.9), (-1.0, 1.6, -0.9)], "up": (0, 0, 1)}),
}


def _content_bbox(fig, pad=4.0):
    """Bounding box (inches) of everything drawn, without the empty axes area."""
    from matplotlib.transforms import Bbox
    renderer = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        for artist in ax.get_children():
            if artist is ax.patch or not artist.get_visible() or artist in ax.spines.values():
                continue
            try:
                box = artist.get_window_extent(renderer)
            except (TypeError, NotImplementedError, RuntimeError):
                continue
            if box.width > 0 or box.height > 0:
                boxes.append(box)
        if ax.title.get_text():
            boxes.append(ax.title.get_window_extent(renderer))
    box = Bbox.union(boxes).padded(pad)
    return box.transformed(fig.dpi_scale_trans.inverted())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cases", nargs="*", help=f"cases to make (default: all): {', '.join(CASES)}")
    parser.add_argument("--preview", help="also write PNG copies of the drawings to this folder")
    parser.add_argument("--no-mesh", action="store_true", help="only make the drawings")
    args = parser.parse_args(argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.cases or CASES:
        params_cls, draw, options = CASES[name]
        params = params_cls()
        fig = draw(params)
        colours = getattr(fig, "group_colours", None)
        fig.canvas.draw()
        box = _content_bbox(fig)
        fig.savefig(OUTPUT_DIR / f"{name}_dimensions.svg", metadata={"Date": None}, bbox_inches=box)
        if args.preview:
            fig.savefig(os.path.join(args.preview, f"{name}_dimensions.png"), dpi=110, facecolor="white", bbox_inches=box)
        plt.close(fig)
        if args.no_mesh:
            print(f"{name}: drawing")
            continue
        path = OUTPUT_DIR / f"{name}_mesh.png"
        if params.dim == 2:
            n = mesh_figure_2d(params, path, level=options.get("level", "leaf"), colours=colours)
        else:
            n = mesh_figure_3d(params, path, options["views"], options["up"], colours=colours)
        print(f"{name}: drawing, mesh with {n} cells, {path.stat().st_size / 1024:.0f} kB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
