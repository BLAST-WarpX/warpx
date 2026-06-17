#!/usr/bin/env python3
"""
Interactive matplotlib plotting of WarpX 3D Cartesian (x, y, z) plotfiles.

3D-Cartesian port of warpx_rtz_plotter.py. yt is used ONLY to read the data
(load each plotfile and pull a uniform covering grid into numpy); all plotting is
done directly with matplotlib on the Qt5Agg backend. openPMD-viewer-style: build
a time series from a set of plotfile directories, then slice/plot a field at a
chosen iteration.

A slice is taken perpendicular to one axis ("normal") at a chosen coordinate; the
remaining two axes are plotted. By convention z is placed on the horizontal axis
when present (so the y=0 cut gives the familiar x-z plane), and the z-normal cut
gives the x-y plane.

Example
-------
    from warpx_cart_plotter import WarpXCartSeries

    ts = WarpXCartSeries("diags", prefix="plt_field_diags")
    print(ts.iterations)          # available step numbers
    print(ts.avail_fields)        # field names (Bx, By, Bz, Ex, ..., rho, ...)

    # Bz on the y = 0 cut (the x-z plane), latest iteration:
    ts.plot_slice("Bz", normal="y", coord=0.0)
    ts.show()                     # blocks in an interactive Qt window

    # convenience wrappers + a specific iteration / fixed range:
    ts.plot_xz("Bz", y=0.0, iteration=2000, vmin=-1.5, vmax=1.5)   # x-z plane
    ts.plot_xy("Bz", z=4.0)                                        # x-y plane
    ts.show()
"""

import glob
import os
import re
from dataclasses import dataclass

import matplotlib

# Interactive Qt window by default. Respect an explicit MPLBACKEND override
# (e.g. headless "Agg" for movie rendering) and never fail at import if Qt is
# unavailable. Set the backend before importing pyplot.
if not os.environ.get("MPLBACKEND"):
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        pass
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
@dataclass
class SliceInfo:
    """Coordinate metadata for a 2D slice (openPMD-viewer-ish)."""
    horiz_name: str       # e.g. "z"
    vert_name: str        # e.g. "x"
    horiz_edges: np.ndarray
    vert_edges: np.ndarray
    extent: tuple         # (h_min, h_max, v_min, v_max)
    slice_name: str       # axis sliced across, e.g. "y"
    slice_value: float    # actual coordinate of the slice
    time: float           # simulation time [s]
    iteration: int        # step number


# For each slice-normal axis: (horizontal axis, vertical axis, transpose?).
# The plane array from arr[...] has its axes in (x, y, z) order with the normal
# removed; transpose only when that order does not already equal (vert, horiz).
_PLANE = {
    "x": ("z", "y", False),  # plane axes (y, z) -> rows=y(vert), cols=z(horiz)
    "y": ("z", "x", False),  # plane axes (x, z) -> rows=x(vert), cols=z(horiz)
    "z": ("x", "y", True),   # plane axes (x, y) -> transpose to rows=y(vert), cols=x(horiz)
}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class WarpXCartSeries:
    """A time series of WarpX 3D Cartesian plotfiles, plotted with matplotlib."""

    def __init__(self, source, prefix="plt"):
        """
        Parameters
        ----------
        source : str | list[str]
            One of:
              * a list of plotfile directory paths,
              * a glob pattern (e.g. "diags/plt_field_diags*"),
              * a parent directory containing plotfiles named ``<prefix>*``.
        prefix : str
            Plotfile name prefix used when ``source`` is a parent directory.
        """
        self.paths = self._discover(source, prefix)
        if not self.paths:
            raise FileNotFoundError(f"No plotfiles found for source={source!r}, prefix={prefix!r}")
        self.iterations = np.array([self._step_of(p) for p in self.paths], dtype=np.int64)
        self._ds_cache = {}
        self._grid_cache = {}
        self._avail = None
        self._eb_func = None      # cached callable f(x, y, z) for the EB wall
        self._eb_inputs = None    # path of the inputs file the EB came from

    # ----------------------------- discovery -----------------------------
    @staticmethod
    def _step_of(path):
        m = re.search(r"(\d+)\s*$", os.path.basename(os.path.normpath(path)))
        return int(m.group(1)) if m else -1

    def _discover(self, source, prefix):
        if isinstance(source, (list, tuple)):
            cand = list(source)
        elif any(ch in source for ch in "*?["):
            cand = glob.glob(source)
        elif os.path.isdir(source) and glob.glob(os.path.join(source, "Header")):
            cand = [source]
        else:
            cand = glob.glob(os.path.join(source, prefix + "*"))
        cand = [p for p in cand if os.path.isdir(p) and os.path.exists(os.path.join(p, "Header"))]
        return sorted(cand, key=self._step_of)

    # ----------------------------- indexing ------------------------------
    def _index_for(self, iteration=None, index=-1):
        if iteration is not None:
            hits = np.nonzero(self.iterations == int(iteration))[0]
            if len(hits) == 0:
                raise ValueError(
                    f"iteration {iteration} not found. Available: {self.iterations.tolist()}"
                )
            return int(hits[0])
        return int(np.arange(len(self.paths))[index])

    # ----------------------------- yt access -----------------------------
    def _ds(self, idx):
        if idx not in self._ds_cache:
            import yt
            yt.set_log_level(50)
            self._ds_cache[idx] = yt.load(self.paths[idx])
        return self._ds_cache[idx]

    def _covering_grid(self, idx):
        if idx not in self._grid_cache:
            ds = self._ds(idx)
            self._grid_cache[idx] = ds.covering_grid(
                level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
            )
        return self._grid_cache[idx]

    @property
    def avail_fields(self):
        """Sorted list of plottable field names (the short names, e.g. 'Bz')."""
        if self._avail is None:
            ds = self._ds(0)
            self._avail = sorted({f[1] for f in ds.field_list})
        return self._avail

    def _resolve_field(self, ds, field):
        if isinstance(field, tuple):
            return field
        for ftype, fname in ds.field_list:
            if fname == field:
                return (ftype, fname)
        raise KeyError(f"field {field!r} not in {sorted(f[1] for f in ds.field_list)}")

    # ----------------------------- data ----------------------------------
    def axes(self, idx=0):
        """Cell-center coordinate arrays (x, y, z) and their edges."""
        ds = self._ds(idx)
        nx, ny, nz = (int(n) for n in ds.domain_dimensions)
        le = np.array(ds.domain_left_edge.to_value())
        re = np.array(ds.domain_right_edge.to_value())
        x_e = np.linspace(le[0], re[0], nx + 1)
        y_e = np.linspace(le[1], re[1], ny + 1)
        z_e = np.linspace(le[2], re[2], nz + 1)
        def centers(e):
            return 0.5 * (e[:-1] + e[1:])
        return {
            "x": centers(x_e), "y": centers(y_e), "z": centers(z_e),
            "x_edges": x_e, "y_edges": y_e, "z_edges": z_e,
            "shape": (nx, ny, nz),
        }

    def get_field(self, field, iteration=None, index=-1):
        """Return the full 3D array (nx, ny, nz) and a coords dict."""
        idx = self._index_for(iteration, index)
        ds = self._ds(idx)
        cg = self._covering_grid(idx)
        fld = self._resolve_field(ds, field)
        arr = np.asarray(cg[fld])  # (nx, ny, nz)
        ax = self.axes(idx)
        ax["time"] = float(ds.current_time.to_value())
        ax["iteration"] = int(self.iterations[idx])
        return arr, ax

    # ----------------------------- slicing -------------------------------
    def get_slice(self, field, normal="y", coord=0.0, iteration=None, index=-1):
        """2D slice perpendicular to ``normal`` at the cell nearest ``coord``.

        Returns (F2d, SliceInfo). ``coord=None`` selects the domain mid-plane.
        """
        if normal not in _PLANE:
            raise ValueError(f"normal must be one of 'x','y','z', got {normal!r}")
        arr, ax = self.get_field(field, iteration, index)

        nrm = _AXIS_INDEX[normal]
        cc = ax[normal]
        if coord is None:
            coord = 0.5 * (ax[normal + "_edges"][0] + ax[normal + "_edges"][-1])
        i = int(np.argmin(np.abs(cc - coord)))

        sl = [slice(None)] * 3
        sl[nrm] = i
        plane = arr[tuple(sl)]  # axes are the two non-normal axes, in (x,y,z) order

        h_name, v_name, transpose = _PLANE[normal]
        F = plane.T if transpose else plane  # -> (vert, horiz)
        info = SliceInfo(
            horiz_name=h_name, vert_name=v_name,
            horiz_edges=ax[h_name + "_edges"], vert_edges=ax[v_name + "_edges"],
            extent=(ax[h_name + "_edges"][0], ax[h_name + "_edges"][-1],
                    ax[v_name + "_edges"][0], ax[v_name + "_edges"][-1]),
            slice_name=normal, slice_value=float(cc[i]),
            time=ax["time"], iteration=ax["iteration"],
        )
        return F, info

    # --------------------------- EB wall ---------------------------------
    def _find_inputs(self):
        """Locate a WarpX inputs file holding ``warpx.eb_implicit_function``.

        Searches: the cwd, then each plotfile's parent directory, for a file
        named ``warpx_used_inputs`` (or any ``*used_inputs*``) that contains an
        ``eb_implicit_function`` entry.
        """
        cands = ["warpx_used_inputs"]
        seen = set()
        for p in self.paths:
            parent = os.path.dirname(os.path.normpath(p))
            for d in (parent, os.path.dirname(parent)):
                if d and d not in seen:
                    seen.add(d)
                    cands.append(os.path.join(d, "warpx_used_inputs"))
                    cands.extend(sorted(glob.glob(os.path.join(d, "*used_inputs*"))))
        for c in cands:
            if os.path.isfile(c):
                with open(c) as fh:
                    if "eb_implicit_function" in fh.read():
                        return c
        return None

    def _load_eb_function(self, inputs_file=None):
        """Parse ``warpx.eb_implicit_function`` into a numpy-vectorized callable.

        The implicit function is a ';'-separated list of ``name=expr``
        assignments followed by the body expression, written in the AMReX
        parser dialect (``min``/``max``/``sqrt``/arithmetic). The EB wall is the
        zero level set of the body. Returns ``f(x, y, z)`` operating on arrays.
        """
        if inputs_file is None and self._eb_func is not None:
            return self._eb_func

        path = inputs_file or self._find_inputs()
        if path is None or not os.path.isfile(path):
            raise FileNotFoundError(
                "Could not find a WarpX inputs file with 'eb_implicit_function'. "
                "Pass eb_inputs=<path> (e.g. eb_inputs='warpx_used_inputs')."
            )
        with open(path) as fh:
            text = fh.read()
        m = re.search(r'eb_implicit_function\s*=\s*"([^"]*)"', text)
        if m is None:
            m = re.search(r"eb_implicit_function\s*=\s*(.+)", text)
        if m is None:
            raise ValueError(f"No 'eb_implicit_function' found in {path!r}")
        expr = m.group(1).strip().strip('"')

        # AMReX parser dialect -> numpy. min/max are 2-arg, element-wise.
        env = {
            "__builtins__": {},
            "min": np.minimum, "max": np.maximum,
            "sqrt": np.sqrt, "abs": np.abs, "exp": np.exp,
            "log": np.log, "log10": np.log10, "pow": np.power,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
            "atan2": np.arctan2, "sinh": np.sinh, "cosh": np.cosh,
            "tanh": np.tanh, "floor": np.floor, "ceil": np.ceil,
            "fmod": np.fmod, "pi": np.pi,
        }

        # Resolve `my_constants.<name> = <expr>` declared elsewhere in the inputs
        # so the implicit function can reference them (e.g. R_w). WarpX evaluates
        # these in the same parser dialect and they may reference one another, so
        # resolve by fixed point; trailing "# value" echoes are stripped and any
        # unresolved (runtime-only) symbols are skipped rather than fatal.
        constants = {}
        decls = re.findall(r"^\s*my_constants\.(\w+)\s*=\s*(.+?)\s*$",
                           text, re.MULTILINE)
        pending = [(n, v.split("#", 1)[0].strip().strip('"')) for n, v in decls]
        for _ in range(len(pending) + 1):
            if not pending:
                break
            still, progressed = [], False
            for cname, cexpr in pending:
                try:
                    constants[cname] = eval(cexpr, env, dict(constants))  # noqa: S307
                    progressed = True
                except Exception:
                    still.append((cname, cexpr))
            pending = still
            if not progressed:
                break

        stmts = [s.strip() for s in expr.split(";") if s.strip()]
        if not stmts:
            raise ValueError(f"Empty eb_implicit_function in {path!r}")
        assigns, body = stmts[:-1], stmts[-1]

        def f(x, y, z):
            local = dict(constants)
            local.update({"x": x, "y": y, "z": z})
            for s in assigns:
                name, val = s.split("=", 1)
                local[name.strip()] = eval(val, env, local)  # noqa: S307
            return eval(body, env, local)  # noqa: S307

        if inputs_file is None:
            self._eb_func = f
            self._eb_inputs = path
        return f

    def _plane_coords(self, info):
        """Full (x, y, z) meshgrids over a slice plane, shaped like the field
        array (vert, horiz). The slice-normal axis is held at its plane value."""
        hc = 0.5 * (info.horiz_edges[:-1] + info.horiz_edges[1:])
        vc = 0.5 * (info.vert_edges[:-1] + info.vert_edges[1:])
        Hg, Vg = np.meshgrid(hc, vc)  # (nv, nh)
        coords = {info.horiz_name: Hg, info.vert_name: Vg,
                  info.slice_name: np.full_like(Hg, info.slice_value)}
        return hc, vc, coords["x"], coords["y"], coords["z"]

    def overlay_eb(self, ax, info, inputs_file=None, **contour_kw):
        """Draw the EB wall (zero contour of the implicit function) on ``ax``."""
        f = self._load_eb_function(inputs_file)
        hc, vc, X, Y, Z = self._plane_coords(info)
        feb = np.asarray(f(X, Y, Z), dtype=float)
        if not np.isfinite(feb).any() or feb.min() > 0 or feb.max() < 0:
            return  # the wall does not cross this plane
        kw = dict(levels=[0.0], colors="lime", linewidths=1.5)
        kw.update(contour_kw)
        ax.contour(hc, vc, feb, **kw)

    # ----------------------------- plotting ------------------------------
    @staticmethod
    def _symmetric_norm(F):
        a = np.nanmax(np.abs(F)) if np.isfinite(F).any() else 1.0
        return (-a, a) if a > 0 else (-1.0, 1.0)

    def plot_slice(self, field, normal="y", coord=0.0, iteration=None, index=-1,
                   ax=None, cmap="RdBu_r", vmin=None, vmax=None,
                   symmetric=True, equal_aspect=False,
                   plot_EB=False, eb_inputs=None, eb_kwargs=None, **pcm_kw):
        """pcolormesh of ``field`` on the plane perpendicular to ``normal``.

        If ``plot_EB`` is True, overlay the embedded-boundary wall as the zero
        contour of ``warpx.eb_implicit_function`` (read from ``eb_inputs`` or an
        auto-discovered ``warpx_used_inputs``). ``eb_kwargs`` is passed to the
        underlying ``ax.contour`` call (e.g. ``{"colors": "k"}``).
        """
        F, info = self.get_slice(field, normal, coord, iteration, index)
        if ax is None:
            # wide for z-containing planes, square-ish for the x-y plane
            figsize = (6, 6) if "z" not in (info.horiz_name, info.vert_name) else (9, 4)
            _, ax = plt.subplots(figsize=figsize)
        if vmin is None and vmax is None and symmetric:
            vmin, vmax = self._symmetric_norm(F)
        m = ax.pcolormesh(info.horiz_edges, info.vert_edges, F,
                          cmap=cmap, vmin=vmin, vmax=vmax, shading="flat", **pcm_kw)
        if plot_EB:
            self.overlay_eb(ax, info, inputs_file=eb_inputs, **(eb_kwargs or {}))
        ax.set_xlabel(f"{info.horiz_name} [m]")
        ax.set_ylabel(f"{info.vert_name} [m]")
        ax.set_title(f"{self._fname(field)}  {info.slice_name}={info.slice_value:.3f} m  "
                     f"t={info.time:.3e} s  (it {info.iteration})")
        if equal_aspect:
            ax.set_aspect("equal")
        ax.figure.colorbar(m, ax=ax, label=self._fname(field))
        return ax

    # convenience wrappers -------------------------------------------------
    def plot_xz(self, field, y=0.0, **kw):
        """x-z plane at y ~ ``y`` (slice normal to y)."""
        return self.plot_slice(field, normal="y", coord=y, **kw)

    def plot_yz(self, field, x=0.0, **kw):
        """y-z plane at x ~ ``x`` (slice normal to x)."""
        return self.plot_slice(field, normal="x", coord=x, **kw)

    def plot_xy(self, field, z=None, **kw):
        """x-y plane at z ~ ``z`` (default: mid-plane; slice normal to z)."""
        kw.setdefault("equal_aspect", True)
        return self.plot_slice(field, normal="z", coord=z, **kw)

    @staticmethod
    def _fname(field):
        return field[1] if isinstance(field, tuple) else field

    @staticmethod
    def show(block=True):
        plt.show(block=block)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Plot a WarpX 3D Cartesian field slice with matplotlib")
    ap.add_argument("source", help="plotfile dir, parent dir, or glob (e.g. diags/plt_field_diags*)")
    ap.add_argument("--prefix", default="plt_field_diags")
    ap.add_argument("--field", default="Bz")
    ap.add_argument("--normal", default="y", choices=["x", "y", "z"])
    ap.add_argument("--coord", type=float, default=0.0, help="coordinate of the slice plane [m]")
    ap.add_argument("--iteration", type=int, default=None)
    ap.add_argument("--index", type=int, default=-1)
    ap.add_argument("--plot-EB", action="store_true",
                    help="overlay the EB wall (zero contour of eb_implicit_function)")
    ap.add_argument("--eb-inputs", default=None,
                    help="path to the WarpX inputs file (default: auto-discover warpx_used_inputs)")
    args = ap.parse_args()

    ts = WarpXCartSeries(args.source, prefix=args.prefix)
    print("iterations:", ts.iterations.tolist())
    print("fields    :", ts.avail_fields)
    ts.plot_slice(args.field, normal=args.normal, coord=args.coord,
                  iteration=args.iteration, index=args.index,
                  plot_EB=args.plot_EB, eb_inputs=args.eb_inputs)
    ts.show()
