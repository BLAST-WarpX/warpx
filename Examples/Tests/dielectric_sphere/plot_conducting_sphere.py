#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from conducting_sphere_solution import (
    ProblemParameters,
    analytic_fields,
    analytic_solution,
    load_plotfile,
)
from matplotlib.patches import Circle

PLOT_PREFIX = "conducting_sphere_dielectric_shell_stl"
DPI = 180


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot x-z slices of the conducting-sphere dielectric-shell STL test."
    )
    parser.add_argument("plotfile", help="WarpX plotfile to read")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for plots; default is <run-dir>/plots",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="suffix for output filenames; default is the x-cell count",
    )
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=(-2.0, 2.0),
        metavar=("XMIN", "XMAX"),
        help="x limits for slice plots and centerline",
    )
    parser.add_argument(
        "--zlim",
        nargs=2,
        type=float,
        default=(-2.0, 2.0),
        metavar=("ZMIN", "ZMAX"),
        help="z limits for x-z slice plots",
    )
    parser.add_argument(
        "--slice-y",
        type=float,
        default=0.0,
        help="physical y location for x-z slice plots",
    )
    parser.add_argument(
        "--log-error-floor",
        type=float,
        default=1.0e-12,
        help="floor applied before log10 field-error plots",
    )
    parser.add_argument(
        "--centerline-y",
        type=float,
        default=0.0,
        help="physical y location for centerline plots",
    )
    parser.add_argument(
        "--centerline-z",
        type=float,
        default=0.0,
        help="physical z location for centerline plots",
    )
    return parser.parse_args()


def default_output_dir(plotfile):
    path = Path(plotfile)
    if path.parent.name == "diags":
        return path.parent.parent / "plots"
    return Path("plots")


def filename_suffix(args, data):
    if args.suffix is None:
        return f"_{data.dims[0]}"
    if args.suffix == "":
        return ""
    if args.suffix.startswith("_"):
        return args.suffix
    return f"_{args.suffix}"


def symmetric_limit(*values):
    max_abs = 0.0
    for value in values:
        finite = np.asarray(value)[np.isfinite(value)]
        if finite.size > 0:
            max_abs = max(max_abs, float(np.max(np.abs(finite))))
    return max_abs if max_abs > 0.0 else 1.0


def interpolate_axis(values, coords, location, axis):
    if location < coords[0] or location > coords[-1]:
        raise ValueError(
            f"slice location {location} is outside the cell-centered range "
            f"[{coords[0]}, {coords[-1]}]"
        )

    upper = int(np.searchsorted(coords, location, side="left"))
    if upper < coords.size and np.isclose(location, coords[upper]):
        return np.take(values, upper, axis=axis)

    lower = upper - 1
    weight = (location - coords[lower]) / (coords[upper] - coords[lower])
    return (1.0 - weight) * np.take(values, lower, axis=axis) + weight * np.take(
        values, upper, axis=axis
    )


def xz_slice(data, values, y_slice):
    return interpolate_axis(values, data.y, y_slice, axis=1)


def centerline(data, values, y_line, z_line):
    values_xz = interpolate_axis(values, data.y, y_line, axis=1)
    return interpolate_axis(values_xz, data.z, z_line, axis=1)


def zoom_mask(data, xlim, zlim):
    in_x = (data.x >= xlim[0]) & (data.x <= xlim[1])
    in_z = (data.z >= zlim[0]) & (data.z <= zlim[1])
    return in_x[:, np.newaxis] & in_z[np.newaxis, :]


def sphere_slice_radius(radius, y_slice):
    radius2 = radius**2 - y_slice**2
    if radius2 <= 0.0:
        return None
    return np.sqrt(radius2)


def add_geometry(ax, params, y_slice):
    conductor_radius = sphere_slice_radius(params.conductor_radius, y_slice)
    dielectric_radius = sphere_slice_radius(params.dielectric_radius, y_slice)

    if conductor_radius is not None:
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                conductor_radius,
                fill=False,
                color="black",
                linewidth=1.3,
            )
        )
    if dielectric_radius is not None:
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                dielectric_radius,
                fill=False,
                color="black",
                linestyle="--",
                linewidth=1.2,
            )
        )


def plot_image(
    fig,
    ax,
    data,
    values,
    title,
    xlim,
    zlim,
    y_slice,
    params,
    cmap,
    norm=None,
    vmin=None,
    vmax=None,
    cbar_label=None,
    add_colorbar=True,
):
    extent = [
        data.left_edge[0],
        data.right_edge[0],
        data.left_edge[2],
        data.right_edge[2],
    ]
    image = ax.imshow(
        values.T,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )
    add_geometry(ax, params, y_slice)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    if add_colorbar:
        fig.colorbar(image, ax=ax, label=cbar_label)
    return image


def save_figure(fig, output_dir, stem, suffix, args):
    path = output_dir / f"{stem}_xz_y{args.slice_y:g}{suffix}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path)


def plot_phi(data, exact, output_dir, suffix, args, params):
    phi = xz_slice(data, data.phi, args.slice_y)
    phi_exact = xz_slice(data, exact.phi, args.slice_y)
    phi_error = phi - phi_exact
    mask = zoom_mask(data, args.xlim, args.zlim)
    phi_limit = symmetric_limit(phi[mask], phi_exact[mask])
    error_limit = symmetric_limit(phi_error[mask])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    plot_image(
        fig,
        axes[0],
        data,
        phi,
        r"WarpX $\phi$",
        args.xlim,
        args.zlim,
        args.slice_y,
        params,
        "RdBu_r",
        vmin=-phi_limit,
        vmax=phi_limit,
    )
    plot_image(
        fig,
        axes[1],
        data,
        phi_exact,
        r"analytic $\phi$",
        args.xlim,
        args.zlim,
        args.slice_y,
        params,
        "RdBu_r",
        vmin=-phi_limit,
        vmax=phi_limit,
    )
    plot_image(
        fig,
        axes[2],
        data,
        phi_error,
        r"$\phi-\phi_{\rm analytic}$",
        args.xlim,
        args.zlim,
        args.slice_y,
        params,
        "RdBu_r",
        vmin=-error_limit,
        vmax=error_limit,
    )
    save_figure(fig, output_dir, f"{PLOT_PREFIX}_phi", suffix, args)


def plot_e_components(data, exact, output_dir, suffix, args, params):
    mask = zoom_mask(data, args.xlim, args.zlim)
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11.5), constrained_layout=True)

    for row, name, sim, analytic in (
        (0, "Ex", data.ex, exact.ex),
        (1, "Ey", data.ey, exact.ey),
        (2, "Ez", data.ez, exact.ez),
    ):
        sim_slice = xz_slice(data, sim, args.slice_y)
        analytic_slice = xz_slice(data, analytic, args.slice_y)
        error = sim_slice - analytic_slice
        field_limit = symmetric_limit(sim_slice[mask], analytic_slice[mask])
        error_limit = symmetric_limit(error[mask])
        plot_image(
            fig,
            axes[row, 0],
            data,
            sim_slice,
            f"WarpX {name}",
            args.xlim,
            args.zlim,
            args.slice_y,
            params,
            "RdBu_r",
            vmin=-field_limit,
            vmax=field_limit,
        )
        plot_image(
            fig,
            axes[row, 1],
            data,
            analytic_slice,
            f"analytic {name}",
            args.xlim,
            args.zlim,
            args.slice_y,
            params,
            "RdBu_r",
            vmin=-field_limit,
            vmax=field_limit,
        )
        plot_image(
            fig,
            axes[row, 2],
            data,
            error,
            f"{name} error",
            args.xlim,
            args.zlim,
            args.slice_y,
            params,
            "RdBu_r",
            vmin=-error_limit,
            vmax=error_limit,
        )

    save_figure(fig, output_dir, f"{PLOT_PREFIX}_e_components", suffix, args)


def plot_e_magnitude_error(data, exact, output_dir, suffix, args, params):
    mask = zoom_mask(data, args.xlim, args.zlim)
    ex = xz_slice(data, data.ex, args.slice_y)
    ey = xz_slice(data, data.ey, args.slice_y)
    ez = xz_slice(data, data.ez, args.slice_y)
    ex_exact = xz_slice(data, exact.ex, args.slice_y)
    ey_exact = xz_slice(data, exact.ey, args.slice_y)
    ez_exact = xz_slice(data, exact.ez, args.slice_y)

    e_mag = np.sqrt(ex**2 + ey**2 + ez**2)
    e_exact_mag = np.sqrt(ex_exact**2 + ey_exact**2 + ez_exact**2)
    e_error_mag = np.sqrt(
        (ex - ex_exact) ** 2 + (ey - ey_exact) ** 2 + (ez - ez_exact) ** 2
    )
    log_error = np.log10(np.maximum(e_error_mag, args.log_error_floor))
    e_limit = max(
        float(np.max(e_mag[mask])),
        float(np.max(e_exact_mag[mask])),
        args.log_error_floor,
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    plot_image(
        fig,
        axes[0],
        data,
        e_mag,
        r"WarpX $|E|$",
        args.xlim,
        args.zlim,
        args.slice_y,
        params,
        "magma",
        vmin=0.0,
        vmax=e_limit,
    )
    plot_image(
        fig,
        axes[1],
        data,
        e_exact_mag,
        r"analytic $|E|$",
        args.xlim,
        args.zlim,
        args.slice_y,
        params,
        "magma",
        vmin=0.0,
        vmax=e_limit,
    )
    plot_image(
        fig,
        axes[2],
        data,
        log_error,
        r"$\log_{10}(|\Delta E|)$",
        args.xlim,
        args.zlim,
        args.slice_y,
        params,
        "viridis",
        cbar_label=r"$\log_{10}(|\Delta E|)$",
    )
    save_figure(fig, output_dir, f"{PLOT_PREFIX}_e_magnitude_error", suffix, args)


def plot_materials(data, output_dir, suffix, args, params):
    epsilon = xz_slice(data, data.epsilon, args.slice_y)
    eb_covered = xz_slice(data, data.eb_covered, args.slice_y)

    region = np.zeros_like(epsilon)
    region[epsilon > 1.0 + 1.0e-12] = 2.0
    region[eb_covered > 0.5] = 1.0

    cmap = mcolors.ListedColormap(["#d9d9d9", "#4d4d4d", "#2b8cbe"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.0), constrained_layout=True)
    image = plot_image(
        fig,
        ax,
        data,
        region,
        "material regions",
        args.xlim,
        args.zlim,
        args.slice_y,
        params,
        cmap,
        norm=norm,
        add_colorbar=False,
    )
    cbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["vacuum", "conductor", "dielectric"])
    save_figure(fig, output_dir, f"{PLOT_PREFIX}_materials", suffix, args)


def sphere_chord_half_width(radius, y_line, z_line):
    radius2 = radius**2 - y_line**2 - z_line**2
    if radius2 <= 0.0:
        return None
    return np.sqrt(radius2)


def add_centerline_regions(ax, y_line, z_line, params, add_labels):
    conductor_half_width = sphere_chord_half_width(
        params.conductor_radius, y_line, z_line
    )
    dielectric_half_width = sphere_chord_half_width(
        params.dielectric_radius, y_line, z_line
    )

    if dielectric_half_width is None:
        return

    conductor_label = "conductor" if add_labels else None
    dielectric_label = "dielectric" if add_labels else None

    if conductor_half_width is None:
        ax.axvspan(
            -dielectric_half_width,
            dielectric_half_width,
            color="#d8f0f3",
            alpha=0.8,
            label=dielectric_label,
        )
        return

    ax.axvspan(
        -dielectric_half_width,
        -conductor_half_width,
        color="#d8f0f3",
        alpha=0.8,
        label=dielectric_label,
    )
    ax.axvspan(
        conductor_half_width,
        dielectric_half_width,
        color="#d8f0f3",
        alpha=0.8,
    )
    ax.axvspan(
        -conductor_half_width,
        conductor_half_width,
        color="0.85",
        label=conductor_label,
    )


def save_centerline_figure(fig, output_dir, suffix, args):
    path = output_dir / (
        f"{PLOT_PREFIX}_centerline_y{args.centerline_y:g}_"
        f"z{args.centerline_z:g}{suffix}.png"
    )
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path)


def plot_centerline(data, output_dir, suffix, args, params):
    y = np.full_like(data.x, args.centerline_y)
    z = np.full_like(data.x, args.centerline_z)
    phi_exact, ex_exact, ey_exact, ez_exact = analytic_fields(
        data.x,
        y,
        z,
        params.conductor_radius,
        params.dielectric_radius,
        params.epsilon_r,
        params.applied_field,
    )

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 11.5), sharex=True)
    curves = (
        (
            r"$\phi$",
            centerline(data, data.phi, args.centerline_y, args.centerline_z),
            phi_exact,
        ),
        (
            "Ex",
            centerline(data, data.ex, args.centerline_y, args.centerline_z),
            ex_exact,
        ),
        (
            "Ey",
            centerline(data, data.ey, args.centerline_y, args.centerline_z),
            ey_exact,
        ),
        (
            "Ez",
            centerline(data, data.ez, args.centerline_y, args.centerline_z),
            ez_exact,
        ),
    )

    for ax, (label, sim, analytic) in zip(axes, curves):
        add_centerline_regions(
            ax,
            args.centerline_y,
            args.centerline_z,
            params,
            ax is axes[0],
        )
        ax.plot(data.x, sim, color="#1f77b4", linewidth=1.6, label="WarpX")
        ax.plot(
            data.x,
            analytic,
            color="black",
            linestyle="--",
            linewidth=1.3,
            label="analytic",
        )
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="best", ncol=2)
    axes[-1].set_xlabel("x")
    axes[-1].set_xlim(*args.xlim)
    fig.suptitle(
        f"centerline at y = {args.centerline_y:.6g}, z = {args.centerline_z:.6g}"
    )
    fig.tight_layout()
    save_centerline_figure(fig, output_dir, suffix, args)


def main():
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.plotfile)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = ProblemParameters()
    data = load_plotfile(args.plotfile)
    exact = analytic_solution(data, params)
    suffix = filename_suffix(args, data)

    plot_phi(data, exact, output_dir, suffix, args, params)
    plot_e_components(data, exact, output_dir, suffix, args, params)
    plot_e_magnitude_error(data, exact, output_dir, suffix, args, params)
    plot_centerline(data, output_dir, suffix, args, params)
    plot_materials(data, output_dir, suffix, args, params)


if __name__ == "__main__":
    main()
