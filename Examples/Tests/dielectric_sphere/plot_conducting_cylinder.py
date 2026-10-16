#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from conducting_cylinder_solution import (
    ProblemParameters,
    analytic_fields,
    analytic_solution,
    load_plotfile,
)
from matplotlib.patches import Circle

PLOT_PREFIX = "conducting_cylinder_dielectric_shell"
DPI = 180


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the conducting-cylinder dielectric-shell test."
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
        help="x limits for 2D plots and centerline",
    )
    parser.add_argument(
        "--zlim",
        nargs=2,
        type=float,
        default=(-2.0, 2.0),
        metavar=("ZMIN", "ZMAX"),
        help="z limits for 2D plots",
    )
    parser.add_argument(
        "--log-error-floor",
        type=float,
        default=1.0e-12,
        help="floor applied before log10 field-error plots",
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


def zoom_mask(data, xlim, zlim):
    return (
        (data.xx >= xlim[0])
        & (data.xx <= xlim[1])
        & (data.zz >= zlim[0])
        & (data.zz <= zlim[1])
    )


def symmetric_limit(*values):
    max_abs = 0.0
    for value in values:
        finite = np.asarray(value)[np.isfinite(value)]
        if finite.size > 0:
            max_abs = max(max_abs, float(np.max(np.abs(finite))))
    return max_abs if max_abs > 0.0 else 1.0


def add_geometry(ax, params):
    conductor = Circle(
        (0.0, 0.0),
        params.conductor_radius,
        fill=False,
        color="black",
        linewidth=1.3,
    )
    dielectric = Circle(
        (0.0, 0.0),
        params.dielectric_radius,
        fill=False,
        color="black",
        linestyle="--",
        linewidth=1.2,
    )
    ax.add_patch(conductor)
    ax.add_patch(dielectric)


def plot_image(
    fig,
    ax,
    data,
    values,
    title,
    xlim,
    zlim,
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
        data.left_edge[1],
        data.right_edge[1],
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
    add_geometry(ax, params)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    if add_colorbar:
        fig.colorbar(image, ax=ax, label=cbar_label)
    return image


def save_figure(fig, output_dir, stem, suffix):
    path = output_dir / f"{stem}{suffix}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path)


def plot_phi(data, exact, output_dir, suffix, args, params):
    mask = zoom_mask(data, args.xlim, args.zlim)
    phi_error = data.phi - exact.phi
    phi_limit = symmetric_limit(data.phi[mask], exact.phi[mask])
    error_limit = symmetric_limit(phi_error[mask])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    plot_image(
        fig,
        axes[0],
        data,
        data.phi,
        r"WarpX $\phi$",
        args.xlim,
        args.zlim,
        params,
        "RdBu_r",
        vmin=-phi_limit,
        vmax=phi_limit,
    )
    plot_image(
        fig,
        axes[1],
        data,
        exact.phi,
        r"analytic $\phi$",
        args.xlim,
        args.zlim,
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
        params,
        "RdBu_r",
        vmin=-error_limit,
        vmax=error_limit,
    )
    save_figure(fig, output_dir, f"{PLOT_PREFIX}_phi", suffix)


def plot_e_components(data, exact, output_dir, suffix, args, params):
    mask = zoom_mask(data, args.xlim, args.zlim)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), constrained_layout=True)

    for row, name, sim, analytic in (
        (0, "Ex", data.ex, exact.ex),
        (1, "Ez", data.ez, exact.ez),
    ):
        error = sim - analytic
        field_limit = symmetric_limit(sim[mask], analytic[mask])
        error_limit = symmetric_limit(error[mask])
        plot_image(
            fig,
            axes[row, 0],
            data,
            sim,
            f"WarpX {name}",
            args.xlim,
            args.zlim,
            params,
            "RdBu_r",
            vmin=-field_limit,
            vmax=field_limit,
        )
        plot_image(
            fig,
            axes[row, 1],
            data,
            analytic,
            f"analytic {name}",
            args.xlim,
            args.zlim,
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
            params,
            "RdBu_r",
            vmin=-error_limit,
            vmax=error_limit,
        )

    save_figure(fig, output_dir, f"{PLOT_PREFIX}_e_components", suffix)


def plot_e_magnitude_error(data, exact, output_dir, suffix, args, params):
    mask = zoom_mask(data, args.xlim, args.zlim)
    e_mag = np.sqrt(data.ex**2 + data.ez**2)
    e_exact_mag = np.sqrt(exact.ex**2 + exact.ez**2)
    e_error_mag = np.sqrt((data.ex - exact.ex) ** 2 + (data.ez - exact.ez) ** 2)
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
        params,
        "viridis",
        cbar_label=r"$\log_{10}(|\Delta E|)$",
    )
    save_figure(fig, output_dir, f"{PLOT_PREFIX}_e_magnitude_error", suffix)


def plot_materials(data, output_dir, suffix, args, params):
    region = np.zeros_like(data.epsilon)
    region[data.epsilon > 1.0 + 1.0e-12] = 2.0
    region[data.eb_covered > 0.5] = 1.0

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
        params,
        cmap,
        norm=norm,
        add_colorbar=False,
    )
    cbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["vacuum", "conductor", "dielectric"])
    save_figure(fig, output_dir, f"{PLOT_PREFIX}_materials", suffix)


def interpolate_to_z(data, values, z_line):
    if z_line < data.z[0] or z_line > data.z[-1]:
        raise ValueError(
            f"centerline z={z_line} is outside the cell-centered range "
            f"[{data.z[0]}, {data.z[-1]}]"
        )

    upper = int(np.searchsorted(data.z, z_line, side="left"))
    if upper < data.z.size and np.isclose(z_line, data.z[upper]):
        return values[:, upper]

    lower = upper - 1
    weight = (z_line - data.z[lower]) / (data.z[upper] - data.z[lower])
    return (1.0 - weight) * values[:, lower] + weight * values[:, upper]


def circle_chord_half_width(radius, z_line):
    radius2 = radius**2
    z2 = z_line**2
    if z2 >= radius2:
        return None
    return np.sqrt(radius2 - z2)


def add_centerline_regions(ax, z_line, params, add_labels):
    conductor_half_width = circle_chord_half_width(params.conductor_radius, z_line)
    dielectric_half_width = circle_chord_half_width(params.dielectric_radius, z_line)

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


def plot_centerline(data, output_dir, suffix, args, params):
    z_line = args.centerline_z
    z = np.full_like(data.x, z_line)
    phi_exact, ex_exact, ez_exact = analytic_fields(
        data.x,
        z,
        params.conductor_radius,
        params.dielectric_radius,
        params.epsilon_r,
        params.applied_field,
    )

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.5), sharex=True)
    curves = (
        (r"$\phi$", interpolate_to_z(data, data.phi, z_line), phi_exact),
        ("Ex", interpolate_to_z(data, data.ex, z_line), ex_exact),
        ("Ez", interpolate_to_z(data, data.ez, z_line), ez_exact),
    )

    for ax, (label, sim, analytic) in zip(axes, curves):
        add_centerline_regions(ax, z_line, params, ax is axes[0])
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
    fig.suptitle(f"centerline at z = {z_line:.6g}")
    fig.tight_layout()
    save_figure(fig, output_dir, f"{PLOT_PREFIX}_centerline", suffix)


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
