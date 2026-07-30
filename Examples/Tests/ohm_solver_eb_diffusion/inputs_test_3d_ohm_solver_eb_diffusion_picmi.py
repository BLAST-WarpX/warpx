#!/usr/bin/env python3
#
# --- Test script for the kinetic-fluid hybrid model in WarpX wherein the
# --- embedded-boundary handling of the B-field push is exercised by simulating
# --- resistive diffusion of a magnetic eigenmode inside a conducting square
# --- cavity that is rotated with respect to the grid (a prism extruded along
# --- y). With a uniform resistivity eta and no plasma (rho = 0 everywhere,
# --- with hybrid_pic_model.holmstrom_vacuum_region=True), the generalized
# --- Ohm's law reduces exactly to E = eta/mu0 * curl(B), so
# ---     dBy/dt = eta/mu0 * laplacian(By),
# --- and the Neumann eigenmode By = B1*cos(pi*(zr - a/2)/a) (zr the rotated
# --- in-plane coordinate) decays at the analytic rate
# ---     gamma = eta/mu0 * (pi/a)^2.
# --- The L2 error of By against the analytic solution at fixed time bounds
# --- the accuracy of the embedded-boundary treatment for the stair-step
# --- masks and for the conformal (enlarged cell technique) Faraday update
# --- (hybrid_pic_model.use_conformal_eb), which imposes the wall at the
# --- actual embedded surface with conservation-consistent circulations.

import argparse
import sys

import numpy as np

from pywarpx import picmi

constants = picmi.constants

# Cavity geometry: square of side CAVITY_SIDE rotated by THETA about the
# y-axis, extruded along y (same geometry family as the test in
# Examples/Tests/embedded_boundary_rotated_cube)
THETA = np.pi / 8
CAVITY_SIDE = 1.06  # cavity side length (m)
HALF_WIDTH = CAVITY_SIDE / 2.0
ETA = 1.0e-3  # plasma resistivity (Ohm m)
B1 = 0.01  # initial eigenmode amplitude (T)
N_FLOOR = 1.0e18  # vacuum density floor (m^-3)
MAX_STEPS = 200

DIFFUSIVITY = ETA / constants.mu0  # magnetic diffusivity (m^2/s)
DECAY_RATE = DIFFUSIVITY * (np.pi / CAVITY_SIDE) ** 2  # analytic decay rate (1/s)


def setup_simulation(resolution, substeps, use_conformal_eb, verbose, split_z=False):
    """Create the PICMI simulation object.

    Parameters
    ----------
    resolution: int
        Number of cells along x and z (the y direction always has 8 cells).
    substeps: int
        Number of B-field substeps per step (must be even).
    use_conformal_eb: bool
        Use the conformal (enlarged cell technique) embedded-boundary Faraday
        update instead of the default stair-step approximation.
    verbose: int
        WarpX verbosity.
    split_z: bool
        Decompose along z so box seams cross the conformal borrowing planes.
    """
    # Run to t_end = 0.5/gamma (the mode decays to exp(-0.5) = 0.607) with a
    # resolution-independent dt: convergence scans then measure purely spatial
    # error, and the accumulated time-integration error stays subdominant.
    t_end = 0.5 / DECAY_RATE
    dt = t_end / MAX_STEPS

    # Thin periodic y: the default split is along y only, so the conformal
    # face borrowing (acting within x-z planes) stays inside each box;
    # split_z instead forces box seams across the borrowing planes to
    # exercise the cross-box reduction of the face-extension passes.
    n_cell = [resolution, 8, resolution]
    y_extent = 0.2
    if split_z:
        decomposition = dict(
            warpx_max_grid_size=2048,
            warpx_max_grid_size_z=max(resolution // 2, 8),
            warpx_blocking_factor=8,
        )
    else:
        decomposition = dict(
            warpx_max_grid_size=2048,
            warpx_max_grid_size_y=4,
            warpx_blocking_factor=8,
            warpx_blocking_factor_y=4,
        )

    grid = picmi.Cartesian3DGrid(
        number_of_cells=n_cell,
        lower_bound=[-0.8, -y_extent, -0.8],
        upper_bound=[0.8, y_extent, 0.8],
        lower_boundary_conditions=["dirichlet", "periodic", "dirichlet"],
        upper_boundary_conditions=["dirichlet", "periodic", "dirichlet"],
        lower_boundary_conditions_particles=["absorbing", "periodic", "absorbing"],
        upper_boundary_conditions_particles=["absorbing", "periodic", "absorbing"],
        **decomposition,
    )

    sim = picmi.Simulation(
        time_step_size=dt,
        max_steps=MAX_STEPS,
        particle_shape=1,
        verbose=verbose,
    )
    sim.grid_type = "staggered"

    sim.solver = picmi.HybridPICSolver(
        grid=grid,
        gamma=1.0,
        Te=1.0,
        n0=N_FLOOR,
        n_floor=N_FLOOR,
        plasma_resistivity=ETA,
        substeps=substeps,
        holmstrom_vacuum_region=True,
        use_conformal_eb=True if use_conformal_eb else None,
    )

    sim.embedded_boundary = picmi.EmbeddedBoundary(
        implicit_function=(
            "xr=x*cos(-theta)+z*sin(-theta);"
            "zr=-x*sin(-theta)+z*cos(-theta);"
            "max(max(xr-hw,-(xr+hw)),max(zr-hw,-(zr+hw)))"
        ),
        theta=THETA,
        hw=HALF_WIDTH,
    )

    # Initial B field: the (0,1) Neumann eigenmode of the cavity. By depends
    # only on the in-plane coordinates so it is exactly divergence free and
    # the projection-based divergence cleaner can be skipped.
    B_init = picmi.AnalyticInitialField(
        Bx_expression="0",
        By_expression="B1*cos(pi/a*(-x*sin(-theta)+z*cos(-theta)-a/2))",
        Bz_expression="0",
        warpx_do_initial_div_cleaning=False,
        B1=B1,
        a=CAVITY_SIDE,
        theta=THETA,
    )
    sim.add_applied_field(B_init)

    field_diag = picmi.FieldDiagnostic(
        name="diag1",
        grid=grid,
        period=MAX_STEPS,
        data_list=["B", "J", "J_displacement"],
        write_dir="diags",
        warpx_format="plotfile",
    )
    sim.add_diagnostic(field_diag)

    return sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        help="toggle whether this script is run as a short CI test",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--resolution",
        help="number of cells along x and z",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--conformal",
        help="use the conformal embedded-boundary wall "
        "(hybrid_pic_model.use_conformal_eb): enlarged-cell ECT Faraday on "
        "the staggered grid",
        action="store_true",
    )
    parser.add_argument(
        "--substeps",
        help="number of B-field substeps per step (even)",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--split-z",
        help="decompose along z so box seams cross the conformal borrowing "
        "planes (cross-box seam test)",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="WarpX verbosity",
        type=int,
        default=0,
    )
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    sim = setup_simulation(
        args.resolution,
        args.substeps,
        args.conformal,
        args.verbose,
        args.split_z,
    )
    sim.step()


if __name__ == "__main__":
    main()
