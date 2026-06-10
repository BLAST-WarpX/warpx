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
# --- The L2 error of By against the analytic solution at fixed time, measured
# --- at two resolutions, gives the order of accuracy of the embedded-boundary
# --- treatment: ~1st order for the default stair-step approximation and
# --- ~2nd order for the conformal (enlarged cell technique) update enabled
# --- with hybrid_pic_model.use_conformal_eb.

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

# Parameters for the --pec-j variant, which checks the PEC current boundary
# condition at the embedded boundary: a uniform external current (inert for
# the By diffusion away from the wall) plus a thermal proton fill that presses
# against the conducting wall and spills deposited current onto interior edges
J_EXT = 0.1 * B1 * np.pi / (CAVITY_SIDE * constants.mu0)  # A/m^2
T_ION = 10.0  # eV
N_PLASMA = 1.0e18  # m^-3
PEC_J_STEPS = 24  # the boundary-condition variant needs no decay time


def setup_simulation(resolution, substeps, use_conformal_eb, pec_j, verbose):
    """Create the PICMI simulation object.

    Parameters
    ----------
    resolution: int
        Number of cells along x and z (the y direction always has 8 cells).
    substeps: int
        Number of B-field substeps per step (must be even).
    use_conformal_eb: bool
        Use the conformal (enlarged cell technique) embedded-boundary update
        instead of the default stair-step approximation.
    pec_j: bool
        Add a uniform external current and a thermal proton fill, and output
        the current densities, to test the embedded-boundary PEC current
        boundary condition.
    verbose: int
        WarpX verbosity.
    """
    # Run to t_end = 0.5/gamma (mode decays to exp(-0.5) = 0.607 of its
    # initial amplitude) with the same time step at every resolution so that
    # the measured convergence is purely spatial
    t_end = 0.5 / DECAY_RATE
    dt = t_end / MAX_STEPS
    max_steps = PEC_J_STEPS if pec_j else MAX_STEPS

    if pec_j:
        # Full-depth y (thin y boxes trip an OpenMP deposition issue with
        # particles) and a z split for parallel runs; this variant uses the
        # stair-step masks, so no conformal face borrowing occurs
        n_cell = [resolution, resolution, resolution]
        y_extent = 0.8
        decomposition = dict(
            warpx_max_grid_size=2048,
            warpx_max_grid_size_z=max(resolution // 2, 8),
            warpx_blocking_factor=8,
        )
    else:
        # Thin periodic y; split MPI domains only along y so that the
        # conformal face borrowing, which acts within x-z planes, never
        # crosses a box boundary
        n_cell = [resolution, 8, resolution]
        y_extent = 0.2
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
        max_steps=max_steps,
        particle_shape=1,
        verbose=verbose,
    )

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
        Jy_external_function=f"{J_EXT}" if pec_j else None,
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
    # the projection-based divergence cleaner can be skipped. The --pec-j
    # variant starts from B=0 instead: parser-based B initialization combined
    # with particles and the hybrid solver triggers a pre-existing OpenMP
    # deposition crash, and the boundary-condition check is driven entirely
    # by the external current and the deposited ion current
    if not pec_j:
        B_init = picmi.AnalyticInitialField(
            Bx_expression="0",
            By_expression="B1*cos(pi/a*(-x*sin(-theta)+z*cos(-theta)-a/2))",
            Bz_expression="0",
            B1=B1,
            a=CAVITY_SIDE,
            theta=THETA,
            warpx_do_initial_div_cleaning=False,
        )
        sim.add_applied_field(B_init)

    if pec_j:
        # Thermal protons filling the cavity (zero density inside the wall);
        # particles stream against the conducting wall and their deposition
        # would spill current onto interior edges without the PEC J boundary
        # condition
        vth = np.sqrt(T_ION * constants.q_e / constants.m_p)
        ions = picmi.Species(
            name="ions",
            particle_type="proton",
            initial_distribution=picmi.AnalyticDistribution(
                density_expression=(
                    "n_p*(abs(x*cos(-theta)+z*sin(-theta))<hw)"
                    "*(abs(-x*sin(-theta)+z*cos(-theta))<hw)"
                ),
                momentum_expressions=["0", "0", "0"],
                warpx_momentum_spread_expressions=[f"{vth}"] * 3,
                warpx_density_min=0.5 * N_PLASMA,
                n_p=N_PLASMA,
                theta=THETA,
                hw=HALF_WIDTH,
            ),
        )
        sim.add_species(
            ions,
            layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=4),
        )

    field_diag = picmi.FieldDiagnostic(
        name="diag1",
        grid=grid,
        period=max_steps,
        data_list=["B", "E", "J", "J_displacement"]
        if pec_j
        else ["B", "J", "J_displacement"],
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
        help="use the conformal (enlarged cell technique) EB update",
        action="store_true",
    )
    parser.add_argument(
        "--substeps",
        help="number of B-field substeps per step (even)",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--pec-j",
        help="test the embedded-boundary PEC current boundary condition",
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
        args.resolution, args.substeps, args.conformal, args.pec_j, args.verbose
    )
    sim.step()


if __name__ == "__main__":
    main()
