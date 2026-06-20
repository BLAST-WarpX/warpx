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

from pywarpx import callbacks, fields, picmi

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

# Tolerance of the in-situ electron-pressure Neumann check in the --pec-j
# variant: largest pressure jump across the wall (samples a quarter cell on
# either side along the analytic wall normal) relative to the peak pressure.
# Calibrated at resolution 32 with at least 2x margin; without the even
# embedded-boundary reflection of the pressure the jump is order unity.
TOL_PE_NEUMANN = 0.06


def setup_simulation(
    resolution,
    substeps,
    use_conformal_eb,
    pec_j,
    verbose,
    split_z=False,
    grid_type="staggered",
    divb_clean=False,
):
    """Create the PICMI simulation object.

    Parameters
    ----------
    resolution: int
        Number of cells along x and z (the y direction always has 8 cells).
    substeps: int
        Number of B-field substeps per step (must be even).
    use_conformal_eb: bool
        Use the conformal (enlarged cell technique) embedded-boundary update
        instead of the default stair-step approximation. On a collocated grid
        this selects the direct level-set mirror B fill (the staggered ECT is
        Yee-only); the near-wall order of that mirror is the edge-order
        diagnostic's target.
    grid_type: str
        "staggered" (Yee, default) or "collocated" (nodal). Collocated forces
        direct current deposition (Esirkepov is unsupported there).
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
        # Thin periodic y. The default split is along y only, so the conformal
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
        max_steps=max_steps,
        particle_shape=1,
        verbose=verbose,
    )
    sim.grid_type = grid_type
    if grid_type == "collocated":
        # the collocated (nodal) hybrid path forbids Esirkepov deposition
        sim.current_deposition_algo = "direct"

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

    if divb_clean:
        # The hybrid div(B)/div(J) Marder clean is not exposed as a PICMI
        # kwarg; set the hybrid_pic_model.* inputs directly through the bucket.
        from pywarpx import hybridpicmodel

        hybridpicmodel.divb_clean_alpha = 0.1
        hybridpicmodel.divj_clean_alpha = 0.1

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
        data_list=["B", "E", "J", "J_displacement", "rho"]
        if pec_j
        else ["B", "J", "J_displacement"],
        write_dir="diags",
        warpx_format="plotfile",
    )
    sim.add_diagnostic(field_diag)

    if pec_j:
        # In-situ check of the electron-pressure embedded-boundary condition
        # (the pressure field is internal to the solver and not written by
        # any diagnostic): the wall supports the plasma back-pressure, so the
        # even mirror across the embedded boundary must leave no pressure
        # jump at the wall, and the pressure deep inside the conductor is
        # exactly zero.
        state = {"step": 0}

        def check_electron_pressure():
            state["step"] += 1
            if state["step"] < max_steps:
                return

            pe = fields.ElectronPressureFPWrapper()[...]
            assert np.all(np.isfinite(pe)), (
                "non-finite electron pressure (equation of state applied to "
                "the mirrored charge density inside the conductor?)"
            )

            # nodal coordinates and the analytic rotated-frame wall distance
            nx, ny, nz = pe.shape
            x = np.linspace(-0.8, 0.8, nx)
            z = np.linspace(-0.8, 0.8, nz)
            h = x[1] - x[0]
            hy = 2.0 * y_extent / (ny - 1)
            X, Z = np.meshgrid(x, z, indexing="ij")
            xr = X * np.cos(THETA) - Z * np.sin(THETA)
            zr = X * np.sin(THETA) + Z * np.cos(THETA)
            s = HALF_WIDTH - np.maximum(np.abs(xr), np.abs(zr))  # >0 in fluid

            # deep inside the conductor the pressure is set exactly to zero
            deep = np.broadcast_to((s < -2.5 * h)[:, None, :], pe.shape)
            assert np.max(np.abs(pe[deep])) == 0.0, (
                "nonzero electron pressure deep inside the conductor"
            )

            def trilinear(arr, px, py, pz):
                fi = (px + 0.8) / h
                fj = (py + y_extent) / hy
                fk = (pz + 0.8) / h
                i0 = np.clip(np.floor(fi).astype(int), 0, nx - 2)
                j0 = np.clip(np.floor(fj).astype(int), 0, ny - 2)
                k0 = np.clip(np.floor(fk).astype(int), 0, nz - 2)
                wx, wy, wz = fi - i0, fj - j0, fk - k0
                v = 0.0
                for di in (0, 1):
                    for dj in (0, 1):
                        for dk in (0, 1):
                            w = (
                                (wx if di else 1.0 - wx)
                                * (wy if dj else 1.0 - wy)
                                * (wz if dk else 1.0 - wz)
                            )
                            v = v + w * arr[i0 + di, j0 + dj, k0 + dk]
                return v

            # pressure jump across the xr = +hw wall face, sampled a
            # quarter cell on either side along the analytic outward
            # normal (samples deeper inside would mix in the correctly
            # zeroed deep-interior nodes past the one-cell mirror band)
            zr_s = np.linspace(-0.6, 0.6, 13) * HALF_WIDTH
            y_s = np.zeros_like(zr_s)
            jump = np.empty_like(zr_s)
            for sign, buf in ((1.0, "plus"), (-1.0, "minus")):
                xr_s = HALF_WIDTH + sign * 0.25 * h
                px = xr_s * np.cos(THETA) + zr_s * np.sin(THETA)
                pz = -xr_s * np.sin(THETA) + zr_s * np.cos(THETA)
                if buf == "plus":
                    jump = trilinear(pe, px, y_s, pz)
                else:
                    jump = jump - trilinear(pe, px, y_s, pz)
            resid = np.max(np.abs(jump)) / np.max(pe)
            print(f"electron pressure wall jump (relative): {resid:.3e}")
            assert resid < TOL_PE_NEUMANN, (
                f"electron pressure jump at the wall {resid:.3e} exceeds "
                f"{TOL_PE_NEUMANN} (Neumann embedded-boundary condition)"
            )

        callbacks.installafterstep(check_electron_pressure)

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
        "--split-z",
        help="decompose along z so box seams cross the conformal borrowing "
        "planes (cross-box seam test)",
        action="store_true",
    )
    parser.add_argument(
        "--grid-type",
        choices=["staggered", "collocated"],
        default="staggered",
        help="field staggering: staggered (Yee, default) or collocated (nodal, "
        "the level-set mirror B fill -- edge-order diagnostic)",
    )
    parser.add_argument(
        "--divb-clean",
        action="store_true",
        help="enable the hybrid div(B)/div(J) Marder clean (alpha=0.1) -- "
        "separates curved-wall div growth from the fill error in the "
        "edge-order diagnosis",
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
        args.pec_j,
        args.verbose,
        args.split_z,
        args.grid_type,
        args.divb_clean,
    )
    sim.step()


if __name__ == "__main__":
    main()
