#!/usr/bin/env python3
#
# --- Theta-pinch liftoff of an annular plasma column (hybrid-PIC solver).
# ---
# --- A thin 3D slab (one blocking factor of cells, periodic in z) is cut
# --- around the formation-section midplane of a hybrid-PIC formation
# --- simulation: an annular deuterium column rests inside a conducting
# --- cylindrical wall (embedded boundary), threaded by a -0.1 T bias field.
# --- A uniform external axial field, applied through the external vector
# --- potential machinery with a Hermite-smoothstep time ramp, reverses the
# --- field up to +1.5 T: the annulus lifts off the wall region and implodes
# --- (theta-pinch formation with a reversed bias core).
# ---
# --- The slab geometry isolates the in-plane (r, theta) dynamics: this case
# --- is the testbed for the implosion dynamics and for probing the growth of
# --- numerically seeded azimuthal modes (the Cartesian grid breaks the
# --- rotational symmetry with an m=4 perturbation), measured by the
# --- companion analysis script from the azimuthal spectra of rho and Bz.
# ---
# --- NOTE (multi-box runs): the conformal face borrowing does not
# --- communicate across grid boxes (inherited ECT limitation, see the
# --- use_conformal_eb documentation), and in this slab the z direction is a
# --- single block, so any decomposition puts box seams across the wall
# --- circle. Single-box (single-GPU) runs are unaffected; for mode-growth
# --- studies on many ranks, ghost-aware borrowing is pending follow-up work.

import argparse
import sys

import numpy as np

from pywarpx import picmi

constants = picmi.constants

# ----------------------------------------------------------------------------
# Physics parameters: slab at the z = 5 m midplane of the formation section
# ----------------------------------------------------------------------------
M_AMU = 2  # deuterium
N_I = 1.5e20  # reference ion density (m^-3)
N_FLOOR_FRAC = 0.05
T_I0 = 5.0  # ion temperature (eV)
T_E0 = 5.0  # electron temperature (eV): no electron pressure

# Annular column (radii in m) and low-density interior fill
R_INNER = 0.7
R_OUTER = 0.8
R_PART = 0.6  # the annulus carries the inventory of a full column of this radius
R_WALL = 0.8  # conducting wall radius (embedded boundary)

# External field ramp: Hermite smoothstep from the bias field to the
# (reversed) main field over TAU_RAMP
BZ_BIAS = -0.025  # T
BZ_REV = 0.85  # T
TAU_RAMP = 8.0e-6  # s

# Resistivity (density-scaled power law) and hyper-resistivity
ETA_PLASMA = 1.0e-6  # Ohm m, bulk plasma
ETA_VAC_FRAC = 5.0e-2  # vacuum resistivity as a fraction of the CFL limit
ETA_POWER = 3.0
N_TRANSITION_FRAC = 0.4

# Time stepping
F_T_CI = 0.01  # dt as a fraction of the (reversed-field) ion cyclotron period
SUBSTEPS = 256  # initial RKF45 substep count

NZ = 8  # one blocking factor of cells: thin periodic slab


def hermite_ramp_expression(b0, b1, tau):
    """Time scaling of the unit-Bz external vector potential: Hermite
    smoothstep h(s) = s^2*(3-2*s) from b0 (bias) to b1 (reversed field)."""
    s = f"min(max(t/{tau:.9e},0),1)"
    return f"({b0:.9e} + ({b1 - b0:.9e})*({s})*({s})*(3-2*({s})))"


def power_law_resistivity(
    eta_plasma, eta_vac, power, n_floor_frac, n_transition_frac, n0
):
    """Density-scaled resistivity interpolating from eta_plasma in the bulk
    (n >= n_transition) to eta_vac at the density floor, as a power law of
    the density (see the formation simulation this case is derived from)."""
    a = (n_floor_frac / n_transition_frac) ** power
    res_str = (
        f"eta_plasma + (eta_vac - eta_plasma)"
        f"*max(0.0, (rho_f/max(rho,rho_f))**({power:.6g}) - ({a:.12g}))"
        f"/({1.0 - a:.12g})"
    )
    return {
        "plasma_resistivity": res_str,
        "eta_plasma": eta_plasma,
        "eta_vac": eta_vac,
        "rho_f": constants.q_e * n0 * n_floor_frac,
    }


def setup_simulation(
    resolution,
    nppc,
    max_steps,
    diag_period,
    verbose,
    holmstrom=False,
    eta_power=ETA_POWER,
    eta_ntrans=N_TRANSITION_FRAC,
    f_t_ci=F_T_CI,
    r_outer=R_OUTER,
    wall_supported=True,
    marder=True,
    marder_alpha=0.1,
    marder_target="grad_pe_only",
    marder_level="half_steps",
    marder_interval=1,
    n_floor_frac=N_FLOOR_FRAC,
    isotropic_resistivity=True,
    isotropic_hyper=True,
    substeps=SUBSTEPS,
    substep_rtol=1.0e-3,
    te=T_E0,
    bz_rev=BZ_REV,
    bz_bias=BZ_BIAS,
    nz=NZ,
    grid_type="collocated",
):
    """Create the PICMI simulation object.

    Parameters
    ----------
    resolution: int
        Number of cells along x and y (z always has NZ cells at the same
        spacing).
    nppc: int
        Macroparticles per cell in the loaded region.
    max_steps: int
        Number of steps to run.
    diag_period: int
        Field-diagnostic period in steps.
    verbose: int
        WarpX verbosity.
    """
    m_i = M_AMU * constants.m_p
    n_floor = n_floor_frac * N_I
    vth = np.sqrt(constants.q_e * T_I0 / m_i)

    # cell sizes (cubic cells; z slab centered on the z = 5 m midplane)
    dx = 2.0 / resolution
    lz = nz * dx
    zmin, zmax = 5.0 - lz / 2.0, 5.0 + lz / 2.0

    # time step from the reversed-field ion cyclotron period
    w_ci = constants.q_e * abs(bz_rev) / m_i
    t_ci = 2.0 * np.pi / w_ci
    dt = f_t_ci * t_ci

    # ion skin depth and Alfven speed at the reversed field, for the
    # Chacon-style hyper-resistivity floor; CFL-limited vacuum resistivity
    w_pi = np.sqrt(constants.q_e**2 * N_I / (constants.ep0 * m_i))
    l_i = constants.c / w_pi
    vA = abs(bz_rev) / np.sqrt(constants.mu0 * N_I * m_i)
    dL2 = 1.0 / (2.0 / dx**2 + 1.0 / dx**2)  # 1/(1/dx^2+1/dy^2+1/dz^2), cubic
    eta_max = constants.mu0 * dL2 / (2.0 * dt)
    eta_hyper = constants.mu0 * 0.2 * l_i * vA * dL2

    grid = picmi.Cartesian3DGrid(
        number_of_cells=[resolution, resolution, nz],
        lower_bound=[-1.0, -1.0, zmin],
        upper_bound=[1.0, 1.0, zmax],
        lower_boundary_conditions=["dirichlet", "dirichlet", "periodic"],
        upper_boundary_conditions=["dirichlet", "dirichlet", "periodic"],
        lower_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
        upper_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
        warpx_max_grid_size=2048,
        warpx_max_grid_size_x=max(resolution // 2, 8),
        warpx_blocking_factor=8,
    )

    sim = picmi.Simulation(
        time_step_size=dt,
        max_steps=max_steps,
        particle_shape=1,
        verbose=verbose,
    )
    sim.current_deposition_algo = "direct"
    sim.grid_type = grid_type

    # External reversal field: uniform Bz through A = (-y/2, x/2, 0) * f(t),
    # with the Hermite ramp from the bias to the reversed field in f(t)
    A_ext = {
        "uniform_reversal": {
            "Ax_external_function": "-0.5*y",
            "Ay_external_function": "0.5*x",
            "Az_external_function": "0",
            "A_time_external_function": hermite_ramp_expression(
                bz_bias, bz_rev, TAU_RAMP
            ),
        },
    }

    sim.solver = picmi.HybridPICSolver(
        grid=grid,
        gamma=5.0 / 3.0,
        Te=te,
        n0=N_I,
        n_floor=n_floor,
        plasma_hyper_resistivity=eta_hyper,
        substeps=substeps,
        holmstrom_vacuum_region=True if holmstrom else None,
        eb_deposit_fold="pec" if wall_supported else None,
        eb_rho_dirichlet=True if wall_supported else None,
        marder_alpha=marder_alpha if marder else None,
        marder_target=marder_target if marder else None,
        marder_correction_level=marder_level if marder else None,
        marder_max_iterations=10 if marder else None,
        marder_rtol=1.0e-3 if marder else None,
        marder_substep_interval=marder_interval if marder else None,
        isotropic_resistivity=isotropic_resistivity,
        isotropic_hyper_resistivity=isotropic_hyper,
        use_rkf45=True,
        substep_rtol=substep_rtol,
        substep_atol=1.0e-8,
        max_substep_attempts=1000,
        use_conformal_eb=True,
        A_external=A_ext,
        **power_law_resistivity(
            ETA_PLASMA,
            ETA_VAC_FRAC * eta_max,
            eta_power,
            n_floor_frac,
            eta_ntrans,
            N_I,
        ),
    )

    sim.embedded_boundary = picmi.EmbeddedBoundary(
        implicit_function="(x**2+y**2-R_w**2)", R_w=R_WALL
    )

    # Annular column carrying the inventory of a full column of radius
    # R_PART at N_I, plus a low-density interior fill (no plasma between the
    # annulus and the wall)
    n_annulus = N_I * R_PART**2 / (r_outer**2 - R_INNER**2)
    n_fill = 2.0 * n_floor
    r_expr = "sqrt(x*x+y*y)"
    ions = picmi.Species(
        name="ions",
        mass=m_i,
        charge="q_e",
        initial_distribution=picmi.AnalyticDistribution(
            density_expression=(
                f"n_a*(({r_expr}>=R_in)*({r_expr}<=R_out))+n_f*({r_expr}<R_in)"
            ),
            momentum_expressions=["0", "0", "0"],
            warpx_momentum_spread_expressions=[f"{vth}"] * 3,
            warpx_density_min=0.5 * n_fill,
            n_a=n_annulus,
            n_f=n_fill,
            R_in=R_INNER,
            R_out=r_outer,
        ),
    )
    sim.add_species(
        ions,
        layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=nppc),
    )

    ion_ion_coulomb = picmi.CoulombCollisions(
        name="ion_ion_Coulomb",
        species=[ions, ions],
        CoulombLog=12,
    )
    sim.collisions = [ion_ion_coulomb]

    field_diag = picmi.FieldDiagnostic(
        name="field_diag",
        grid=grid,
        period=diag_period,
        data_list=["B", "E", "rho", "J"],
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
        help="number of cells along x and y",
        type=int,
        default=72,
    )
    parser.add_argument(
        "--nppc",
        help="macroparticles per cell",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--steps",
        help="number of steps (default: full ramp plus one cyclotron period)",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--diag-steps",
        help="field diagnostic period in steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--eta-power",
        help="power of the density-scaled resistivity ramp",
        type=float,
        default=ETA_POWER,
    )
    parser.add_argument(
        "--eta-ntrans",
        help="transition density fraction of the resistivity ramp",
        type=float,
        default=N_TRANSITION_FRAC,
    )
    parser.add_argument(
        "--f-tci",
        help="time step as a fraction of the reversed-field ion cyclotron "
        "period (reduce to manage whistler stiffness, e.g. with the Hall "
        "term active in low-density regions)",
        type=float,
        default=F_T_CI,
    )
    parser.add_argument(
        "--r-outer",
        help="outer radius of the annular column (set near R_WALL=0.8 for a "
        "wall-supported start)",
        type=float,
        default=R_OUTER,
    )
    parser.add_argument(
        "--wall-supported",
        help="treat the embedded boundary as a supporting wall: reflecting "
        "deposit fold (mass conserving) and Neumann density mirror instead "
        "of the PEC image fold and Dirichlet-0 density",
        action="store_true",
    )
    parser.add_argument(
        "--holmstrom",
        help="zero the Ohm's-law E in vacuum regions (Holmstrom treatment): "
        "suppresses the floor-density Hall amplification in the gap between "
        "the column and the wall",
        action="store_true",
    )
    parser.add_argument(
        "--marder",
        help="enable the transitional Marder divergence cleaning of the "
        "substep E in the low-density band (0 < rho <= n_floor*q_e), with "
        "the pressure-only target (a natural companion to --holmstrom)",
        action="store_true",
    )
    parser.add_argument(
        "--no-isotropic-resistivity",
        dest="isotropic_resistivity",
        action="store_false",
        help="disable the corner-curl isotropization of the resistive "
        "diffusion (on by default; A/B the grid m=4 mode amplitude)",
    )
    parser.set_defaults(isotropic_resistivity=True)
    parser.add_argument(
        "--no-isotropic-hyper",
        dest="isotropic_hyper",
        action="store_false",
        help="disable the isotropic Mehrstellen/Patra-Karttunen hyper-"
        "resistivity Laplacian (on by default)",
    )
    parser.set_defaults(isotropic_hyper=True)
    parser.add_argument(
        "--nz",
        dest="nz",
        help="number of axial (z) cells in the periodic slab (default 8 = thin "
        "2D slab; larger lets the liftoff interface relieve axially, closer to "
        "a full 3D run). Stays one z-box for the conformal EB.",
        type=int,
        default=NZ,
    )
    parser.add_argument(
        "--bz-rev",
        dest="bz_rev",
        help="reversed (target) external Bz in Tesla (default 1.5; lower = "
        "gentler compression / less stiff)",
        type=float,
        default=BZ_REV,
    )
    parser.add_argument(
        "--bz-bias",
        dest="bz_bias",
        help="initial bias external Bz in Tesla (default -0.1)",
        type=float,
        default=BZ_BIAS,
    )
    parser.add_argument(
        "--te",
        dest="te",
        help="electron temperature in eV (electron pressure / grad-Pe term; "
        "0 = none, the slab default. A finite Te provides back-pressure that "
        "stabilizes the compression near stagnation, as in production runs)",
        type=float,
        default=T_E0,
    )
    parser.add_argument(
        "--substeps",
        help="number of RKF45 B-field substeps per ion step (more substeps "
        "shrink dt_B and relieve the whistler/resistive stiffness that the "
        "adaptive solver otherwise hits near stagnation)",
        type=int,
        default=SUBSTEPS,
    )
    parser.add_argument(
        "--substep-rtol",
        dest="substep_rtol",
        help="relative tolerance of the RKF45 substep error controller",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--marder-alpha",
        help="Marder damping factor (with --marder)",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--marder-target",
        help="Marder divergence target (with --marder)",
        choices=["ohm", "grad_pe_only", "zero"],
        default="grad_pe_only",
    )
    parser.add_argument(
        "--marder-level",
        help="Marder application cadence (with --marder)",
        choices=["all_substeps", "half_steps", "full_steps"],
        default="half_steps",
    )
    parser.add_argument(
        "--marder-interval",
        help="apply every N substep E evaluations (with --marder, all_substeps level)",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--n-floor-frac",
        help="vacuum density floor as a fraction of N_I (sets the "
        "vacuum/transition classification of the holmstrom, Marder and "
        "resistivity-ramp treatments; lower is stiffer)",
        type=float,
        default=N_FLOOR_FRAC,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="WarpX verbosity",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--grid-type",
        help="field grid staggering: 'staggered' (Yee, enlarged-cell conformal "
        "EB) or 'collocated' (nodal, level-set conformal EB)",
        choices=["staggered", "collocated"],
        default="collocated",
    )
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    if args.test:
        resolution = 32
        nppc = 4
        max_steps = 10
        diag_period = 10
    else:
        resolution = args.resolution
        nppc = args.nppc
        m_i = M_AMU * constants.m_p
        t_ci = 2.0 * np.pi * m_i / (constants.q_e * abs(args.bz_rev))
        dt = args.f_tci * t_ci
        max_steps = (
            args.steps if args.steps is not None else int((TAU_RAMP + t_ci) / dt)
        )
        diag_period = (
            args.diag_steps if args.diag_steps is not None else max(max_steps // 100, 1)
        )

    sim = setup_simulation(
        resolution,
        nppc,
        max_steps,
        diag_period,
        args.verbose,
        args.holmstrom,
        args.eta_power,
        args.eta_ntrans,
        args.f_tci,
        args.r_outer,
        args.wall_supported,
        args.marder,
        args.marder_alpha,
        args.marder_target,
        args.marder_level,
        args.marder_interval,
        args.n_floor_frac,
        args.isotropic_resistivity,
        args.isotropic_hyper,
        args.substeps,
        args.substep_rtol,
        args.te,
        args.bz_rev,
        args.bz_bias,
        args.nz,
        grid_type=args.grid_type,
    )
    sim.step()


if __name__ == "__main__":
    main()
