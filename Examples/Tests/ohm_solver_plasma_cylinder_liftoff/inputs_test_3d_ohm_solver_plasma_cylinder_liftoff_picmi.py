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
R_INNER = 0.65
R_OUTER = 0.75
R_PART = 0.385  # the annulus carries the inventory of a full column of this radius;
# n_annulus = N_I * R_PART^2 / (R_OUTER^2 - R_INNER^2). With R_INNER=0.65, R_OUTER=0.75
# this is ~1.6e20 m^-3 (~1.06 N_I). Reduce R_PART to lower the loading density.
# ANNULUS_SMOOTH_CELLS tanh-softens the radial edges so the gradient is grid-resolved.
ANNULUS_SMOOTH_CELLS = 2.0
R_WALL = 0.8  # conducting wall radius (embedded boundary)

# External field ramp: Hermite smoothstep from the bias field to the
# (reversed) main field over TAU_RAMP
BZ_BIAS = -0.01  # T
BZ_REV = 0.5  # T
TAU_RAMP = 10.0e-6  # s

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
    n_floor_frac=N_FLOOR_FRAC,
    isotropic_resistivity=True,
    isotropic_hyper=True,
    isotropic_gradient=False,
    substeps=SUBSTEPS,
    substep_rtol=1.0e-3,
    te=T_E0,
    bz_rev=BZ_REV,
    bz_bias=BZ_BIAS,
    nz=NZ,
    annulus_smooth_cells=ANNULUS_SMOOTH_CELLS,
    grid_type="collocated",
    use_conformal_eb=True,
    eb_b_straight_mirror=False,
    eta_hyper_mult=1.0,
    eta_nodal=False,
    holmstrom_blend_pow=0.0,
    holmstrom_blend_width=2.0,
    holmstrom_switch_mode="edge",
    equilibrium_b=False,
    vacuum=False,
    fill_frac=None,
    particle_shape=1,
    deposition="direct",
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
    # Interior fill density. The legacy default (2*n_floor) sits inside the
    # holmstrom blend window [n_floor, blend_width*n_floor] whenever
    # blend_width >= 2, making the whole interior switch-decision-sensitive
    # every substep on staggered grids; --fill-frac decouples the fill from
    # the floor so it can be placed above the window.
    n_fill = fill_frac * N_I if fill_frac is not None else 2.0 * n_floor
    vth = np.sqrt(constants.q_e * T_I0 / m_i)

    # cell sizes (cubic cells; z slab centered on the z = 5 m midplane)
    dx = 2.0 / resolution
    annulus_w = annulus_smooth_cells * dx  # tanh edge width of the radial profile
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
    eta_hyper = eta_hyper_mult * constants.mu0 * 0.2 * l_i * vA * dL2

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
        particle_shape=particle_shape,
        verbose=verbose,
    )
    sim.current_deposition_algo = deposition
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
        eb_deposit_fold="pec",
        eb_rho_dirichlet=True,
        isotropic_resistivity=isotropic_resistivity,
        isotropic_hyper_resistivity=isotropic_hyper,
        isotropic_gradient=isotropic_gradient if isotropic_gradient else None,
        eta_nodal_interp=True if eta_nodal else None,
        holmstrom_blend_pow=holmstrom_blend_pow if holmstrom_blend_pow > 0 else None,
        holmstrom_blend_width=holmstrom_blend_width
        if holmstrom_blend_pow > 0
        else None,
        holmstrom_switch_mode=holmstrom_switch_mode
        if holmstrom_switch_mode != "edge"
        else None,
        use_rkf45=True,
        substep_rtol=substep_rtol,
        substep_atol=1.0e-8,
        max_substep_attempts=1000,
        # The conformal wall treatment is collocated-only (staggered aborts);
        # on a staggered grid use eb_b_straight_mirror for the B wall instead.
        use_conformal_eb=use_conformal_eb if grid_type == "collocated" else None,
        eb_b_straight_mirror=True if eb_b_straight_mirror else None,
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

    # Seed the initial grid B with the PLASMA self-field only. The split-field
    # external-A machinery adds B_ext(t=0) to Bfield_fp at PIC-loop start
    # (HybridPICInitializeRhoJandB, upstream #6047), so seeding the bias here
    # would double-count it: the loop-start add stacks the external on top of
    # the seed and the per-step subtract/add telescopes the double forever
    # (measured: on-axis Bz = 2*bz_bias at the first CI checkpoint). The
    # Hermite ramp has f(0) = bz_bias, so the external add supplies the bias.
    # The seeds below are divergence-free, so the initial projection div
    # cleaner is disabled.
    if equilibrium_b and not vacuum:
        # Self-consistent MHD (theta-pinch) equilibrium seed instead of the uniform
        # bias: the diamagnetic axial field in radial force balance with the annular
        # plasma pressure, d/dr[P + B_z^2/(2 mu0)] = 0, referenced to the bias field at
        # the annulus/wall. P is the TOTAL (ion + electron) pressure n(r)(T_i+T_e)q_e.
        # This is the exact fixed point of the Grad-Shafranov J = r P'(psi) iteration
        # for a fixed radial density, so B and the particle diamagnetic current are
        # in equilibrium at t=0 and the startup transient (which stiffens the first
        # RKF45 step) is removed. B_z(r) is z-independent and axial, hence discretely
        # divergence-free. The step-1 external subtract leaves the plasma self-field as
        # the diamagnetic deviation B_eq(r) - bz_bias, sustained by the ion current.
        n_ann_eq = N_I * R_PART**2 / (r_outer**2 - R_INNER**2)
        n_fill_eq = n_fill
        k_eq = 2.0 * constants.mu0 * (T_I0 + te) * constants.q_e
        sgn = "" if bz_bias >= 0 else "-"
        rr = "sqrt(x*x+y*y)"
        # Match the (tanh-smoothed) loaded density profile so the seed B is in
        # force balance with the density the particles actually carry.
        if annulus_w > 0.0:
            e_in = f"0.5*(1.0+tanh(({rr}-{R_INNER})/{annulus_w}))"
            e_out = f"0.5*(1.0+tanh(({r_outer}-{rr})/{annulus_w}))"
            n_of_r = f"({n_fill_eq}*(1.0-{e_in})+{n_ann_eq}*{e_in}*{e_out})"
        else:
            n_of_r = (
                f"({n_ann_eq}*(({rr}>={R_INNER})*({rr}<={r_outer}))"
                f"+{n_fill_eq}*({rr}<{R_INNER}))"
            )
        # Seed only the diamagnetic deviation B_eq(r) - bz_bias (the plasma
        # self-field sustained by the ion current); the external add supplies
        # the uniform bias part of B_eq.
        bz_seed = f"{sgn}sqrt({bz_bias**2}+{k_eq}*({n_ann_eq}-{n_of_r}))-({bz_bias})"
    else:
        bz_seed = "0"
    sim.add_applied_field(
        picmi.AnalyticInitialField(
            Bx_expression="0",
            By_expression="0",
            Bz_expression=f"{bz_seed}",
            warpx_do_initial_div_cleaning=False,
        )
    )

    sim.embedded_boundary = picmi.EmbeddedBoundary(
        implicit_function="(x**2+y**2-R_w**2)", R_w=R_WALL
    )

    # Annular column carrying the inventory of a full column of radius R_PART at
    # N_I, plus a low-density interior fill (no plasma between the annulus and the
    # wall). The radial edges are tanh-softened over annulus_smooth_cells cells so
    # the density gradient -- and the diamagnetic current its curl drives -- is
    # grid-resolved rather than a one-cell step (whose curl is a spurious current
    # sheet). annulus_smooth_cells=0 recovers the hard top-hat.
    if vacuum:
        # Vacuum run (--vacuum): no plasma. density_expression 0 -> no particles are
        # injected, so rho = 0 everywhere (clamped to the floor) and the field
        # evolves as the external-A-driven resistive vacuum B with no plasma
        # feedback -- isolates the conformal-EB Faraday push / external ramp /
        # div-clean from the Ohm's-law plasma coupling.
        density_expression = "0"
        dist_kwargs = {}
    else:
        n_annulus = N_I * R_PART**2 / (r_outer**2 - R_INNER**2)
        r_expr = "sqrt(x*x+y*y)"
        dist_kwargs = dict(n_a=n_annulus, n_f=n_fill, R_in=R_INNER, R_out=r_outer)
        if annulus_w > 0.0:
            # smooth top-hat: fill inside R_in, annulus in [R_in, R_out], 0 beyond,
            # with tanh transitions of width annulus_w.
            edge_in = f"0.5*(1.0+tanh(({r_expr}-R_in)/sw))"
            edge_out = f"0.5*(1.0+tanh((R_out-{r_expr})/sw))"
            density_expression = f"n_f*(1.0-{edge_in})+n_a*{edge_in}*{edge_out}"
            dist_kwargs["sw"] = annulus_w
        else:
            density_expression = (
                f"n_a*(({r_expr}>=R_in)*({r_expr}<=R_out))+n_f*({r_expr}<R_in)"
            )
    ions = picmi.Species(
        name="ions",
        mass=m_i,
        charge="q_e",
        initial_distribution=picmi.AnalyticDistribution(
            density_expression=density_expression,
            momentum_expressions=["0", "0", "0"],
            warpx_momentum_spread_expressions=[f"{vth}"] * 3,
            warpx_density_min=0.05 * n_fill,
            **dist_kwargs,
        ),
    )
    sim.add_species(
        ions,
        layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=nppc),
    )

    if not vacuum:
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
        data_list=["B", "E", "rho", "J", "J_displacement", "divB"],
        write_dir="diags",
        warpx_format="plotfile",
    )
    sim.add_diagnostic(field_diag)

    return sim


def install_wall_scraper(species_name, r_standoff, verbose=False):
    """Hold the plasma off the (metal) EB wall with a pure-Python particle scraper.

    A ``callfromparticlescraper`` hook (fired each step in WarpX::Evolve *before* the
    C++ EB scrape and Redistribute) flags ions with ``sqrt(x**2 + y**2) > r_standoff``
    invalid so Redistribute removes them the same step. AMReX marks a particle invalid
    by clearing bit 63 of its packed ``idcpu`` (``particle_impl::make_invalid``;
    ``is_valid == idcpu >> 63``); we do that bit-clear directly with ``xp`` on the
    zero-copy SoA ``idcpu`` view, which aliases WarpX's own buffer (numpy on CPU, cupy
    on GPU), so the flag writes straight back with no copy on either backend.
    NOTE: this deliberately avoids ``amrex.pack_ids``/``unpack_ids`` -- those are
    host-NumPy-only (``py::array_t``) and raise on a cupy device array, which is the
    GPU crash this replaces.

    This is a stand-in for a dielectric standoff, modelling a quartz liner so the
    plasma never contacts the PEC wall. It tests the particle radius directly rather
    than scraping on the ``distance_to_eb`` signed distance, because AMReX
    ``FillSignedDistance`` clamps the deep-fluid distance to a ``ls_roof`` of only
    ``(nGrow+1)*dx`` (a few cells): a signed-distance standoff wider than that roof
    spuriously scrapes the entire interior annulus. Testing radius sidesteps the clamp.

    The proper follow-up is a DielectricEB object (a second embedded boundary carrying
    eps_r / surface charge) so the standoff and macroscopic dielectric physics live in
    the field solve.
    """
    from pywarpx import callbacks
    from pywarpx._libwarpx import libwarpx
    from pywarpx.LoadThirdParty import load_cupy
    from pywarpx.particle_containers import ParticleContainerWrapper

    ions = ParticleContainerWrapper(species_name)
    r2_standoff = r_standoff * r_standoff
    # ~(1 << 63): AND-mask that clears the valid/sign bit of a uint64 idcpu.
    keep_valid_bits = 0x7FFFFFFFFFFFFFFF
    names_checked = []

    def _repair_soa_names_once():
        # On some CUDA toolchains the compile-time SoA names (a static
        # constexpr initializer_list in PIdx) arrive EMPTY at runtime, so
        # every name-based accessor (get_real_comp_index) throws. Restore
        # the 3D PIdx name table before the first lookup. Only safe while
        # no runtime real/int components have been added (they would have
        # appeared in real_soa_names too, which we verify is empty).
        if names_checked:
            return
        names_checked.append(True)
        pc = ions.particle_container
        if not list(pc.real_soa_names):
            pc.set_soa_compile_time_names(["x", "y", "z", "w", "ux", "uy", "uz"], [])

    @callbacks.callfromparticlescraper
    def _scrape_beyond_standoff():
        _repair_soa_names_once()
        xp, _ = load_cupy()  # cupy on GPU, numpy on CPU -- zero-copy either way
        clear_bit63 = xp.uint64(keep_valid_bits)
        level = 0
        xs = ions.get_particle_real_arrays("x", level)
        ys = ions.get_particle_real_arrays("y", level)
        idcpus = ions.get_particle_idcpu_arrays(level)
        nflagged = 0
        for x, y, idcpu in zip(xs, ys, idcpus):
            mask = (x * x + y * y) > r2_standoff
            if not bool(mask.any()):
                continue
            # Clear bit 63 in place (make_invalid) -> removed at next Redistribute.
            idcpu[mask] &= clear_bit63
            nflagged += int(mask.sum())
        if verbose and nflagged:
            libwarpx.amr.Print(
                f"[wall scraper] flagged {nflagged} '{species_name}' beyond "
                f"r = {r_standoff:.5f} m\n"
            )

    return _scrape_beyond_standoff


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
        dest="holmstrom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="zero the Ohm's-law E in vacuum regions (Holmstrom treatment): "
        "suppresses the floor-density Hall amplification in the gap between "
        "the column and the wall. Default ON: without it, noise fluctuations "
        "in the sub-floor band drive spurious vacuum currents that pump "
        "magnetic flux and wall-band div(B) secularly until the B substep "
        "fails mid-ramp (see PR #6994 flux-conservation study); "
        "--no-holmstrom restores the raw Ohm's-law vacuum response.",
    )
    parser.add_argument(
        "--isotropic-resistivity",
        dest="isotropic_resistivity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="corner-curl isotropization of the resistive diffusion. "
        "Default ON (both grids): the all-operators A/B (with the #7040 "
        "isotropized grad Pe in the set) cut peak rho m4/m0 to 0.053 "
        "collocated / 0.145 Yee vs the 0.344 / 0.466 baselines.",
    )
    parser.add_argument(
        "--isotropic-hyper",
        dest="isotropic_hyper",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="isotropic Mehrstellen/Patra-Karttunen hyper-resistivity "
        "Laplacian. Default ON (both grids; see --isotropic-resistivity).",
    )
    parser.add_argument(
        "--isotropic-gradient",
        dest="isotropic_gradient",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="transverse-smoothed (isotropized) electron-pressure gradient "
        "(PR #7040); removes the cos(4*theta) modulation of |grad Pe| that "
        "feeds the particle push. Default ON (both grids; measured the "
        "decisive piece of the all-operators m=4 reduction, and the "
        "operator that finally makes isotropization a net win on Yee).",
    )
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
        "--annulus-smooth-cells",
        dest="annulus_smooth_cells",
        type=float,
        default=ANNULUS_SMOOTH_CELLS,
        help="tanh-smoothing width (in cells) of the annulus radial density edges, "
        "so the density gradient and its diamagnetic current are grid-resolved. "
        "0 = hard top-hat (a one-cell step whose curl is a spurious current sheet).",
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
        "--n-floor-frac",
        help="vacuum density floor as a fraction of N_I (sets the "
        "vacuum/transition classification of the holmstrom and "
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
        help="field grid staggering: 'collocated' (nodal, level-set conformal "
        "EB) or 'staggered' (Yee, staircase EB; pair with "
        "--eb-b-straight-mirror for the B wall)",
        choices=["staggered", "collocated"],
        default="collocated",
    )
    parser.add_argument(
        "--no-conformal-eb",
        dest="conformal_eb",
        action="store_false",
        default=True,
        help="disable the conformal (ECT/level-set) EB wall solve "
        "(hybrid_pic_model.use_conformal_eb=0); falls back to the standard "
        "masked/staircase EB. Baseline for the 'cost of conformal walls' comparison.",
    )
    parser.add_argument(
        "--eb-b-straight-mirror",
        dest="eb_b_straight_mirror",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="impose the wall on the staggered (Yee) B with the collocated-style "
        "direct level-set mirror after each Faraday push, instead of leaving the "
        "covered faces staircase-zeroed (hybrid_pic_model.eb_b_straight_mirror). "
        "Best with a standoff holding plasma off the wall. Default: ON on "
        "staggered (inert on collocated).",
    )
    parser.add_argument(
        "--mc-gather",
        dest="mc_gather",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="momentum-conserving field gathering (algo.field_gathering): E/B "
        "are interpolated to the nodes before the particle gather, so on staggered "
        "grids the gather is single-valued per node with matched shapes (full "
        "order), instead of the Galerkin gather whose order drops by one in the "
        "staggered directions against Direct deposition. Default: ON on both "
        "grid types (disable with --no-mc-gather).",
    )
    parser.add_argument(
        "--holmstrom-switch-mode",
        dest="holmstrom_switch_mode",
        choices=["edge", "node", "cell"],
        default=None,
        help="sampling mode of the density deciding the holmstrom vacuum switch "
        "and blend weight (hybrid_pic_model.holmstrom_switch_mode): 'edge' = "
        "per-component half-cell-shifted average (legacy), 'node' = endpoint "
        "min (vacuum-favoring, single-valued per node), 'cell' = cell-centered "
        "average (one decision per cell for all three E components). "
        "Default: 'cell' on staggered (C4-equivariant), 'edge' on collocated "
        "(where the sampling is single-valued anyway).",
    )
    parser.add_argument(
        "--eta-hyper-mult",
        type=float,
        default=1.0,
        dest="eta_hyper_mult",
        help="multiplier on the Chacon-style hyper-resistivity floor "
        "(eta_hyper = mult * 0.2 * mu0 * l_i * vA * dL2). Stabilization knob "
        "for the staggered (Yee) runs.",
    )
    parser.add_argument(
        "--eta-nodal",
        dest="eta_nodal",
        action="store_true",
        help="evaluate the (hyper-)resistivity once per node and interpolate "
        "to the staggered E locations (hybrid_pic_model.eta_nodal_interp): a "
        "single-valued eta across the plasma/vacuum seam on Yee grids.",
    )
    parser.add_argument(
        "--holmstrom-blend-pow",
        type=float,
        default=None,
        dest="holmstrom_blend_pow",
        help="smooth the holmstrom vacuum switch: over rho in [floor, "
        "width*floor] the Hall/pressure content of E ramps in as "
        "((rho-floor)/((width-1)*floor))**pow from the vacuum value "
        "(0 = legacy binary switch). Default: 1 on staggered (the blend is "
        "the load-bearing seam regularization there), 0 on collocated.",
    )
    parser.add_argument(
        "--holmstrom-blend-width",
        type=float,
        default=None,
        dest="holmstrom_blend_width",
        help="upper edge of the blend window in units of the density floor. "
        "Default: 4 on staggered (narrower windows measurably choke the "
        "moving plasma/vacuum seam), 2 on collocated (inert while the blend "
        "pow is 0).",
    )
    parser.add_argument(
        "--ni-mult",
        type=float,
        default=1.0,
        dest="ni_mult",
        help="multiplier on the reference density N_I (scales the annulus "
        "loading, the density floor, l_i and vA consistently).",
    )
    parser.add_argument(
        "--fill-frac",
        type=float,
        default=0.20,
        dest="fill_frac",
        help="interior fill density as a fraction of N_I, decoupled from the "
        "density floor. Feeds both the loaded profile and the --equilibrium-b "
        "seed. Default 0.20 (= the staggered w4 blend-window top, so the "
        "interior is not switch-decision-sensitive); the legacy 2*n_floor "
        "coupling is --fill-frac 0.10 at the default floor.",
    )
    parser.add_argument(
        "--particle-shape",
        type=int,
        choices=[1, 2, 3],
        default=2,
        dest="particle_shape",
        help="macroparticle shape-factor order (interpolation.nox). Order 2 "
        "(TSC) smooths deposition/gather noise; pair with --standoff-cells "
        ">= shape so the wider stencil never reaches the wall band. "
        "Default: 2 on both grid types.",
    )
    parser.add_argument(
        "--deposition",
        choices=["direct", "esirkepov"],
        default=None,
        dest="deposition",
        help="ion current deposition scheme. 'esirkepov' is charge-conserving "
        "(div J_i = -d(rho)/dt discretely on the Yee staggering) and, unlike "
        "'direct', honors the EB reduced-particle-shape clamp near cut cells; "
        "staggered grids only (WarpX aborts on collocated). Charge conservation "
        "is important for keeping the current sheet coherent. "
        "Default: esirkepov on staggered, direct on collocated.",
    )
    parser.add_argument(
        "--equilibrium-b",
        action="store_true",
        dest="equilibrium_b",
        help="seed a self-consistent MHD (theta-pinch) equilibrium B (diamagnetic radial "
        "force balance with the total ion+electron pressure) instead of the uniform bias.",
    )
    parser.add_argument(
        "--standoff-cells",
        type=float,
        default=0.0,
        dest="standoff_cells",
        help="dielectric standoff: hold the plasma this many cells off the (metal) field "
        "wall with a Python radius scraper (install_wall_scraper). Models a quartz liner "
        "so the plasma never contacts the PEC. 0 = no standoff.",
    )
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="run with NO plasma (empty ion species): rho=0 everywhere, so the field "
        "evolves as the external-A-driven resistive vacuum B through the conformal EB, "
        "with no Ohm's-law plasma feedback. Isolates the field solver (ECT Faraday / "
        "external ramp / div-clean). Forces the uniform-bias seed; skips collisions "
        "and the wall scraper.",
    )
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    # Grid-type-resolved defaults (the validated apples-to-apples configs;
    # explicit flags always override). Both grids: TSC (order-2) shapes,
    # momentum-conserving gather, fill above the blend window. Staggered adds
    # the charge-conserving Esirkepov deposition (keeps the current sheet
    # coherent: div J_i consistent with d(rho)/dt discretely; aborts on
    # collocated, which uses direct), the level-set wall treatment (straight
    # mirror + component-center update masks), and the cell-sampled holmstrom
    # switch with the p1w4 blend (the seam regularization); collocated keeps
    # its conformal wall. BOTH grids now default to the full isotropized
    # operator set (resistivity corner-curl, hyper-resistivity Laplacian, and
    # the #7040 grad Pe): the all-operators n=128 A/B cut peak rho m4/m0 to
    # 0.053 collocated / 0.145 Yee vs the 0.344 / 0.466 baselines (the old
    # staggered OFF default predated the isotropized gradient).
    staggered = args.grid_type == "staggered"
    if args.deposition is None:
        args.deposition = "esirkepov" if staggered else "direct"
    if args.isotropic_resistivity is None:
        args.isotropic_resistivity = True
    if args.isotropic_hyper is None:
        args.isotropic_hyper = True
    if args.isotropic_gradient is None:
        args.isotropic_gradient = True
    if args.eb_b_straight_mirror is None:
        args.eb_b_straight_mirror = staggered
    if args.holmstrom_switch_mode is None:
        args.holmstrom_switch_mode = "cell" if staggered else "edge"
    if args.holmstrom_blend_pow is None:
        args.holmstrom_blend_pow = 1.0 if staggered else 0.0
    if args.holmstrom_blend_width is None:
        args.holmstrom_blend_width = 4.0 if staggered else 2.0

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

    # Tie the annulus outer radius to the dielectric standoff so the plasma is loaded
    # exactly up to the standoff radius: nothing is scraped at t=0, and the total ion
    # inventory (= N_I*R_PART^2, independent of r_outer via the n_annulus formula) is
    # identical across standoff values -- a fair comparison. Without this, a larger
    # standoff scrapes away more of a fixed 0.7-0.8 annulus and the runs no longer
    # carry the same plasma.
    if args.standoff_cells > 0.0:
        r_outer_eff = R_WALL - args.standoff_cells * (2.0 / resolution)
    else:
        r_outer_eff = args.r_outer
    if not args.vacuum and r_outer_eff <= R_INNER:
        parser.error(
            f"effective annulus outer radius {r_outer_eff:.4f} m (R_WALL - "
            f"standoff_cells*dx at this resolution) does not exceed "
            f"R_INNER={R_INNER} m; the annulus inverts (negative loading "
            "density). Increase --resolution or reduce --standoff-cells."
        )

    if args.ni_mult != 1.0:
        # N_I is a module global read inside setup_simulation at call time, so
        # scaling it here rescales the loading, the floor, l_i and vA together.
        global N_I
        N_I = N_I * args.ni_mult

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
        r_outer_eff,
        args.wall_supported,
        args.n_floor_frac,
        args.isotropic_resistivity,
        args.isotropic_hyper,
        args.isotropic_gradient,
        args.substeps,
        args.substep_rtol,
        args.te,
        args.bz_rev,
        args.bz_bias,
        args.nz,
        annulus_smooth_cells=args.annulus_smooth_cells,
        grid_type=args.grid_type,
        use_conformal_eb=args.conformal_eb,
        eb_b_straight_mirror=args.eb_b_straight_mirror,
        eta_hyper_mult=args.eta_hyper_mult,
        eta_nodal=args.eta_nodal,
        holmstrom_blend_pow=args.holmstrom_blend_pow,
        holmstrom_blend_width=args.holmstrom_blend_width,
        holmstrom_switch_mode=args.holmstrom_switch_mode,
        equilibrium_b=args.equilibrium_b,
        vacuum=args.vacuum,
        fill_frac=args.fill_frac,
        particle_shape=args.particle_shape,
        deposition=args.deposition,
    )

    # Momentum-conserving gather: set on the picmi Simulation object (the picmi
    # layer writes algo.field_gathering itself at initialize, so a raw bucket set
    # here would be overwritten).
    if args.mc_gather:
        sim.field_gathering_algo = "momentum-conserving"

    # Dielectric standoff: hold the plasma args.standoff_cells cells off the (metal)
    # field wall with a Python particle scraper (a callfromparticlescraper hook that
    # flags ions beyond r_standoff for removal). Scraping on radius -- not the AMReX
    # signed-distance field, which is clamped a few cells out and would over-scrape a
    # wide annulus -- keeps the standoff exact at any width. r_standoff matches the
    # annulus outer radius (r_outer_eff) so nothing is scraped at t=0. See
    # install_wall_scraper.
    if args.standoff_cells > 0.0 and not args.vacuum:
        r_standoff = R_WALL - args.standoff_cells * (2.0 / resolution)
        install_wall_scraper("ions", r_standoff, verbose=args.verbose)

    sim.step()


if __name__ == "__main__":
    main()
