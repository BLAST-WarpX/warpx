#!/usr/bin/env python3
"""Exercise material evolution and current sources on static refined grids."""

import argparse
import sys
from pathlib import Path

import numpy as np

from pywarpx import amr, libwarpx, picmi, warpx

parser = argparse.ArgumentParser()
parser.add_argument("--dims", choices=("2", "3", "RZ"), required=True)
parser.add_argument("--source", choices=("port", "antenna", "material"), required=True)
parser.add_argument(
    "--placement", choices=("inside", "outside", "crossing", "axial"), default="inside"
)
parser.add_argument(
    "--method", choices=("backwardeuler", "laxwendroff"), default="backwardeuler"
)
parser.add_argument("--restart")
parser.add_argument("--eb", action="store_true")
parser.add_argument("--annular", action="store_true")
parser.add_argument("--implicit", action="store_true")
parser.add_argument("--theta", type=float, default=0.5)
parser.add_argument("--plasma", action="store_true")
parser.add_argument("--petsc", action="store_true")
args, remaining = parser.parse_known_args()
sys.argv = sys.argv[:1] + remaining

ndim = 3 if args.dims == "3" else 2
is_rz = args.dims == "RZ"
lower = [0.0, -0.5] if is_rz else [-0.5] * ndim
upper = [0.5, 0.5] if is_rz else [0.5] * ndim
refined_lower = [0.0, -0.375] if is_rz else [-0.375] * ndim
refined_upper = [0.4375, 0.375] if is_rz else [0.375] * ndim
if args.annular:
    refined_lower[0] = 0.0625
if args.placement == "outside":
    refined_upper[0] = 0.125 if is_rz else -0.25
elif args.placement == "crossing" or args.source == "antenna":
    refined_upper[0] = 0.25 if is_rz else 0.0
elif args.placement == "axial":
    refined_lower[-1] = -0.0625

grid_type = {
    "2": picmi.Cartesian2DGrid,
    "3": picmi.Cartesian3DGrid,
    "RZ": picmi.CylindricalGrid,
}[args.dims]
refined_regions = [[1, refined_lower, refined_upper]]
grid = grid_type(
    number_of_cells=[32] * ndim,
    lower_bound=lower,
    upper_bound=upper,
    lower_boundary_conditions=["none", "periodic"] if is_rz else ["periodic"] * ndim,
    upper_boundary_conditions=["dirichlet", "periodic"]
    if is_rz
    else ["periodic"] * ndim,
    lower_boundary_conditions_particles=["none", "periodic"]
    if is_rz
    else ["periodic"] * ndim,
    upper_boundary_conditions_particles=["reflecting", "periodic"]
    if is_rz
    else ["periodic"] * ndim,
    refined_regions=refined_regions,
    warpx_max_grid_size=16,
    warpx_blocking_factor=4,
)
amr.n_error_buf = 0

dt = 1.0e-12
simulation = picmi.Simulation(
    solver=picmi.ElectromagneticSolver(
        grid=grid,
        method="Yee",
        divE_cleaning=args.source == "antenna" and not args.implicit and not is_rz,
    ),
    max_steps=2,
    time_step_size=dt,
    particle_shape="linear",
    warpx_current_deposition_algo="esirkepov",
    warpx_use_filter=0,
    warpx_serialize_initial_conditions=1,
    warpx_amr_restart=args.restart,
    warpx_evolve_scheme=(
        picmi.ThetaImplicitEMEvolveScheme(
            theta=args.theta,
            nonlinear_solver=picmi.NewtonNonlinearSolver(
                verbose=False,
                require_convergence=True,
                max_iterations=8,
                relative_tolerance=1.0e-12,
                absolute_tolerance=1.0e-14,
                linear_solver=(
                    picmi.PETScKSPLinearSolver()
                    if args.petsc
                    else picmi.GMRESLinearSolver(
                        verbose_int=0, relative_tolerance=1.0e-6, max_iterations=100
                    )
                ),
            ),
        )
        if args.implicit
        else None
    ),
    warpx_embedded_boundary=(
        picmi.EmbeddedBoundary(
            implicit_function="0.0625**2-x*x-y*y" if ndim == 3 else "0.0625**2-x*x"
        )
        if args.eb
        else None
    ),
)
# Parser fields distinguish fine/coarse coordinates and exercise non-vacuum mu.
for name, expression in (
    ("epsilon", f"{picmi.constants.ep0}*(2+x)"),
    ("sigma", f"{picmi.constants.ep0}*1.e10*(2+x)"),
    ("mu", f"{picmi.constants.mu0}*(2+0.5*z)"),
):
    simulation.add_macroscopic_property(
        picmi.MacroscopicProperty(
            name=name,
            implicit_function=expression,
            method=args.method if name == "sigma" else None,
        )
    )

if args.source == "antenna" and not args.implicit and not is_rz:
    # Keep the two charge time slices used by the continuity oracle below,
    # but do not apply particle-only Gauss cleaning to this conducting medium.
    warpx.get_bucket("macroscopic").dive_cleaning_vacuum_patches_only = 1

if args.plasma:
    for particle_type in ("electron", "proton"):
        simulation.add_species(
            picmi.Species(
                particle_type=particle_type,
                name=particle_type,
                initial_distribution=picmi.UniformDistribution(density=1.0e12),
            ),
            layout=picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=[1] * ndim),
        )

waveform = str(Path(__file__).resolve().parent / "current_profile.txt")
if args.source == "port":
    transverse_lo = 0.1875 if is_rz else -0.125
    transverse_hi = 0.3125 if is_rz else 0.125
    simulation.add_current_controlled_port(
        picmi.CurrentControlledPort(
            direction=2,
            terminal_0_lower_bound=[transverse_lo, -0.125, -0.125],
            terminal_0_upper_bound=[transverse_hi, 0.125, -0.125],
            terminal_1_lower_bound=[transverse_lo, -0.125, 0.125],
            terminal_1_upper_bound=[transverse_hi, 0.125, 0.125],
            file=waveform,
        )
    )
elif args.source == "antenna":
    simulation.add_prescribed_current_injection(
        picmi.PrescribedCurrentInjection(
            file=waveform,
            drives=[
                picmi.PrescribedCurrentDrive(
                    lower_bound=[0.125 if is_rz else -0.25, -0.25, -0.25],
                    upper_bound=[0.375 if is_rz else 0.25, 0.25, 0.25],
                    direction=2,
                    area=np.pi * (0.375**2 - 0.125**2) if is_rz else 0.25,
                )
            ],
        )
    )
if is_rz and (args.source == "material" or args.eb):
    warpx.E_ext_grid_init_style = "constant"
    warpx.E_external_grid = [0, 0, 1]
elif args.source == "material" or args.eb or args.plasma:
    simulation.add_applied_field(
        picmi.AnalyticInitialField(
            Ex_expression="0",
            Ey_expression="0" if is_rz else "1.0",
            Ez_expression="1.0" if is_rz else "0",
        )
    )
elif args.placement in ("crossing", "axial"):
    simulation.add_applied_field(
        picmi.AnalyticInitialField(
            Bx_expression="-1.e-4*cos(2*pi*x)*sin(2*pi*y)" if ndim == 3 else "0",
            By_expression="1.e-4*sin(2*pi*x)*cos(2*pi*y)"
            if ndim == 3
            else "1.e-4*sin(2*pi*x)*cos(2*pi*z)",
            Bz_expression="0",
        )
    )

diagnostic_fields = ["E", "B", "J", "divB", "rho"]
diagnostic_fields += ["sigma", "epsilon", "mu"]
simulation.add_diagnostic(
    picmi.FieldDiagnostic(
        name="diag1",
        grid=grid,
        period=1,
        data_list=diagnostic_fields,
        write_dir="diags",
        warpx_file_prefix="diag1",
    )
)
simulation.add_diagnostic(
    picmi.Checkpoint(name="chk", period=1, write_dir="diags", warpx_file_prefix="chk")
)
independent_rho_before = None
if is_rz and args.eb and not args.restart:
    simulation.initialize_inputs()
    simulation.initialize_warpx()
    for level, patch in ((0, "fp"), (1, "fp"), (1, "cp")):
        field = simulation.fields.get(f"Efield_{patch}", dir="z", level=level)
        values = np.asarray(field[...])
        origin = np.array(field.box_array().minimal_box().small_end)
        dr = 0.5 / (32 * (2 if level == 1 and patch == "fp" else 1))
        radius = (np.arange(values.shape[0]) + origin[0]) * dr
        values[radius < 0.0625] = 0
        field[...] = values
if args.plasma:
    simulation.step(1)
    for name in ("electron", "proton"):
        particles = simulation.particles.get(name)
        charge = -picmi.constants.q_e if name == "electron" else picmi.constants.q_e
        mass = picmi.constants.m_e if name == "electron" else picmi.constants.m_p
        expected = charge * dt / mass / (1 + args.theta * 0.01)
        for level in (0, 1):
            sampled = []
            for tile in libwarpx.libwarpx_so.WarpXParIter(particles, level):
                values = tile.soa().to_numpy(copy=True).real
                mask = (np.abs(values["y"]) < 0.05) & (np.abs(values["z"]) < 0.05)
                mask &= values["x"] > 0.4 if level == 0 else np.abs(values["x"]) < 0.05
                sampled.extend(values["uy"][mask])
            assert sampled
            np.testing.assert_allclose(
                sampled,
                expected,
                rtol=1.0e-6,
                atol=0,
                err_msg=f"{name}: trial electric field gathered on level {level}",
            )
    simulation.step(1)
elif (args.implicit or is_rz) and args.source == "antenna":
    if args.restart:
        simulation.initialize_inputs()
        simulation.initialize_warpx()
    else:
        simulation.step(1)
    # Charge deposition now leaves rho unscaled until explicit finalization.
    # Compare physical charge densities, including boundary folding, on both levels.
    independent_rho_before = [
        np.squeeze(
            np.asarray(simulation.particles.get_charge_density(level, False, True)[...])
        )
        for level in (0, 1)
    ]
    simulation.step(1)
else:
    simulation.step(1 if args.restart else 2)

if args.eb:
    for level, patch in ((0, "fp"), (1, "fp"), (1, "cp")):
        ey_field = simulation.fields.get(
            f"Efield_{patch}", dir="z" if is_rz else "y", level=level
        )
        ey = np.squeeze(np.asarray(ey_field[...]))
        origin = np.array(ey_field.box_array().minimal_box().small_end)
        dx = (upper[0] - lower[0]) / (32 * (2 if level == 1 and patch != "cp" else 1))
        x = lower[0] + (np.arange(ey.shape[0]) + origin[0]) * dx
        if ndim == 3:
            y = lower[1] + (np.arange(ey.shape[1]) + origin[1] + 0.5) * dx
            covered = x[:, None] ** 2 + y[None, :] ** 2 < 0.03**2
        else:
            covered = np.abs(x) < 0.03
        assert np.any(covered)
        np.testing.assert_array_equal(ey[covered], 0)
        assert np.max(np.abs(ey)) > 0.5
        print(f"level {level} {patch}: covered EB electric fields stay zero")

if args.source == "antenna":
    # Discrete continuity is a stronger oracle than normalization alone. The
    # nodal endpoint charge must match the divergence of the staggered Jz.
    for level in (0, 1):
        if independent_rho_before is not None:
            rho_after = np.squeeze(
                np.asarray(
                    simulation.particles.get_charge_density(level, False, True)[...]
                )
            )
            charge_change = rho_after - independent_rho_before[level]
        else:
            rho = np.asarray(simulation.fields.get("rho_fp", level=level)[...])
            rho = np.squeeze(rho)
            charge_change = rho[..., 1] - rho[..., 0]
        jz = np.asarray(simulation.fields.get("current_fp", dir="z", level=level)[...])
        jz = np.squeeze(jz)
        dz = (upper[-1] - lower[-1]) / (32 * 2**level)
        interior = (slice(1, -1),) * ndim
        rate = charge_change[interior] / dt
        divergence = np.diff(jz, axis=ndim - 1) / dz
        divergence = divergence[(slice(1, -1),) * (ndim - 1) + (slice(None),)]
        if independent_rho_before is not None and level == 0:
            # Parent J includes restricted fine current. Independently deposited
            # parent particles describe leaf cells outside the interface stencil.
            dr = (upper[0] - lower[0]) / 32
            x = lower[0] + (np.arange(rate.shape[0]) + 1) * dr
            uncovered = x > refined_upper[0] + 2 * dr
            rate, divergence = rate[uncovered], divergence[uncovered]
        scale = max(np.max(np.abs(rate)), np.max(np.abs(divergence)))
        assert scale > 0
        error = np.max(np.abs(rate + divergence)) / scale
        print(f"level {level} discrete continuity error: {error}")
        assert error < 2.0e-10

if args.source == "port":
    status = np.asarray(libwarpx.warpx.get_current_controlled_port_status())
    if args.plasma:
        for name in ("electron", "proton"):
            particles = simulation.particles.get(name)
            assert particles.sum_particle_energy(False) > 0
    print(f"terminal status: {status}")
    np.testing.assert_allclose(status, [2000.0] * 3, rtol=2.0e-12, atol=0)

    if args.placement in ("crossing", "axial"):
        from composite_contour import measure_contours

        cells = 64
        planes = np.unique(
            [-0.125, 0.125]
            + [
                lower[-1] + (i + 0.5) / cells
                for i in range(3 * cells // 8, 5 * cells // 8)
            ]
        )
        currents = measure_contours(simulation, lower, upper, ndim, planes, 1)
        print(f"independent composite currents: {currents}")
        np.testing.assert_allclose(currents, 2000.0, rtol=2.0e-10, atol=0)
        sys.exit(0)

    # Measure native B independently of the status API on every represented patch.
    for level, patch in ((0, "fp"), (1, "fp"), (1, "cp"), (1, "aux")):
        if level == 1 and args.placement == "outside":
            continue
        dx = np.array(upper) - lower
        dx /= 32 * (2 if level == 1 and patch != "cp" else 1)

        reference = simulation.fields.get(
            f"Bfield_{patch}",
            dir="x" if args.dims == "3" else "y" if not is_rz else "theta",
            level=level,
        )
        origin = np.array(reference.box_array().minimal_box().small_end)

        def field(component):
            return np.squeeze(
                np.asarray(
                    simulation.fields.get(
                        f"Bfield_{patch}", dir=component, level=level
                    )[...]
                )
            )

        def index(coordinate, axis, offset=0):
            return (
                int(np.floor((coordinate - lower[axis]) / dx[axis] - offset + 0.5))
                - origin[axis]
            )

        planes = [index(z, ndim - 1) for z in (-0.125, 0, 0.125)]
        if args.dims == "3":
            bx, by = field("x"), field("y")
            # Yee curl contours surround the vector-potential top hat.
            lo, hi = index(-0.125, 0), index(0.125, 0)
            z = lower[2] + (np.array(planes) + origin[2]) * dx[2]
            mu = picmi.constants.mu0 * (2 + 0.5 * (z + 0.5 * dx[2]))
            currents = (
                (
                    np.sum(bx[lo : hi + 1, lo - 1][:, planes], axis=0)
                    + np.sum(by[hi, lo : hi + 1][:, planes], axis=0)
                    - np.sum(bx[lo : hi + 1, hi][:, planes], axis=0)
                    - np.sum(by[lo - 1, lo : hi + 1][:, planes], axis=0)
                )
                * dx[0]
                / mu
            )
        elif is_rz:
            bt = field("theta")
            lo, hi = index(transverse_lo, 0, 0.5), index(transverse_hi, 0, 0.5)
            planes = [index(z, ndim - 1, 0.5) for z in (-0.125, 0, 0.125)]
            radii = lower[0] + (np.array([lo, hi]) + origin[0] + 0.5) * dx[0]
            z = lower[1] + (np.array(planes) + origin[1] + 0.5) * dx[1]
            mu = picmi.constants.mu0 * (2 + 0.5 * z)
            currents = (
                2 * np.pi * (radii[1] * bt[hi, planes] - radii[0] * bt[lo, planes]) / mu
            )
        else:
            by = field("y")
            lo, hi = index(transverse_lo, 0), index(transverse_hi, 0)
            z = lower[1] + (np.array(planes) + origin[1]) * dx[1]
            mu = picmi.constants.mu0 * (2 + 0.5 * (z + 0.5 * dx[1]))
            currents = (by[hi, planes] - by[lo, planes]) / mu
        print(f"independent level {level} {patch} currents: {currents}")
        np.testing.assert_allclose(currents, 2000.0, rtol=2.0e-10, atol=0)
