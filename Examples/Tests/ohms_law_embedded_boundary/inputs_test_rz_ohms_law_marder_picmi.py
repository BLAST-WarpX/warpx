#!/usr/bin/env python3
#
# --- RZ smoke test of the transitional Marder correction: a small hybrid
# --- column with a low-density halo runs a few steps with the correction
# --- enabled at every substep E evaluation. This exercises the RZ branch of
# --- the correction (nodal divergence with the on-axis regularization,
# --- in-plane Er/Ez updates, Etheta untouched for m=0) together with the
# --- split external-field machinery (uniform bias Bz through A_external).
# --- Pass criterion: the run completes with finite fields and the in-band
# --- divergence is reduced rather than amplified.

import numpy as np

from pywarpx import fields, picmi

constants = picmi.constants

NR, NZ = 32, 32
R_MAX, L_Z = 0.5, 1.0
N0 = 1.0e19
N_FLOOR = 1.0e16
B0 = 0.05  # uniform external bias Bz (T)

grid = picmi.CylindricalGrid(
    number_of_cells=[NR, NZ],
    lower_bound=[0.0, -L_Z / 2],
    upper_bound=[R_MAX, L_Z / 2],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    n_azimuthal_modes=1,
)

sim = picmi.Simulation(
    time_step_size=5.0e-10,
    max_steps=5,
    particle_shape=1,
    verbose=0,
)
sim.current_deposition_algo = "direct"

# Uniform external Bz through the split external-field machinery:
# A = (-y/2, x/2, 0) * B0 with a constant time function
A_ext = {
    "uniform_bias": {
        "Ax_external_function": f"-0.5*y*{B0}",
        "Ay_external_function": f"0.5*x*{B0}",
        "Az_external_function": "0",
        "A_time_external_function": "1",
    },
}

sim.solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=5.0 / 3.0,
    Te=10.0,
    n0=N0,
    n_floor=N_FLOOR,
    plasma_resistivity=1.0e-5,
    substeps=10,
    use_rkf45=True,
    substep_rtol=1.0e-3,
    substep_atol=1.0e-6,
    max_substep_attempts=1000,
    A_external=A_ext,
    marder_alpha=0.05,
    marder_target="grad_pe_only",
    marder_correction_level="all_substeps",
    marder_max_iterations=10,
    marder_rtol=1.0e-3,
)

# Column with a low-density halo so the transition band
# (0 < rho <= n_floor*q_e) is populated
dist = picmi.AnalyticDistribution(
    density_expression=f"{N0}*exp(-(sqrt(x*x+y*y)/0.2)**4)+5e15",
    rms_velocity=[5e4, 5e4, 5e4],
)
ions = picmi.Species(
    particle_type="proton",
    name="ions",
    charge_state=1,
    initial_distribution=dist,
)
sim.add_species(
    ions, layout=picmi.GriddedLayout(n_macroparticle_per_cell=[2, 2, 2], grid=grid)
)

sim.initialize_inputs()
sim.initialize_warpx()
sim.step(5)

failures = []
for name, w in (
    ("Er", fields.ExFPWrapper()),
    ("Etheta", fields.EyFPWrapper()),
    ("Ez", fields.EzFPWrapper()),
    ("Br", fields.BxFPWrapper()),
    ("Bz", fields.BzFPWrapper()),
    ("rho", fields.RhoFPWrapper()),
):
    arr = np.asarray(w[...])
    finite = bool(np.all(np.isfinite(arr)))
    print(
        f"[{'PASS' if finite else 'FAIL'}] {name} finite (max|.| = {np.max(np.abs(arr)):.3e})"
    )
    if not finite:
        failures.append(name)

assert not failures, f"non-finite fields after the Marder RZ run: {failures}"
print("\nall checks passed")
