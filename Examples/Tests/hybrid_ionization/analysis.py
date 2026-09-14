#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Validate deterministic hybrid He+ -> He2+ charge and energy coupling."""

import argparse
from pathlib import Path

import numpy as np
import yt

yt.funcs.mylog.setLevel(0)


# Exact SI definitions (epsilon_0 is the 2018 CODATA value).  Keeping these
# local leaves the fast regression dependent only on NumPy and yt.
Boltzmann = 1.380649e-23
c = 299792458.0
elementary_charge = 1.602176634e-19
epsilon_0 = 8.8541878128e-12


FIELD_NAMES = (
    "rho",
    "rho_ions",
    "Te",
    "Pe",
    "hybrid_ionization_electron_source_fp",
    "hybrid_ionization_binding_energy_fp",
)


def load_snapshot(path: Path, electromagnetic_fields: tuple[str, ...]):
    dataset = yt.load(str(path))
    grid = dataset.covering_grid(
        level=0,
        left_edge=dataset.domain_left_edge,
        dims=dataset.domain_dimensions,
    )
    fields = {
        name: np.asarray(grid["boxlib", name].v, dtype=np.float64).squeeze()
        for name in FIELD_NAMES + electromagnetic_fields
    }
    particles = dataset.all_data()
    particle_ids = np.asarray(particles["ions", "particle_id"].v)
    order = np.argsort(particle_ids)

    def particle_field(name: str) -> np.ndarray:
        return np.asarray(particles["ions", name].v)[order]

    particle_data = {
        "id": particle_ids[order],
        "weight": particle_field("particle_weight"),
        "ionization_level": particle_field("particle_ionizationLevel"),
        "sentinel": particle_field("particle_sentinel"),
    }
    for stem in ("position", "momentum"):
        for direction in ("x", "y", "z"):
            name = f"particle_{stem}_{direction}"
            try:
                particle_data[name] = particle_field(name)
            except yt.utilities.exceptions.YTFieldNotFound:
                # Plotfiles map reduced-dimensional coordinates to the axes
                # they carry; compare every component that is present.
                pass
    return fields, particle_data


parser = argparse.ArgumentParser()
parser.add_argument("--geometry", choices=("cartesian", "rcylinder"), required=True)
parser.add_argument("--precision", choices=("SINGLE", "DOUBLE"), default="DOUBLE")
parser.add_argument(
    "--particle-precision", choices=("SINGLE", "DOUBLE"), default="DOUBLE"
)
parser.add_argument("--compare-reference", type=Path)
args = parser.parse_args()

if args.precision == "SINGLE":
    state_rtol = 6.0e-5
    charge_rtol = 5.0e-5
    ledger_rtol = 1.2e-4
else:
    # Te and Pe pass through one QDSMC gather/scatter.  The deposited charge
    # and explicit source ledgers retain tighter, deposition-level bounds.
    state_rtol = 2.0e-8
    charge_rtol = 3.0e-11
    ledger_rtol = 2.0e-9
particle_rtol = 3.0e-6 if args.particle_precision == "SINGLE" else 3.0e-13

n0 = 1.0e20
temperature_0_eV = 200.0
gamma = 5.0 / 3.0
ionization_energy_eV = 54.4177650
temperature_1_eV = (temperature_0_eV - (gamma - 1.0) * ionization_energy_eV) / 2.0
temperature_0 = temperature_0_eV * elementary_charge / Boltzmann
temperature_1 = temperature_1_eV * elementary_charge / Boltzmann

if args.geometry == "cartesian":
    electromagnetic_fields = (
        "Ex",
        "Ey",
        "Ez",
        "Bx",
        "By",
        "Bz",
        "jx",
        "jy",
        "jz",
    )
    dx = 1.0e-3
    volumes = np.full((4, 4), dx**2)
    expected_ion_number = n0 * (4.0 * dx) ** 2
    core = np.ones((4, 4), dtype=bool)
    expected_source_support = np.ones((4, 4), dtype=bool)
else:
    electromagnetic_fields = ("Er", "Et", "Ez", "Br", "Bt", "Bz")
    dr = 1.0e-3
    edges = np.arange(13, dtype=np.float64) * dr
    volumes = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    expected_ion_number = n0 * np.pi * ((10.0 * dr) ** 2 - (2.0 * dr) ** 2)
    core = np.zeros(12, dtype=bool)
    core[3:9] = True
    # Shape-1 deposition from particles in cells 2:10 reaches one neighboring
    # diagnostic cell on either side after nodal-to-cell centering.
    expected_source_support = np.zeros(12, dtype=bool)
    expected_source_support[1:11] = True

initial_path = Path("diags/diag000001")
if args.compare_reference is not None:
    # A restarted run begins after step 1 and therefore writes only step 2.
    # Use the uninterrupted pre-event plot that produced its checkpoint.
    initial_path = args.compare_reference.with_name("diag000001")
initial_fields, initial_particles = load_snapshot(initial_path, electromagnetic_fields)
final_fields, final_particles = load_snapshot(
    Path("diags/diag000002"), electromagnetic_fields
)

for snapshot in (initial_fields, final_fields):
    for name in FIELD_NAMES + electromagnetic_fields:
        assert snapshot[name].shape == volumes.shape, (name, snapshot[name].shape)
        assert np.all(np.isfinite(snapshot[name])), name

# The user integer component precedes the operator-owned charge state.  Both
# must survive the update by name, while every physical particle quantity is
# unchanged.
np.testing.assert_array_equal(initial_particles["id"], final_particles["id"])
np.testing.assert_array_equal(initial_particles["sentinel"], 37)
np.testing.assert_array_equal(final_particles["sentinel"], 37)
np.testing.assert_array_equal(initial_particles["ionization_level"], 1)
np.testing.assert_array_equal(final_particles["ionization_level"], 2)
np.testing.assert_allclose(
    final_particles["weight"],
    initial_particles["weight"],
    rtol=particle_rtol,
    atol=0.0,
)
for name, initial_value in initial_particles.items():
    if not name.startswith("particle_position_"):
        continue
    np.testing.assert_allclose(
        final_particles[name], initial_value, rtol=particle_rtol, atol=1.0e-30
    )
initial_particle_momentum = []
final_particle_momentum = []
for direction in ("x", "y", "z"):
    name = f"particle_momentum_{direction}"
    if name not in initial_particles:
        continue
    np.testing.assert_allclose(
        final_particles[name],
        initial_particles[name],
        rtol=particle_rtol,
        atol=1.0e-40,
    )
    initial_particle_momentum.append(
        np.sum(initial_particles["weight"] * initial_particles[name])
    )
    final_particle_momentum.append(
        np.sum(final_particles["weight"] * final_particles[name])
    )
np.testing.assert_allclose(
    final_particle_momentum,
    initial_particle_momentum,
    rtol=particle_rtol,
    atol=1.0e-40,
)
if args.geometry == "cartesian":
    assert initial_particle_momentum[0] > 0.0
    np.testing.assert_array_equal(initial_particle_momentum[1:], 0.0)
else:
    np.testing.assert_array_equal(initial_particle_momentum, 0.0)

particle_number = np.sum(initial_particles["weight"], dtype=np.float64)
np.testing.assert_allclose(
    particle_number, expected_ion_number, rtol=particle_rtol, atol=0.0
)
particle_charge_initial = np.sum(
    initial_particles["weight"] * initial_particles["ionization_level"],
    dtype=np.float64,
)
particle_charge_final = np.sum(
    final_particles["weight"] * final_particles["ionization_level"],
    dtype=np.float64,
)
np.testing.assert_allclose(
    particle_charge_initial, expected_ion_number, rtol=particle_rtol
)
np.testing.assert_allclose(
    particle_charge_final, 2.0 * expected_ion_number, rtol=particle_rtol
)

rho_initial = initial_fields["rho"]
rho_final = final_fields["rho"]
source_initial = initial_fields["hybrid_ionization_electron_source_fp"]
source_final = final_fields["hybrid_ionization_electron_source_fp"]
binding_initial = initial_fields["hybrid_ionization_binding_energy_fp"]
binding_final = final_fields["hybrid_ionization_binding_energy_fp"]

np.testing.assert_array_equal(source_initial, 0.0)
np.testing.assert_array_equal(binding_initial, 0.0)
np.testing.assert_array_equal(source_final[~expected_source_support], 0.0)
assert np.all(source_final[expected_source_support] > 0.0)
np.testing.assert_allclose(
    source_final,
    (rho_final - rho_initial) / elementary_charge,
    rtol=charge_rtol,
    atol=charge_rtol * n0,
)
np.testing.assert_allclose(
    binding_final,
    source_final * ionization_energy_eV * elementary_charge,
    rtol=charge_rtol,
    atol=charge_rtol * n0 * ionization_energy_eV * elementary_charge,
)

# Dynamic charge deposition must agree independently in total rho and the
# requested per-species rho field.  The electron source closes both the grid
# and particle charge-state changes.
for fields in (initial_fields, final_fields):
    np.testing.assert_allclose(
        fields["rho_ions"], fields["rho"], rtol=charge_rtol, atol=0.0
    )
grid_charge_initial = np.sum(rho_initial / elementary_charge * volumes)
grid_charge_final = np.sum(rho_final / elementary_charge * volumes)
liberated_electrons = np.sum(source_final * volumes)
np.testing.assert_allclose(grid_charge_initial, expected_ion_number, rtol=charge_rtol)
np.testing.assert_allclose(
    grid_charge_final, 2.0 * expected_ion_number, rtol=charge_rtol
)
np.testing.assert_allclose(liberated_electrons, expected_ion_number, rtol=charge_rtol)
np.testing.assert_allclose(
    grid_charge_final - grid_charge_initial,
    liberated_electrons,
    rtol=charge_rtol,
)
np.testing.assert_allclose(
    grid_charge_initial, particle_charge_initial, rtol=charge_rtol + particle_rtol
)
np.testing.assert_allclose(
    grid_charge_final, particle_charge_final, rtol=charge_rtol + particle_rtol
)

# The moving Cartesian ions exercise the dynamic-Z current path independently
# of charge deposition.  MultipleParticles stores normalized proper velocity
# u/c; direct current deposition uses v=c*u/sqrt(1+u^2).  Uniform periodic
# deposition is exact after cell centering.
if args.geometry == "cartesian":
    normalized_proper_velocity = 1.0e-3
    velocity = (
        c * normalized_proper_velocity / np.sqrt(1.0 + normalized_proper_velocity**2)
    )
    np.testing.assert_allclose(
        initial_fields["jx"], rho_initial * velocity, rtol=state_rtol
    )
    np.testing.assert_allclose(
        final_fields["jx"], rho_final * velocity, rtol=state_rtol
    )
    np.testing.assert_allclose(
        final_fields["jx"], 2.0 * initial_fields["jx"], rtol=charge_rtol
    )
    np.testing.assert_array_equal(initial_fields["jy"], 0.0)
    np.testing.assert_array_equal(initial_fields["jz"], 0.0)
    np.testing.assert_array_equal(final_fields["jy"], 0.0)
    np.testing.assert_array_equal(final_fields["jz"], 0.0)

# In the uniform Cartesian plasma and the eroded constant-density RCYL core,
# the exact ideal-gas update is
#   n0*T0/(gamma-1) = 2*n0*T1/(gamma-1) + n0*I.
np.testing.assert_allclose(initial_fields["Te"][core], temperature_0, rtol=state_rtol)
np.testing.assert_allclose(final_fields["Te"][core], temperature_1, rtol=state_rtol)
np.testing.assert_allclose(
    initial_fields["Pe"][core],
    rho_initial[core] / elementary_charge * Boltzmann * initial_fields["Te"][core],
    rtol=state_rtol,
)
np.testing.assert_allclose(
    final_fields["Pe"][core],
    rho_final[core] / elementary_charge * Boltzmann * final_fields["Te"][core],
    rtol=state_rtol,
)

binding_energy = np.sum(binding_final * volumes)
expected_binding_energy = expected_ion_number * ionization_energy_eV * elementary_charge
electron_energy_change = np.sum(
    (final_fields["Pe"] - initial_fields["Pe"]) / (gamma - 1.0) * volumes
)
np.testing.assert_allclose(binding_energy, expected_binding_energy, rtol=ledger_rtol)
np.testing.assert_allclose(electron_energy_change, -binding_energy, rtol=ledger_rtol)

# No material particle is created or kicked.  In this manufactured symmetry B
# stays identically zero, so the electromagnetic momentum integral epsilon_0
# int(E x B)dV is also exactly zero before and after ionization.
if args.geometry == "cartesian":
    e_vectors = [
        [initial_fields[name] for name in ("Ex", "Ey", "Ez")],
        [final_fields[name] for name in ("Ex", "Ey", "Ez")],
    ]
    b_vectors = [
        [initial_fields[name] for name in ("Bx", "By", "Bz")],
        [final_fields[name] for name in ("Bx", "By", "Bz")],
    ]
else:
    e_vectors = [
        [initial_fields[name] for name in ("Er", "Et", "Ez")],
        [final_fields[name] for name in ("Er", "Et", "Ez")],
    ]
    b_vectors = [
        [initial_fields[name] for name in ("Br", "Bt", "Bz")],
        [final_fields[name] for name in ("Br", "Bt", "Bz")],
    ]
for electric, magnetic in zip(e_vectors, b_vectors):
    for component in magnetic:
        np.testing.assert_array_equal(component, 0.0)
    momentum_density = epsilon_0 * np.cross(
        np.moveaxis(np.asarray(electric), 0, -1),
        np.moveaxis(np.asarray(magnetic), 0, -1),
    )
    spatial_axes = tuple(range(volumes.ndim))
    field_momentum = np.sum(
        momentum_density * volumes[..., np.newaxis], axis=spatial_axes
    )
    np.testing.assert_array_equal(field_momentum, 0.0)

if args.compare_reference is not None:
    reference_fields, reference_particles = load_snapshot(
        args.compare_reference, electromagnetic_fields
    )
    restart_rtol = 4.0e-6 if args.precision == "SINGLE" else 3.0e-12
    restart_particle_rtol = 4.0e-6 if args.particle_precision == "SINGLE" else 3.0e-13
    for name in FIELD_NAMES + electromagnetic_fields:
        if np.count_nonzero(reference_fields[name]) == 0:
            np.testing.assert_array_equal(final_fields[name], 0.0)
        else:
            np.testing.assert_allclose(
                final_fields[name],
                reference_fields[name],
                rtol=restart_rtol,
                atol=0.0,
            )
    for name, reference_value in reference_particles.items():
        if name in ("id", "ionization_level", "sentinel"):
            np.testing.assert_array_equal(final_particles[name], reference_value)
        else:
            np.testing.assert_allclose(
                final_particles[name],
                reference_value,
                rtol=restart_particle_rtol,
                atol=1.0e-40,
            )

print(
    f"{args.geometry} hybrid He ionization: "
    f"Nion={expected_ion_number:.16e}, "
    f"T1={temperature_1_eV:.9f} eV, "
    f"binding={binding_energy:.16e} J, "
    f"electron+binding residual={electron_energy_change + binding_energy:.6e} J"
)
