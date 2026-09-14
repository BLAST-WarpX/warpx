#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check geometry-aware kinetic-electron heating in a cylindrical cold ring."""

from pathlib import Path

import numpy as np
import yt
from read_raw_data import _read_buffer
from scipy.constants import c, m_e

DR = 1.0e-3
U0 = 0.1


def plotfiles() -> tuple[Path, Path]:
    files = sorted(
        path
        for path in Path("diags").glob("diag1*")
        if path.is_dir() and ".old." not in path.name
    )
    assert len(files) == 2, f"expected two clean plotfiles, found {files}"
    return files[0], files[1]


def load_fields(path: Path) -> dict[str, np.ndarray]:
    with (path / "Header").open() as header:
        header.readline()
        num_fields = int(header.readline())
        names = [header.readline().strip() for _ in range(num_fields)]
    values = _read_buffer(str(path), str(path / "Level_0" / "Cell_H"), names)
    return {
        name: np.asarray(value, dtype=np.float64).squeeze()
        for name, value in values.items()
    }


def load_reduced(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open() as stream:
        labels = stream.readline().split()
    return labels, np.atleast_2d(np.loadtxt(path))


def load_particle_moments(path: Path) -> dict[str, np.ndarray | float]:
    data = yt.load(str(path)).all_data()
    position_x = np.asarray(
        data["electrons", "particle_position_x"].to_value("m"), dtype=np.float64
    )
    theta = np.asarray(data["electrons", "particle_theta"].v, dtype=np.float64)
    weight = np.asarray(data["electrons", "particle_weight"].v, dtype=np.float64)
    particle_id = np.asarray(data["electrons", "particle_id"].v, dtype=np.int64)
    particle_cpu = np.asarray(data["electrons", "particle_cpu"].v, dtype=np.int64)
    momentum = np.stack(
        [
            np.asarray(
                data["electrons", f"particle_momentum_{axis}"].to_value("kg*m/s"),
                dtype=np.float64,
            )
            for axis in "xyz"
        ],
        axis=1,
    )

    assert position_x.size == theta.size == weight.size == momentum.shape[0] == 2
    np.testing.assert_allclose(position_x, 2.5 * DR, rtol=0.0, atol=1.0e-14 * DR)
    np.testing.assert_allclose(np.sort(theta), [0.0, np.pi], rtol=0.0, atol=1.0e-12)
    assert np.all(weight > 0.0)
    np.testing.assert_allclose(weight[0], weight[1], rtol=1.0e-13)
    identity = np.stack((particle_id, particle_cpu), axis=1)
    assert np.unique(identity, axis=0).shape[0] == 2
    if np.unique(particle_cpu).size > 1:
        assert particle_id[0] == particle_id[1], (
            "MPI fixture must exercise duplicate bare particle IDs"
        )

    px, py, pz = momentum.T
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    proper_velocity = momentum / (m_e * c)
    local_0 = proper_velocity[:, 0] * cos_theta + proper_velocity[:, 1] * sin_theta
    local_1 = -proper_velocity[:, 0] * sin_theta + proper_velocity[:, 1] * cos_theta
    local_2 = proper_velocity[:, 2]
    local = np.stack((local_0, local_1, local_2), axis=1)
    total_weight = np.sum(weight)
    bulk = np.sum(weight[:, None] * local, axis=0) / total_weight

    momentum_squared = px * px + py * py + pz * pz
    particle_energy = (
        momentum_squared
        / m_e
        / (1.0 + np.sqrt(1.0 + momentum_squared / (m_e * c) ** 2))
    )
    total_energy = np.sum(weight * particle_energy)
    bulk_momentum = m_e * c * bulk
    bulk_momentum_squared = np.sum(bulk_momentum * bulk_momentum)
    cold_energy = (
        total_weight
        * bulk_momentum_squared
        / m_e
        / (1.0 + np.sqrt(1.0 + bulk_momentum_squared / (m_e * c) ** 2))
    )
    thermal_variance = np.sum(weight[:, None] * (local - bulk) ** 2) / total_weight
    return {
        "bulk": bulk,
        "total_energy": total_energy,
        "internal_energy": total_energy - cold_energy,
        "thermal_variance": thermal_variance,
        "identity": identity,
    }


initial_path, final_path = plotfiles()
initial_fields = load_fields(initial_path)
final_fields = load_fields(final_path)
assert initial_fields["radiation_material_energy"].shape == (4,)
assert final_fields["radiation_material_energy"].shape == (4,)

initial_particles = load_particle_moments(initial_path)
final_particles = load_particle_moments(final_path)
np.testing.assert_allclose(
    initial_particles["bulk"], [U0, 0.0, 0.0], rtol=2.0e-11, atol=2.0e-12
)
np.testing.assert_allclose(
    final_particles["bulk"], initial_particles["bulk"], rtol=2.0e-11, atol=2.0e-12
)
assert initial_particles["thermal_variance"] < 1.0e-20
assert final_particles["thermal_variance"] > 0.0

material_gain = np.sum(
    final_fields["radiation_material_energy"]
    - initial_fields["radiation_material_energy"]
)
assert material_gain > 0.0
assert (
    abs(initial_particles["internal_energy"])
    < 1.0e-10 * initial_particles["total_energy"]
)

particle_labels, particle_energy = load_reduced(Path("diags/particle_energy.txt"))
_, radiation_energy = load_reduced(Path("diags/radiation_energy.txt"))
assert radiation_energy.shape[1] >= 17
electron_column = next(
    index
    for index, label in enumerate(particle_labels)
    if label.endswith("electrons(J)")
)
photon_column = next(
    index for index, label in enumerate(particle_labels) if label.endswith("photons(J)")
)
absorbed_energy = particle_energy[0, photon_column] - particle_energy[-1, photon_column]
electron_gain = (
    particle_energy[-1, electron_column] - particle_energy[0, electron_column]
)
residual = radiation_energy[-1, -2]
cumulative_residual = radiation_energy[-1, -1]

assert absorbed_energy > 0.0
assert particle_energy[-1, photon_column] > 0.0
np.testing.assert_allclose(
    final_particles["internal_energy"], material_gain, rtol=3.0e-9, atol=1.0e-18
)

precision = np.float64
field_roundoff = (
    32.0
    * np.finfo(precision).eps
    * max(
        abs(radiation_energy[0, 2]),
        abs(radiation_energy[-1, 2]),
        abs(material_gain),
        abs(radiation_energy[-1, 5]),
        abs(radiation_energy[-1, 6]),
    )
)
particle_roundoff = (
    32.0
    * np.finfo(precision).eps
    * max(
        abs(particle_energy[0, electron_column]),
        abs(particle_energy[-1, electron_column]),
        abs(particle_energy[0, photon_column]),
        abs(particle_energy[-1, photon_column]),
        abs(final_particles["internal_energy"]),
        abs(material_gain),
    )
)
field_tolerance = max(1.0e-24, field_roundoff)
particle_tolerance = max(1.0e-24, particle_roundoff)
np.testing.assert_allclose(
    electron_gain, material_gain, rtol=0.0, atol=particle_tolerance
)
np.testing.assert_allclose(
    final_particles["internal_energy"], material_gain, rtol=0.0, atol=particle_tolerance
)
np.testing.assert_allclose(
    radiation_energy[-1, 5], material_gain, rtol=0.0, atol=field_tolerance
)
np.testing.assert_allclose(
    radiation_energy[-1, 6], material_gain, rtol=0.0, atol=field_tolerance
)
np.testing.assert_allclose(
    radiation_energy[-1, 9], material_gain, rtol=0.0, atol=field_tolerance
)
np.testing.assert_allclose(
    cumulative_residual, residual, rtol=0.0, atol=field_tolerance
)
closure_residual = radiation_energy[0, 2] - radiation_energy[-1, 2] - material_gain
np.testing.assert_allclose(residual, closure_residual, rtol=0.0, atol=field_tolerance)
photon_residual = absorbed_energy - material_gain
np.testing.assert_allclose(photon_residual, residual, rtol=0.0, atol=field_tolerance)
electron_radiation_drift = (
    electron_gain + radiation_energy[-1, 2] - radiation_energy[0, 2]
)
np.testing.assert_allclose(
    electron_radiation_drift, -residual, rtol=0.0, atol=particle_tolerance
)

print(f"absorbed photon energy: {absorbed_energy:.16e} J")
print(f"material energy gain:   {material_gain:.16e} J")
print(f"electron energy gain:   {electron_gain:.16e} J")
print(f"initial local bulk:     {initial_particles['bulk']}")
print(f"final local bulk:       {final_particles['bulk']}")
print(f"final thermal variance: {final_particles['thermal_variance']:.16e}")
print(f"final local internal:   {final_particles['internal_energy']:.16e} J")
print(f"initial (id, cpu):      {initial_particles['identity']}")
print(f"kinetic numerical residual: {residual:.16e} J")
