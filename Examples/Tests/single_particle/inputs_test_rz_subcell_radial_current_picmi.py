"""Check the radial current from subcell displacements of two separated protons."""

import numpy as np

from pywarpx import picmi

constants = picmi.constants
nr, nz = 64, 32
dr = dz = 1.0e-3
dt = 1.0e-12
radius = np.array([0.04132, 0.0471])
theta = np.array([0.0, 0.7])
ur = np.array([5.0e4, -5.0e4])
weight = 1.0e9

grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz],
    n_azimuthal_modes=1,
    lower_bound=[0.0, 0.0],
    upper_bound=[nr * dr, nz * dz],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee")
dist = picmi.ParticleListDistribution(
    x=radius * np.cos(theta),
    y=radius * np.sin(theta),
    z=[0.008, 0.024],
    ux=ur * np.cos(theta),
    uy=ur * np.sin(theta),
    uz=[0.0, 0.0],
    weight=[weight, weight],
)
sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=1,
    particle_shape="cubic",
    warpx_current_deposition_algo="esirkepov",
    warpx_use_filter=0,
    verbose=0,
)
ions = picmi.Species(particle_type="proton", name="ions", initial_distribution=dist)
sim.add_species(ions, layout=None)
sim.step(1)

# J_r is centered in r and nodal in z. Exclude the duplicate periodic endpoint.
jr = np.squeeze(np.asarray(sim.fields.get("current_fp", dir="r", level=0)[...]))
volumes = 2 * np.pi * (np.arange(nr) + 0.5) * dr**2 * dz
current = np.array(
    [np.sum(volumes[:, None] * slab) for slab in np.split(jr[:, :nz], 2, axis=1)]
)
expected = constants.q_e * weight * ur / np.sqrt(1 + (ur / constants.c) ** 2)
rtol = max(1.0e-9, 8 * np.finfo(jr.dtype).eps)
np.testing.assert_allclose(current, expected, rtol=rtol, atol=0)
