#!/usr/bin/env python3
"""Check raw refined sources at negative radius before the physical axis fold."""

import numpy as np

from pywarpx import libwarpx, picmi

grid = picmi.CylindricalGrid(
    number_of_cells=[32, 32],
    lower_bound=[0, -0.5],
    upper_bound=[1, 0.5],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    refined_regions=[[1, [0, -0.5], [0.5, 0.5]]],
    warpx_blocking_factor=16,
    warpx_max_grid_size=16,
)
sim = picmi.Simulation(
    solver=picmi.ElectrostaticSolver(grid=grid, method="Multigrid"),
    max_steps=1,
    time_step_size=1e-12,
    particle_shape="cubic",
    warpx_use_filter=0,
)
# A particle shape supplies the native current and charge guard widths.
sim.add_species(
    picmi.Species(
        particle_type="electron",
        name="electrons",
        initial_distribution=picmi.UniformDistribution(density=0.0),
    ),
    layout=picmi.GriddedLayout(n_macroparticle_per_cell=[1, 2, 1], grid=grid),
)
sim.initialize_inputs()
sim.initialize_warpx()
fields = [("current", {"dir": d}) for d in ("r", "theta", "z")] + [("rho", {})]
for name, direction in fields:
    fine = sim.fields.get(name + "_fp", level=1, **direction)
    assert fine.n_grow_vect[0] >= 2
    # Model a shape depositing onto either side of r=0. Include both sides
    # of an axial box boundary and avoid initializing overlapping nodal copies.
    for mfi in fine:
        valid, fab = mfi.validbox(), mfi.fabbox()
        if valid.small_end[0] != 0:
            continue
        data = fine.array(mfi).to_xp(copy=False, order="F")
        for z in (15, 17, 31, 33):
            if valid.small_end[1] <= z <= valid.big_end[1]:
                for r in (-2, 2):
                    data[r - fab.small_end[0], z - fab.small_end[1], 0, :] = z + r

wx = libwarpx.libwarpx_so.get_instance()
wx.sync_current("current_fp")
wx.sync_rho()
for name, direction in fields:
    parent = sim.fields.get(name + "_fp", level=0, **direction)
    patch = sim.fields.get(name + "_cp", level=1, **direction)
    for i in (-1j, 1):
        reference = np.asarray(patch[i, 4:20, ...])
        assert np.max(np.abs(reference)) > 0
        np.testing.assert_allclose(
            np.asarray(parent[i, 4:20, ...]), reference, rtol=1e-6, atol=0
        )
libwarpx.finalize()
