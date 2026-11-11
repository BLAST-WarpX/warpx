"""Start and advance an empty conforming slab through the public PICMI solver."""

from pathlib import Path

from pywarpx import picmi, warpx

grid = picmi.Cartesian2DGrid(
    number_of_cells=[48, 8],
    lower_bound=[0, 0],
    upper_bound=[0.096, 0.016],
    lower_boundary_conditions=["periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic"],
    refined_regions=[[1, [0.032, 0], [0.064, 0.016], [2, 2]]],
    warpx_max_grid_size=128,
    warpx_blocking_factor=2,
)
solver = picmi.ElectromagneticSolver(
    grid=grid,
    method="Yee",
    warpx_composite_xz_interface=True,
    warpx_composite_xz_fixed_shape=True,
)
sim = picmi.Simulation(
    solver=solver, max_steps=2, time_step_size=1.2e-12, warpx_use_filter=False
)
warpx.n_field_gather_buffer = 0
warpx.n_current_deposition_buffer = 0
warpx.do_initial_div_cleaning = False
sim.write_input_file(file_name="inputs")
deck = Path("inputs").read_text()
assert "warpx.composite_xz_fixed_shape = 1" in deck
assert "warpx.composite_xz_interface = 1" in deck
sim.step(2)
assert sim.extension.warpx.getistep(0) == 2
