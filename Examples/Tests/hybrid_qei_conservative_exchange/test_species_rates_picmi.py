#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Check material-energy input wiring and parser-constant name mangling."""

import pywarpx
from pywarpx import picmi

grid = picmi.CylindricalGrid(
    number_of_cells=[8, 8],
    lower_bound=[0, 0],
    upper_bound=[1, 1],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
)
pywarpx.my_constants.add_keywords({"rate_scale": 1})
solver = picmi.HybridPICSolver(
    grid=grid,
    Te=100,
    solve_electron_energy_equation=True,
    electron_ion_relaxation_rate=7,
    electron_ion_relaxation_rate_species={"shell": "rate_scale*Te", "foam": 0},
    electron_energy_transport="finite_volume_implicit",
    conservative_pressure_work=True,
    conservative_pressure_work_pec=True,
    electron_heat_conduction=True,
    electron_thermal_conductivity="rate_scale*Te^2.5",
    electron_conduction_flux_limiter=0,
    electron_conduction_max_substeps=64,
    electron_conduction_verbosity=0,
    rate_scale=2,
)
solver.solver_initialize_inputs()
inputs = pywarpx.hybridpicmodel.argvattrs
assert inputs["electron_ion_relaxation_species"] == ["shell", "foam"]
assert inputs["electron_ion_relaxation_rate(rho,Te,Ti,t)"] == "7"
assert inputs["electron_ion_relaxation_rate_foam(rho_s,rho,Te,Ti,t)"] == "0"
assert solver.mangle_dict["rate_scale"] != "rate_scale"
assert inputs["electron_ion_relaxation_rate_shell(rho_s,rho,Te,Ti,t)"] == (
    solver.mangle_dict["rate_scale"] + "*Te"
)
for name, expected in {
    "electron_energy_transport": "finite_volume_implicit",
    "conservative_pressure_work": True,
    "conservative_pressure_work_pec": True,
    "electron_heat_conduction": True,
    "electron_conduction_flux_limiter": 0,
    "electron_conduction_max_substeps": 64,
    "electron_conduction_verbosity": 0,
}.items():
    assert inputs[name] == expected
assert inputs["electron_thermal_conductivity(rho,Te)"] == (
    solver.mangle_dict["rate_scale"] + "*Te^2.5"
)

# Omitted controls must not emit accidental defaults or parser constants.
pywarpx.hybridpicmodel.clear()
picmi.HybridPICSolver(grid=grid, Te=100).solver_initialize_inputs()
assert not any("conduction" in name or "pressure_work" in name for name in inputs)
assert "electron_energy_transport" not in inputs
assert "electron_ion_relaxation_species" not in inputs
print("Material-energy PICMI input and constant-mangling contract PASS")
