#!/usr/bin/env python3

# from warp import picmi
import argparse

from pywarpx import picmi

parser = argparse.ArgumentParser(description="Gaussian beam PICMI example")

parser.add_argument(
    "--diagformat",
    type=str,
    help="Format of the full diagnostics (plotfile, openpmd, ascent, sensei, ...)",
    default="plotfile",
)
parser.add_argument(
    "--fields_to_plot",
    type=str,
    help="List of fields to write to diagnostics",
    default=["E", "B", "J", "part_per_cell"],
    nargs="*",
)

args = parser.parse_args()

constants = picmi.constants

nx = 32
ny = 32
nz = 32

xmin = -2.0
xmax = +2.0
ymin = -2.0
ymax = +2.0
zmin = -2.0
zmax = +2.0

number_sim_particles = 32768
total_charge = 8.010883097437485e-07

beam_rms_size = 0.25
electron_beam_divergence = -0.04 * constants.c

em_order = 3

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[xmin, ymin, zmin],
    upper_bound=[xmax, ymax, zmax],
    lower_boundary_conditions=["periodic", "periodic", "open"],
    upper_boundary_conditions=["periodic", "periodic", "open"],
    lower_boundary_conditions_particles=["periodic", "periodic", "absorbing"],
    upper_boundary_conditions_particles=["periodic", "periodic", "absorbing"],
    warpx_max_grid_size=16,
)

solver = picmi.ElectromagneticSolver(
    grid=grid, cfl=1.0, stencil_order=[em_order, em_order, em_order]
)

electron_beam = picmi.GaussianBunchDistribution(
    n_physical_particles=total_charge / constants.q_e,
    rms_bunch_size=[beam_rms_size, beam_rms_size, beam_rms_size],
    velocity_divergence=[
        electron_beam_divergence,
        electron_beam_divergence,
        electron_beam_divergence,
    ],
)

proton_beam = picmi.GaussianBunchDistribution(
    n_physical_particles=total_charge / constants.q_e,
    rms_bunch_size=[beam_rms_size, beam_rms_size, beam_rms_size],
)

electrons = picmi.Species(
    particle_type="electron", name="electrons", initial_distribution=electron_beam
)
protons = picmi.Species(
    particle_type="proton", name="protons", initial_distribution=proton_beam
)

field_diag1 = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=10,
    data_list=args.fields_to_plot,
    warpx_format=args.diagformat,
)

part_diag1 = picmi.ParticleDiagnostic(
    name="diag1",
    period=10,
    species=[electrons, protons],
    data_list=["weighting", "momentum"],
    warpx_format=args.diagformat,
)

sim = picmi.Simulation(
    solver=solver,
    max_steps=10,
    verbose=1,
    warpx_current_deposition_algo="direct",
    warpx_use_filter=0,
)

sim.add_species(
    electrons, layout=picmi.PseudoRandomLayout(n_macroparticles=number_sim_particles)
)
sim.add_species(
    protons, layout=picmi.PseudoRandomLayout(n_macroparticles=number_sim_particles)
)

sim.add_diagnostic(field_diag1)
sim.add_diagnostic(part_diag1)

# write_inputs will create an inputs file that can be used to run
# with the compiled version.
# sim.write_input_file(file_name = 'inputs_from_PICMI')

# Alternatively, sim.step will run WarpX, controlling it from Python
sim.step()
