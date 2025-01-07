#!/usr/bin/env python3
# @Eya Dammak supervised by @Remi Lehe, 2024
# --- Input file for particle-boundary interaction testing in RZ.
# --- This input is a simple case of reflection
# --- of one electron on the surface of a sphere.
import numpy as np

from pywarpx import callbacks, particle_containers, picmi

##########################
# numerics parameters
##########################

dt = 1.0e-11

# --- Nb time steps

max_steps = 23
diagnostic_interval = 1

# --- grid

nr = 64
nz = 64

rmin = 0.0
rmax = 2
zmin = -2
zmax = 2

##########################
# numerics components
##########################

grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz],
    n_azimuthal_modes=1,
    lower_bound=[rmin, zmin],
    upper_bound=[rmax, zmax],
    lower_boundary_conditions=["none", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["none", "reflecting"],
    upper_boundary_conditions_particles=["absorbing", "reflecting"],
)


solver = picmi.ElectrostaticSolver(
    grid=grid, method="Multigrid", warpx_absolute_tolerance=1e-7
)

embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function="-(x**2+y**2+z**2-radius**2)", radius=0.2
)

##########################
# physics components
##########################

# one particle
e_dist = picmi.ParticleListDistribution(
    x=0.0, y=0.0, z=-0.25, ux=0.5e10, uy=0.0, uz=1.0e10, weight=1
)

electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    initial_distribution=e_dist,
    warpx_save_particles_at_eb=1,
)

##########################
# diagnostics
##########################

field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=diagnostic_interval,
    data_list=["Er", "Ez", "phi", "rho", "rho_electrons"],
    warpx_format="openpmd",
)

part_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=diagnostic_interval,
    species=[electrons],
    warpx_format="openpmd",
)

##########################
# simulation setup
##########################

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=max_steps,
    warpx_embedded_boundary=embedded_boundary,
    warpx_amrex_the_arena_is_managed=1,
)

sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(n_macroparticle_per_cell=[10, 1, 1], grid=grid),
)
sim.add_diagnostic(part_diag)
sim.add_diagnostic(field_diag)

sim.initialize_inputs()
sim.initialize_warpx()

##########################
# python particle data access
##########################


def concat(list_of_arrays):
    if len(list_of_arrays) == 0:
        # Return a 1d array of size 0
        return np.empty(0)
    else:
        return np.concatenate(list_of_arrays)


def mirror_reflection():
    buffer = particle_containers.ParticleBoundaryBufferWrapper()  # boundary buffer

    # STEP 1: extract the different parameters of the boundary buffer (normal, time, position)
    lev = 0  # level 0 (no mesh refinement here)
    delta_t = concat(
        buffer.get_particle_boundary_buffer("electrons", "eb", "deltaTimeScraped", lev)
    )
    r = concat(buffer.get_particle_boundary_buffer("electrons", "eb", "x", lev))
    theta = concat(buffer.get_particle_boundary_buffer("electrons", "eb", "theta", lev))
    z = concat(buffer.get_particle_boundary_buffer("electrons", "eb", "z", lev))
    x = r * np.cos(theta)  # from RZ coordinates to 3D coordinates
    y = r * np.sin(theta)
    ux = concat(buffer.get_particle_boundary_buffer("electrons", "eb", "ux", lev))
    uy = concat(buffer.get_particle_boundary_buffer("electrons", "eb", "uy", lev))
    uz = concat(buffer.get_particle_boundary_buffer("electrons", "eb", "uz", lev))
    w = concat(buffer.get_particle_boundary_buffer("electrons", "eb", "w", lev))
    nx = concat(buffer.get_particle_boundary_buffer("electrons", "eb", "nx", lev))
    ny = concat(buffer.get_particle_boundary_buffer("electrons", "eb", "ny", lev))
    nz = concat(buffer.get_particle_boundary_buffer("electrons", "eb", "nz", lev))

    # STEP 2: use these parameters to inject particle from the same position in the plasma
    elect_pc = particle_containers.ParticleContainerWrapper(
        "electrons"
    )  # general particle container

    ####this part is specific to the case of simple reflection.
    un = ux * nx + uy * ny + uz * nz
    ux_reflect = -2 * un * nx + ux  # for a "mirror reflection" u(sym)=-2(u.n)n+u
    uy_reflect = -2 * un * ny + uy
    uz_reflect = -2 * un * nz + uz
    elect_pc.add_particles(
        x=x + (dt - delta_t) * ux_reflect,
        y=y + (dt - delta_t) * uy_reflect,
        z=z + (dt - delta_t) * uz_reflect,
        ux=ux_reflect,
        uy=uy_reflect,
        uz=uz_reflect,
        w=w,
    )  # adds the particle in the general particle container at the next step
    #### Can be modified depending on the model of interaction.

    buffer.clear_buffer()  # reinitialise the boundary buffer


callbacks.installafterstep(
    mirror_reflection
)  # mirror_reflection is called at the next step
# using the new particle container modified at the last step

##########################
# simulation run
##########################

sim.step(max_steps)  # the whole process is done "max_steps" times
