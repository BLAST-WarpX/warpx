#!/usr/bin/env python3
# This is the script that tests secondary ion emission when ions hit an embedded boundary
# with a specified secondary emission yield of delta_H = 0.4. Specifically, a callback
# function at each time step ensures that the correct number of secondary electrons is
# emitted when ions impact the embedded boundary, following the given secondary emission
# model defined in sigma_nescap function. This distribution depends on the ion's energy and
# suggests that for an ion incident with 1 keV energy, an average of 0.4 secondary
# electrons will be emitted.
# Simulation is initialized with four ions with i_dist distribution and spherical
# embedded boundary given by implicit function.
import numpy as np
from scipy.constants import e, elementary_charge, m_e, proton_mass

from pywarpx import callbacks, particle_containers, picmi

##########################
# numerics parameters
##########################

dt = 0.000000075

# --- Nb time steps
Te = 0.0259  # in eV
dist_th = np.sqrt(Te * elementary_charge / m_e)

max_steps = 3
diagnostic_interval = 1

# --- grid
nr = 64
nz = 64

rmin = 0.0
rmax = 2
zmin = -2
zmax = 2
delta_H = 0.4
E_HMax = 250

np.random.seed(10025015)
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
i_dist = picmi.ParticleListDistribution(
    x=[
        0.025,
        0.0,
        -0.1,
        -0.14,
    ],
    y=[0.0, 0.0, 0.0, 0],
    z=[-0.26, -0.29, -0.25, -0.23],
    ux=[0.18e6, 0.1e6, 0.15e6, 0.21e6],
    uy=[0.0, 0.0, 0.0, 0.0],
    uz=[8.00e5, 7.20e5, 6.40e5, 5.60e5],
    weight=[1, 1, 1, 1],
)

electrons = picmi.Species(
    particle_type="electron",  # Specify the particle type
    name="electrons",  # Name of the species
)

ions = picmi.Species(
    name="ions",
    particle_type="proton",
    charge=e,
    initial_distribution=i_dist,
    warpx_save_particles_at_eb=1,
)

##########################
# diagnostics
##########################

field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=diagnostic_interval,
    data_list=["Er", "Ez", "phi", "rho"],
    warpx_format="openpmd",
)

part_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=diagnostic_interval,
    species=[ions, electrons],
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
    layout=picmi.GriddedLayout(n_macroparticle_per_cell=[0, 0, 0], grid=grid),
)

sim.add_species(
    ions,
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


def sigma_nascap(energy_kEv, delta_H, E_HMax):
    """
    Compute sigma_nascap for each element in the energy array using a loop.

    Parameters:
    - energy: ndarray or list, energy values in KeV
    - delta_H: float, parameter for the formula
    - E_HMax: float, parameter for the formula in KeV

    Returns:
    - numpy array, computed probability sigma_nascap
    """
    sigma_nascap = np.array([])
    # Loop through each energy value
    for energy in energy_kEv:
        if energy > 0.0:
            sigma = (
                delta_H
                * (E_HMax + 1.0)
                / (E_HMax * 1.0 + energy)
                * np.sqrt(energy / 1.0)
            )
        else:
            sigma = 0.0
        sigma_nascap = np.append(sigma_nascap, sigma)
    return sigma_nascap


def secondary_emission():
    buffer = particle_containers.ParticleBoundaryBufferWrapper()  # boundary buffer
    # STEP 1: extract the different parameters of the boundary buffer (normal, time, position)
    lev = 0  # level 0 (no mesh refinement here)
    n = buffer.get_particle_boundary_buffer_size("ions", "eb")
    elect_pc = particle_containers.ParticleContainerWrapper("electrons")

    if n != 0:
        r = concat(buffer.get_particle_scraped_this_step("ions", "eb", "x", lev))
        theta = concat(
            buffer.get_particle_scraped_this_step("ions", "eb", "theta", lev)
        )
        z = concat(buffer.get_particle_scraped_this_step("ions", "eb", "z", lev))
        x = r * np.cos(theta)  # from RZ coordinates to 3D coordinates
        y = r * np.sin(theta)
        ux = concat(buffer.get_particle_scraped_this_step("ions", "eb", "ux", lev))
        uy = concat(buffer.get_particle_scraped_this_step("ions", "eb", "uy", lev))
        uz = concat(buffer.get_particle_scraped_this_step("ions", "eb", "uz", lev))
        w = concat(buffer.get_particle_scraped_this_step("ions", "eb", "w", lev))
        nx = concat(buffer.get_particle_scraped_this_step("ions", "eb", "nx", lev))
        ny = concat(buffer.get_particle_scraped_this_step("ions", "eb", "ny", lev))
        nz = concat(buffer.get_particle_scraped_this_step("ions", "eb", "nz", lev))
        delta_t = concat(
            buffer.get_particle_scraped_this_step("ions", "eb", "deltaTimeScraped", lev)
        )

        energy_ions = 0.5 * proton_mass * w * (ux**2 + uy**2 + uz**2)
        energy_ions_in_kEv = energy_ions / (e * 1000)
        sigma_nascap_ions = sigma_nascap(energy_ions_in_kEv, delta_H, E_HMax)
        # Loop over all ions that have been scraped in the last timestep
        for i in range(0, len(w)):
            sigma = sigma_nascap_ions[i]
            # Ne_sec is number of the secondary electrons to be emitted
            Ne_sec = int(sigma + np.random.uniform())
            for _ in range(Ne_sec):
                xe = np.array([])
                ye = np.array([])
                ze = np.array([])
                we = np.array([])
                delta_te = np.array([])
                uxe = np.array([])
                uye = np.array([])
                uze = np.array([])

                # Random thermal momenta distribution
                ux_th = np.random.normal(0, dist_th)
                uy_th = np.random.normal(0, dist_th)
                uz_th = np.random.normal(0, dist_th)

                un_th = nx[i] * ux_th + ny[i] * uy_th + nz[i] * uz_th

                if un_th < 0:
                    ux_th_reflect = (
                        -2 * un_th * nx[i] + ux_th
                    )  # for a "mirror reflection" u(sym)=-2(u.n)n+u
                    uy_th_reflect = -2 * un_th * ny[i] + uy_th
                    uz_th_reflect = -2 * un_th * nz[i] + uz_th

                    uxe = np.append(uxe, ux_th_reflect)
                    uye = np.append(uye, uy_th_reflect)
                    uze = np.append(uze, uz_th_reflect)
                else:
                    uxe = np.append(uxe, ux_th)
                    uye = np.append(uye, uy_th)
                    uze = np.append(uze, uz_th)

                xe = np.append(xe, x[i])
                ye = np.append(ye, y[i])
                ze = np.append(ze, z[i])
                we = np.append(we, w[i])
                delta_te = np.append(delta_te, delta_t[i])

                elect_pc.add_particles(
                    x=xe + (dt - delta_te) * uxe,
                    y=ye + (dt - delta_te) * uye,
                    z=ze + (dt - delta_te) * uze,
                    ux=uxe,
                    uy=uye,
                    uz=uze,
                    w=we,
                )


# using the new particle container modified at the last step
callbacks.installafterstep(secondary_emission)
##########################
# simulation run
##########################
sim.step(max_steps)  # the whole process is done "max_steps" times
