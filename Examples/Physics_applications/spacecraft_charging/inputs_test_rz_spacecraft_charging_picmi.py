#!/usr/bin/env python3
#
# --- Input file for spacecraft charging testing in RZ.
# --- This input defines a conducting sphere (spacecraft) immersed in a thermal
# --- plasma with the same given initial conditions as in the article:
# --- (*) J. Deca, G. Lapenta, R. Marchand, S. Markidis;
# ---     Spacecraft charging analysis with the implicit particle-in-cell code iPic3D.
# ---     Part III. A. pages 3-4
# ---     Phys. Plasmas 1 October 2013; 20 (10): 102902. https://doi.org/10.1063/1.4826951.
# --- The conducting sphere starts with an initial potential of 1V and will interact with
# --- the surrounding plasma, initially static. The charging of the spacecraft - by accumulation
# --- of electrons - leads to a decrease of the potential on the surface over the time
# --- until reaching an equilibrium floating potential of ~144.5 V (*).

import numpy as np
import scipy.constants as scc
from mpi4py import MPI as mpi

from pywarpx import picmi
from pywarpx.callbacks import installafterEsolve, installafterInitEsolve
from pywarpx.fields import ExWrapper, EzWrapper, PhiFPWrapper, RhoFPWrapper
from pywarpx.particle_containers import ParticleBoundaryBufferWrapper


# Utilities
class SpaceChargeFieldCorrector(object):
    """
    Class used by the callback functions to calculate the
    correct field around the spacecraft, at each timestep
    (taking into account the charge that has been collected on the spacecraft)
    """

    def __init__(self):
        self.saved_first_iteration_fields = False
        self.spacecraft_potential = 1.0  # Initial voltage: 1V
        self.spacecraft_capacitance = None

    def correct_space_charge_fields(self, q=None):
        """
        Function that will be called at each iteration,
        after each electrostatic solve in WarpX
        """
        assert self.saved_first_iteration_fields

        # Compute the charge that WarpX thinks there is on the spacecraft
        # from phi and rho after the Poisson solver
        q_v = compute_virtual_charge_on_spacecraft()
        if q is None:
            q = compute_actual_charge_on_spacecraft()

        # Correct fields so as to recover the actual charge
        Er = ExWrapper(include_ghosts=True)
        Er[...] += (q - q_v) * self.normalized_Er
        Ez = EzWrapper(include_ghosts=True)
        Ez[...] += (q - q_v) * self.normalized_Ez
        phi = PhiFPWrapper(include_ghosts=True)
        phi[...] += (q - q_v) * self.normalized_phi
        self.spacecraft_potential += (q - q_v) * self.spacecraft_capacitance
        sim.extension.warpx.set_potential_on_eb("%f" % self.spacecraft_potential)
        print("Setting potential to %f" % self.spacecraft_potential)

        # Confirm that the charge on the spacecraft is now correct
        compute_virtual_charge_on_spacecraft()

    def save_normalized_vacuum_Efields(
        self,
    ):
        # Compute the charge that WarpX thinks there is on the spacecraft
        # from phi and rho after the Poisson solver
        q_v = compute_virtual_charge_on_spacecraft()
        self.spacecraft_capacitance = 1.0 / q_v  # the potential was set to 1V

        # Check that this iteration corresponded to a vacuum solve
        rho = RhoFPWrapper(include_ghosts=False)

        # In principle, we should check that `rho` is exactly 0
        # However, due to machine precision errors when adding the charge
        # of ions and electrons, this can be slightly different than 0
        assert np.all(abs(rho[...]) < 1.0e-11)

        # Record fields
        Er = ExWrapper(include_ghosts=True)[:, :]
        self.normalized_Er = Er[...] / q_v
        Ez = EzWrapper(include_ghosts=True)[:, :]
        self.normalized_Ez = Ez[...] / q_v
        phi = PhiFPWrapper(include_ghosts=True)[:, :]
        self.normalized_phi = phi[...] / q_v

        self.saved_first_iteration_fields = True
        self.correct_space_charge_fields(q=0)


def compute_virtual_charge_on_spacecraft():
    """
    Given that we asked WarpX to solve the Poisson
    equation with phi=1 on the spacecraft and phi=0
    on the boundary of the domain, compute the charge
    that WarpX thinks there should be on the spacecraft.
    """
    # Get global array for the whole domain (across MPI ranks)
    phi = PhiFPWrapper(include_ghosts=False)[:, :]
    rho = RhoFPWrapper(include_ghosts=False)[:, :]

    # Check that this codes correspond to the global size of the box
    assert phi.shape == (nr + 1, nz + 1)
    assert rho.shape == (nr + 1, nz + 1)

    dr, dz = sim.extension.warpx.Geom(lev=0).data().CellSize()

    # Compute integral of grad phi over surfaces of the domain
    r = np.linspace(rmin, rmax, len(phi), endpoint=False) + (rmax - rmin) / (
        2 * len(phi)
    )  # shift of the r points because the derivaties are calculated in the middle
    face_z0 = (
        2 * np.pi * 1.0 / dz * ((phi[:, 0] - phi[:, 1]) * r).sum() * dr
    )  # here I am assuming that phi is a numpy array that can handle elementwise mult
    face_zend = 2 * np.pi * 1.0 / dz * ((phi[:, -1] - phi[:, -2]) * r).sum() * dr
    face_rend = 2 * np.pi * 1.0 / dr * ((phi[-1, :] - phi[-2, :]) * rmax).sum() * dz
    grad_phi_integral = face_z0 + face_zend + face_rend

    # Compute integral of rho over volume of the domain
    # (i.e. total charge of the plasma particles)
    rho_integral = (
        (rho[1 : nr - 1, 1 : nz - 1] * r[1 : nr - 1, np.newaxis]).sum()
        * 2
        * np.pi
        * dr
        * dz
    )

    # Compute charge of the spacecraft, based on Gauss theorem
    q_spacecraft = -rho_integral - scc.epsilon_0 * grad_phi_integral
    print("Virtual charge on the spacecraft: %e" % q_spacecraft)
    return q_spacecraft


def compute_actual_charge_on_spacecraft():
    """
    Compute the actual charge on the spacecraft,
    by counting how many electrons and protons
    were collected by the WarpX embedded boundary (EB)
    """
    charge = {"electrons": -scc.e, "protons": scc.e}
    q_spacecraft = 0
    particle_buffer = ParticleBoundaryBufferWrapper()
    for species in charge.keys():
        weights = particle_buffer.get_particle_boundary_buffer(species, "eb", "w", 0)
        sum_weights_over_tiles = sum([w.sum() for w in weights])

        # Reduce across all MPI ranks
        ntot = float(mpi.COMM_WORLD.allreduce(sum_weights_over_tiles, op=mpi.SUM))
        print("Total number of %s collected on spacecraft: %e" % (species, ntot))
        q_spacecraft += ntot * charge[species]

    print("Actual charge on the spacecraft: %e" % q_spacecraft)
    return q_spacecraft


##########################
# numerics parameters
##########################

dt = 1.27e-8

# --- Nb time steps
max_steps = 1000
diagnostic_interval = 10

# --- grid
nr = 40
nz = 80

rmin = 0.0
rmax = 3
zmin = -3
zmax = 3

number_per_cell = 5
number_per_cell_each_dim = [10, 1, 1]


##########################
# physics components
##########################

n = 7.0e9  # plasma density #particles/m^3
Te = 85  # Electron temp in eV
Ti = 0.05 * Te  # Ion temp in eV
qe = picmi.constants.q_e  # elementary charge
m_e = picmi.constants.m_e  # electron mass
m_i = 1836.0 * m_e  # mass of ion
v_eth = (qe * Te / m_e) ** 0.5
v_pth = (qe * Ti / m_i) ** 0.5

# nothing to change in the distribution function?
e_dist = picmi.UniformDistribution(density=n, rms_velocity=[v_eth, v_eth, v_eth])
e_dist2 = picmi.UniformFluxDistribution(
    flux=n * v_eth / (2 * np.pi) ** 0.5,  # Flux for Gaussian with vmean=0
    surface_flux_position=3,
    flux_direction=-1,
    flux_normal_axis="r",
    gaussian_flux_momentum_distribution=True,
    rms_velocity=[v_eth, v_eth, v_eth],
)
electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    initial_distribution=[e_dist, e_dist2],
    warpx_save_particles_at_eb=1,
)

p_dist = picmi.UniformDistribution(density=n, rms_velocity=[v_pth, v_pth, v_pth])
p_dist2 = picmi.UniformFluxDistribution(
    flux=n * v_pth / (2 * np.pi) ** 0.5,  # Flux for Gaussian with vmean=0
    surface_flux_position=3,
    flux_direction=-1,
    flux_normal_axis="r",
    gaussian_flux_momentum_distribution=True,
    rms_velocity=[v_pth, v_pth, v_pth],
)
protons = picmi.Species(
    particle_type="proton",
    name="protons",
    initial_distribution=[p_dist, p_dist2],
    warpx_save_particles_at_eb=1,
)


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
    implicit_function="-(x**2+y**2+z**2-radius**2)",
    potential=1.0,  # arbitrary value ; this will be corrected by a callback function
    radius=0.3277,
)


##########################
# diagnostics
##########################

field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=diagnostic_interval,
    data_list=["Er", "Ez", "phi", "rho", "rho_electrons", "rho_protons"],
    warpx_format="openpmd",
)

part_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=diagnostic_interval,
    species=[electrons, protons],
    warpx_format="openpmd",
)

part_scraping_boundary_diag = picmi.ParticleBoundaryScrapingDiagnostic(
    name="diag2",
    period=-1,  # only at the end, because we also use the buffers in this test
    species=[electrons, protons],
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
    warpx_random_seed=1,
)

layout1 = picmi.GriddedLayout(
    n_macroparticle_per_cell=number_per_cell_each_dim, grid=grid
)
layout2 = picmi.PseudoRandomLayout(n_macroparticles_per_cell=number_per_cell, grid=grid)
sim.add_species(electrons, layout=[layout1, layout2])

sim.add_species(protons, layout=[layout1, layout2])

sim.add_diagnostic(field_diag)
sim.add_diagnostic(part_diag)
sim.add_diagnostic(part_scraping_boundary_diag)

##########################
# simulation run
##########################

spc = SpaceChargeFieldCorrector()

installafterInitEsolve(spc.save_normalized_vacuum_Efields)
installafterEsolve(spc.correct_space_charge_fields)

sim.step(max_steps)
