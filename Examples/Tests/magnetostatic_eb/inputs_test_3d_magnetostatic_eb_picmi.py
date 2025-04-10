#!/usr/bin/env python3
#
# Copyright 2022 S. Eric Clark, LLNL
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
This script tests the magnetostatic solver in a beampipe by initializing a
uniform electron beam with relativistic gamma of 10, and a beam current of 1 kA.
The MLMG soltution from WarpX is computed and compared to the simple analytic
solution.  The analytic and simulated values are plotted over one another for the
electric field and magnetic field solutions after taking the gradient of phi and
the curl of the vector potential A.  This runs in 3 dimensions, has a PEC boundary
at z=0, and a Neumann boundary at z=1.  This exercises the domain boundary conditions
as well as the embedded boundary conditions and will fail if the maximum error is too large.
The script additionally outputs png images with the average fields in a subset of the beampipe
as well as the analytical solution.
"""

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as con

from pywarpx import fields, picmi

##########################
# physics parameters
##########################

V_domain_boundary = 0.0
V_embedded_boundary = 1.0

##########################
# numerics parameters
##########################

dt = 1e-12

# --- Nb time steps

max_steps = 1

# --- grid

nx = 64
ny = 64
nz = 64

xmin = -0.25
xmax = 0.25
ymin = -0.25
ymax = 0.25
zmin = 0.0
zmax = 1.0

##########################
# numerics components
##########################

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[xmin, ymin, zmin],
    upper_bound=[xmax, ymax, zmax],
    lower_boundary_conditions=["neumann", "neumann", "dirichlet"],
    upper_boundary_conditions=["neumann", "neumann", "neumann"],
    lower_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    warpx_potential_lo_z=V_domain_boundary,
    warpx_blocking_factor=8,
    warpx_max_grid_size=32,
)

solver = picmi.ElectrostaticSolver(
    grid=grid,
    method="Multigrid",
    required_precision=1e-7,
    warpx_magnetostatic=True,
    warpx_self_fields_verbosity=3,
)

r_pipe = 0.2

embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function="(x**2+y**2-radius**2)",
    potential=V_embedded_boundary,
    radius=r_pipe,
)


# Beam Current Density
current = 1000  # A
beam_r = 0.1  # m

J = current / (np.pi * beam_r**2)
beam_gamma = 10.0
vz = con.c * np.sqrt(1.0 - 1.0 / beam_gamma**2)
n0 = J / (con.e * vz)


beam_dist = picmi.AnalyticDistribution(
    density_expression="((x**2+y**2)<rmax**2)*n0",
    rmax=beam_r,
    n0=n0,
    directed_velocity=[0.0, 0.0, beam_gamma * vz],
)
beam = picmi.Species(
    particle_type="electron", name="beam", initial_distribution=beam_dist
)
beam_layout = picmi.GriddedLayout(n_macroparticle_per_cell=[2, 2, 2], grid=grid)

##########################
# diagnostics
##########################

particle_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=1,
)
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=1,
    data_list=[
        "Ex",
        "Ey",
        "Ez",
        "Bx",
        "By",
        "Bz",
        "Ax",
        "Ay",
        "Az",
        "Jx",
        "Jy",
        "Jz",
        "phi",
        "rho",
    ],
)

##########################
# simulation setup
##########################

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=max_steps,
    warpx_embedded_boundary=embedded_boundary,
    warpx_field_gathering_algo="momentum-conserving",
    warpx_current_deposition_algo="direct",
    warpx_use_filter=False,
    warpx_serialize_initial_conditions=True,
    warpx_do_dynamic_scheduling=False,
)

sim.add_species(beam, layout=beam_layout, initialize_self_field=True)

sim.add_diagnostic(particle_diag)
sim.add_diagnostic(field_diag)

##########################
# simulation run
##########################

sim.step(max_steps)

##########################
# Testing
##########################

Ex = fields.ExWrapper()
Ey = fields.EyWrapper()

x_vec = Ex.mesh("x")
y_vec = Ex.mesh("y")
z_vec = Ex.mesh("z")


@np.vectorize
def Er_an(r):
    if r < beam_r:
        er = -current * r / (2.0 * np.pi * beam_r**2 * con.epsilon_0 * vz)
    elif r >= beam_r and r < r_pipe:
        er = -current / (2.0 * np.pi * r * con.epsilon_0 * vz)
    else:
        er = np.zeros_like(r)
    return er


# compare to region from 0.5*zmax to 0.9*zmax
z_idx = (z_vec >= 0.5 * zmax) & (z_vec < 0.9 * zmax)

Ex_dat = Ex[...]
Ey_dat = Ey[...]

Ex_mean = Ex_dat[:, :, z_idx].mean(axis=2).T
Ey_mean = Ey_dat[:, :, z_idx].mean(axis=2).T

Ex_nodal = Ex_mean
Ey_nodal = Ey_mean

XM, YM = np.meshgrid(x_vec, y_vec, indexing="xy")
RM = np.sqrt(XM**2 + YM**2)
THM = np.arctan2(YM, XM)

Er_mean = np.cos(THM) * Ex_nodal + np.sin(THM) * Ey_nodal

r_vec = np.sqrt(x_vec**2 + y_vec**2)

r_idx = RM < 0.95 * r_pipe
r_sub = RM[r_idx]

plt.figure(1)
plt.plot(r_vec, Er_an(r_vec))
plt.plot(RM.flatten(), Er_mean.flatten(), ".")
plt.legend(["Analytical", "Electrostatic"])

er_err = np.abs(Er_mean[r_idx] - Er_an(r_sub)).max() / np.abs(Er_an(r_sub)).max()

plt.ylabel(r"$E_r$ (V/m)")
plt.xlabel("r (m)")
plt.title("Max % Error: {} %".format(er_err * 100.0))
plt.tight_layout()
plt.savefig("er_3d.png")

assert er_err < 0.05, "Er Max Error increased above 5%"

########################
# Check B field
########################

Bx = fields.BxWrapper()
By = fields.ByWrapper()

x_vec = Bx.mesh("x")
y_vec = Bx.mesh("y")
z_vec = Bx.mesh("z")

dx = x_vec[1] - x_vec[0]
dy = y_vec[1] - y_vec[0]
x_vec = x_vec + dx / 2.0
y_vec = y_vec + dy / 2.0


@np.vectorize
def Bt_an(r):
    if r < beam_r:
        bt = -current * r * con.mu_0 / (2.0 * np.pi * beam_r**2)
    elif r >= beam_r and r < r_pipe:
        bt = -current * con.mu_0 / (2.0 * np.pi * r)
    else:
        bt = np.zeros_like(r)
    return bt


# compare to region from 0.25*zmax to 0.75*zmax
z_idx = (z_vec >= 0.25 * zmax) & (z_vec < 0.75 * zmax)
z_sub = z_vec[z_idx]

Bx_dat = Bx[...]
By_dat = By[...]

Bx_mean = Bx_dat[:, :, z_idx].mean(axis=2).T
By_mean = By_dat[:, :, z_idx].mean(axis=2).T

# Interpolate B mesh to nodal points excluding last mesh point
Bx_nodal = (Bx_mean[:-1, 1:] + Bx_mean[:-1, :-1]) / 2.0
By_nodal = (By_mean[:-1, :-1] + By_mean[1:, :-1]) / 2.0

x_vec = x_vec[:-1]
y_vec = y_vec[:-1]


XM, YM = np.meshgrid(x_vec, y_vec, indexing="xy")
RM = np.sqrt(XM**2 + YM**2)
THM = np.arctan2(YM, XM)

Bt_mean = -np.sin(THM) * Bx_nodal + np.cos(THM) * By_nodal


r_vec = np.sqrt(x_vec**2 + y_vec**2)

r_idx = RM < 0.95 * r_pipe
r_sub = RM[r_idx]

plt.figure(2)
plt.plot(r_vec, Bt_an(r_vec))
plt.plot(RM[r_idx].flatten(), Bt_mean[r_idx].flatten(), ".")
plt.legend(["Analytical", "Magnetostatic"])

bt_err = np.abs(Bt_mean[r_idx] - Bt_an(r_sub)).max() / np.abs(Bt_an(r_sub)).max()

plt.ylabel(r"$B_{\Theta}$ (T)")
plt.xlabel("r (m)")
plt.title("Max % Error: {} %".format(bt_err * 100.0))
plt.tight_layout()
plt.savefig("bt_3d.png")

assert bt_err < 0.05, "Bt Max Error increased above 5%"
