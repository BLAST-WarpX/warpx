# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# %matplotlib qt

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from pywarpx import callbacks, particle_containers, picmi
from pywarpx.LoadThirdParty import load_cupy

constants = picmi.constants


def energy_to_velocity(energy_ev, m0=9.1093837139e-31):
    """
    Converts beam kinetic energy to relativistic velocity.

    Parameters:
    energy_ev (float): Kinetic energy of the beam in electronvolts (eV).

    Returns:
    float: Velocity of the electron in meters per second (m/s).
    """

    # Calculate electron rest mass energy in Joules
    E_rest_joules = m0 * (constants.c**2)

    # Convert input kinetic energy from eV to Joules
    E_k_joules = energy_ev * constants.q_e

    # Calculate velocity using the relativistic formula
    # v = c * sqrt(1 - (E_rest / (E_k + E_rest))^2)
    gamma_inv = E_rest_joules / (E_k_joules + E_rest_joules)
    velocity = constants.c * np.sqrt(1.0 - (gamma_inv**2))

    return velocity


def to_cpu_array(arr):
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


#### setup "in-situ" plotting
if mpl.get_backend().lower() == "agg":
    backend = "Agg"
    mpl.use(backend)
else:
    backend = "QtAgg"
    mpl.use(backend)
    plt.ion()

figsize = (12, 6)
fig = plt.figure(1, figsize=figsize)
ax = fig.subplots(nrows=1, ncols=1)

# scat = ax.scatter([], [], s=1)
colors = "brgk"

print(backend)


def plot_particles(ps=["electrons"]):
    xp, _ = load_cupy()
    xymin = xp.array(sim.solver.grid.lower_bound)
    xymax = xp.array(sim.solver.grid.upper_bound)
    area = xp.prod(xymax - xymin)
    ax.clear()
    ax.set(xlim=(xymin[0], xymax[0]), ylim=(xymin[1], xymax[1]))
    info = ""
    for i, particle in enumerate(ps):
        p_pc = sim.particles.get(particle)
        mpsum = 0.0
        # psum = 0.0
        for pti in p_pc.iterator(level=0):
            x = to_cpu_array(pti["x"])
            y = to_cpu_array(pti["z"])
            ax.scatter(x, y, c=colors[(i) % len(colors)], s=1)

        # charge = p_pc.sum_particle_charge(0)
        # mpsum = p_pc.total_number_of_particles()
        density = mpsum / area
        info += "N%s = %i, RHO%s = %0.2e, " % (particle, mpsum, particle, density)
    # print(libwarpx.warpx.set_epsilon_r)
    ax.set(title=info)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def H(x):
    return np.heaviside(x, 0.5)


gap = 0.1  # m


sphereVoltage = 50
anodeVoltage = 100
plasma_density = 2.56e9
elec_temp = 3000.0  # K

beam_energy = 50.0  # [eV]
max_beam_energy = max(anodeVoltage, sphereVoltage) + beam_energy  # [eV]

##########################
# numerics parameters
##########################

# --- Number of time steps
max_steps = 1000
nDumps = 100
diagnostic_intervals = "::%i" % (max_steps / nDumps)
step_interval = 100


# --- Grid
xmin = 0.0
zmin = 0.0

nx = 64
nz = 64
xmax = gap
zmax = gap
dx = (xmax - xmin) / nx
dz = (zmax - zmin) / nz


number_per_cell = 5
number_per_cell_each_dim = [number_per_cell, number_per_cell]

dt_frac = 0.95
dt = dt_frac * min(dx, dz) / energy_to_velocity(max(max_beam_energy, 1.0))
total_time = max_steps * dt

##########################
# particle definitions
##########################
v_beam = energy_to_velocity(beam_energy)
v_rms_elec = np.sqrt(constants.kb * elec_temp / constants.m_e)
plasma_flux = plasma_density * v_beam
particle_flux = picmi.UniformFluxDistribution(
    flux=plasma_flux,
    flux_normal_axis="z",
    surface_flux_position=0.0,
    flux_direction=1,
    lower_bound=[xmin, zmin, None],
    upper_bound=[xmax, zmin + dz, None],
    rms_velocity=[v_rms_elec] * 3,
    directed_velocity=[0.0, v_beam, 0.0],
)

electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    initial_distribution=particle_flux,
    warpx_save_particles_at_eb=True,
)


##########################
# numerics components
##########################

grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, nz],
    warpx_max_grid_size=64,
    # warpx_blocking_factor=blocking_factor,
    lower_bound=[xmin, zmin],
    upper_bound=[xmax, zmax],
    bc_xmin="neumann",
    bc_xmax="neumann",
    bc_ymin="dirichlet",
    bc_ymax="dirichlet",
    lower_boundary_conditions_particles=["absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
    warpx_potential_lo_z=0.0,
    warpx_potential_hi_z=anodeVoltage,
    warpx_boundary_particle_eb="Absorbing",
)


solver = picmi.ElectrostaticSolver(
    grid=grid,
    method="Multigrid",
    required_precision=1e-5,
    warpx_absolute_tolerance=1e2,
    warpx_self_fields_verbosity=0,
)

##########################
# geometry
##########################
x0 = gap / 4.0
z0 = gap / 2.0
r = gap / 10.0
implicit_expr = "r^2-(x-x0)^2-(z-z0)^2"


eb = picmi.EmbeddedBoundary(
    implicit_function=implicit_expr, potential=sphereVoltage, x0=x0, z0=z0, r=r
)

x1 = gap * 2.0 / 4.0
z1 = gap / 2.0
implicit_expr = "r^2-(x-x1)^2-(z-z1)^2"
sink0 = picmi.ParticleSink(
    name="sink0", implicit_function=implicit_expr, type="Absorbing", x1=x1, z1=z1, r=r
)


x2 = gap * 3.0 / 4.0
z2 = gap / 2.0

## Test stl file
# stl_file = "circle.stl"
# sphere = trimesh.creation.icosphere(subdivisions=3, radius=r)
# sphere.apply_translation([x2, z2, 0.00])  # Shift center to (x2, y2, z2)
# sphere.export(stl_file )
# sink1 = picmi.ParticleSink(name="sink1", stl_file = stl_file)

implicit_expr = "r^2-(x-x2)^2-(z-z2)^2"
sink1 = picmi.ParticleSink(
    name="sink1", implicit_function=implicit_expr, type="Reflecting", x2=x2, z2=z2, r=r
)
##########################
# diagnostics
##########################

particle_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=diagnostic_intervals,
    data_list=["position", "momentum", "weighting"],
    warpx_format="openpmd",
)
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=diagnostic_intervals,
    data_list=["rho_electrons", "E", "phi"],
    # data_list=["E",'phi'],
    warpx_format="openpmd",
)

##########################
# simulation setup
##########################
layout_elec = picmi.PseudoRandomLayout(
    n_macroparticles_per_cell=number_per_cell, grid=grid
)
# layout_elec = picmi.GriddedLayout(n_macroparticle_per_cell=number_per_cell_each_dim, grid=grid)

sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=False,
    time_step_size=dt,
    warpx_embedded_boundary=eb,
)

sim.add_species(electrons, layout=layout_elec)

sim.add_particle_sink(sink0)
sim.add_particle_sink(sink1)

sim.add_diagnostic(particle_diag)
sim.add_diagnostic(field_diag)


ps = ["electrons"]
callbacks.installcallback("particleinjection", lambda: plot_particles(ps=ps))
##########################
# simulation run
##########################
sim.initialize_inputs()
sim.initialize_warpx()

xp, _ = load_cupy()


def concat(list_of_arrays):
    if len(list_of_arrays) == 0:
        # Return a 1d array of size 0
        return xp.empty(0)
    else:
        return xp.concatenate(list_of_arrays)


if xp.__name__ == "cupy":
    fuse = xp.fuse
else:
    # No-op decorator for NumPy/CPU
    def fuse(func):
        return func


@fuse
def in_range(tile, low, high):
    return (tile >= low) & (tile <= high)


xEB = [x0 - r, x0 + r]
xSink0 = [x1 - r, x1 + r]
xSink1 = [x2 - r, x2 + r]
sinks = [xEB, xSink0, xSink1]
Ns = [0.0, 0.0, 0.0]


def access_particle_buffer():
    buffer = particle_containers.ParticleBoundaryBufferWrapper()
    lev = 0
    xTiles = buffer.get_particle_scraped_this_step("electrons", "eb", "x", lev)
    for i, sink in enumerate(sinks):
        Ns[i] = sum(
            xp.count_nonzero(in_range(xTile, sink[0], sink[1])).item()
            for xTile in xTiles
        )
    print(Ns)


callbacks.installafterstep(access_particle_buffer)
sim.step(max_steps)
if backend == "QtAgg":
    plt.ioff()
