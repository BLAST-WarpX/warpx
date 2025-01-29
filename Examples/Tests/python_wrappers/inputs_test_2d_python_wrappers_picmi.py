#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from pywarpx import picmi

# Number of time steps
max_steps = 100

# Grid
nx = 128
nz = 128

# Domain
xmin = 0.0e-6
zmin = 0.0e-6
xmax = 50.0e-6
zmax = 50.0e-6

# Cell size
dx = (xmax - xmin) / nx
dz = (zmax - zmin) / nz

# Domain decomposition
max_grid_size_x = 64
max_grid_size_z = 64

# PML
nxpml = 10
nzpml = 10
field_boundary = ["open", "open"]

# Spectral order
nox = 8
noz = 8

# Guard cells
nxg = 8
nzg = 8

# Initialize grid
grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, nz],
    lower_bound=[xmin, zmin],
    upper_bound=[xmax, zmax],
    lower_boundary_conditions=field_boundary,
    upper_boundary_conditions=field_boundary,
    guard_cells=[nxg, nzg],
    moving_window_velocity=[0.0, 0.0, 0],
    warpx_max_grid_size_x=max_grid_size_x,
    warpx_max_grid_size_y=max_grid_size_z,
)

# Initialize field solver
solver = picmi.ElectromagneticSolver(
    grid=grid,
    cfl=0.95,
    method="PSATD",
    stencil_order=[nox, noz],
    divE_cleaning=1,
    divB_cleaning=1,
    pml_divE_cleaning=1,
    pml_divB_cleaning=1,
    warpx_psatd_update_with_rho=True,
)

# Initialize diagnostics
diag_field_list = ["E", "B"]
particle_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=10,
    data_list=diag_field_list,
)
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=10,
    data_list=diag_field_list,
)

# Initialize simulation
sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=1,
    particle_shape="cubic",
    warpx_current_deposition_algo="direct",
    warpx_particle_pusher_algo="boris",
    warpx_field_gathering_algo="energy-conserving",
    warpx_use_filter=1,
)

# Add diagnostics to simulation
sim.add_diagnostic(particle_diag)
sim.add_diagnostic(field_diag)

# Write input file to run with compiled version
sim.write_input_file(file_name="inputs_2d")

# Whether to include guard cells in data returned by Python wrappers
include_ghosts = 1


# Compute min and max of fields data
def compute_minmax(data):
    vmax = np.abs(data).max()
    vmin = -vmax
    return vmin, vmax


# Plot fields data either in valid domain or in PML
def plot_data(data, pml, title, name):
    fig, ax = plt.subplots(
        nrows=1, ncols=1, gridspec_kw=dict(wspace=0.5), figsize=[6, 5]
    )
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad="5%")
    lw = 0.8
    ls = "--"
    if pml:
        # Draw PMLs and ghost regions
        ax.axvline(x=0, linewidth=lw, linestyle=ls)
        ax.axvline(x=0 + nxg, linewidth=lw, linestyle=ls)
        ax.axvline(x=-nxpml, linewidth=lw, linestyle=ls)
        ax.axvline(x=nx, linewidth=lw, linestyle=ls)
        ax.axvline(x=nx - nxg, linewidth=lw, linestyle=ls)
        ax.axvline(x=nx + nxpml, linewidth=lw, linestyle=ls)
        ax.axhline(y=0, linewidth=lw, linestyle=ls)
        ax.axhline(y=0 + nzg, linewidth=lw, linestyle=ls)
        ax.axhline(y=-nzpml, linewidth=lw, linestyle=ls)
        ax.axhline(y=nz, linewidth=lw, linestyle=ls)
        ax.axhline(y=nz - nzg, linewidth=lw, linestyle=ls)
        ax.axhline(y=nz + nzpml, linewidth=lw, linestyle=ls)
        # Annotations
        ax.annotate(
            "PML",
            xy=(-nxpml // 2, nz // 2),
            rotation="vertical",
            ha="center",
            va="center",
        )
        ax.annotate(
            "PML",
            xy=(nx + nxpml // 2, nz // 2),
            rotation="vertical",
            ha="center",
            va="center",
        )
        ax.annotate(
            "PML",
            xy=(nx // 2, -nzpml // 2),
            rotation="horizontal",
            ha="center",
            va="center",
        )
        ax.annotate(
            "PML",
            xy=(nx // 2, nz + nzpml // 2),
            rotation="horizontal",
            ha="center",
            va="center",
        )
        ax.annotate(
            "PML ghost",
            xy=(nxg // 2, nz // 2),
            rotation="vertical",
            ha="center",
            va="center",
        )
        ax.annotate(
            "PML ghost",
            xy=(-nxpml - nxg // 2, nz // 2),
            rotation="vertical",
            ha="center",
            va="center",
        )
        ax.annotate(
            "PML ghost",
            xy=(nx - nxg // 2, nz // 2),
            rotation="vertical",
            ha="center",
            va="center",
        )
        ax.annotate(
            "PML ghost",
            xy=(nx + nxpml + nxg // 2, nz // 2),
            rotation="vertical",
            ha="center",
            va="center",
        )
        ax.annotate(
            "PML ghost",
            xy=(nx // 2, nzg // 2),
            rotation="horizontal",
            ha="center",
            va="center",
        )
        ax.annotate(
            "PML ghost",
            xy=(nx // 2, -nzpml - nzg // 2),
            rotation="horizontal",
            ha="center",
            va="center",
        )
        ax.annotate(
            "PML ghost",
            xy=(nx // 2, nz - nzg // 2),
            rotation="horizontal",
            ha="center",
            va="center",
        )
        ax.annotate(
            "PML ghost",
            xy=(nx // 2, nz + nzpml + nzg // 2),
            rotation="horizontal",
            ha="center",
            va="center",
        )
        # Set extent and sliced data
        extent = np.array(
            [-nxg - nxpml, nx + nxpml + nxg, -nzg - nzpml, nz + nzpml + nzg]
        )
    else:
        # Draw ghost regions
        ax.axvline(x=0, linewidth=lw, linestyle=ls)
        ax.axvline(x=nx, linewidth=lw, linestyle=ls)
        ax.axhline(y=0, linewidth=lw, linestyle=ls)
        ax.axhline(y=nz, linewidth=lw, linestyle=ls)
        # Annotations
        ax.annotate(
            "ghost",
            xy=(-nxg // 2, nz // 2),
            rotation="vertical",
            ha="center",
            va="center",
        )
        ax.annotate(
            "ghost",
            xy=(nx + nxg // 2, nz // 2),
            rotation="vertical",
            ha="center",
            va="center",
        )
        ax.annotate(
            "ghost",
            xy=(nx // 2, -nzg // 2),
            rotation="horizontal",
            ha="center",
            va="center",
        )
        ax.annotate(
            "ghost",
            xy=(nx // 2, nz + nzg // 2),
            rotation="horizontal",
            ha="center",
            va="center",
        )
        # Set extent and sliced data
        extent = np.array([-nxg, nx + nxg, -nzg, nz + nzg])
    X = data[:, :].transpose()
    # Min and max for colorbar
    vmin, vmax = compute_minmax(X)
    # Display data as image
    im = ax.imshow(
        X=X, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap="seismic"
    )
    # Add colorbar to plot
    fig.colorbar(im, cax=cax)
    # Set label for x- and y-axis, set title
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    # Set plot title
    suptitle = "PML in (x,z), 4 grids 64 x 64"
    plt.suptitle(suptitle)
    # Save figure
    figname = "figure_" + name + ".png"
    fig.savefig(figname, dpi=100)


# Initialize fields data (unit pulse) and apply smoothing
def init_data(data):
    impulse_1d = np.array([1.0 / 4.0, 1.0 / 2.0, 1.0 / 4.0])
    impulse = np.outer(impulse_1d, impulse_1d)
    data[nx // 2 - 1 : nx // 2 + 2, nz // 2 - 1 : nz // 2 + 2] = impulse


# Initialize inputs and WarpX instance
sim.initialize_inputs()
sim.initialize_warpx()

# Get fields data using Python wrappers
import pywarpx.fields as pwxf

Ex = pwxf.ExFPWrapper(include_ghosts=include_ghosts)
Ey = pwxf.EyFPWrapper(include_ghosts=include_ghosts)
Ez = pwxf.EzFPWrapper(include_ghosts=include_ghosts)
Bx = pwxf.BxFPWrapper(include_ghosts=include_ghosts)
By = pwxf.ByFPWrapper(include_ghosts=include_ghosts)
Bz = pwxf.BzFPWrapper(include_ghosts=include_ghosts)
F = pwxf.FFPWrapper(include_ghosts=include_ghosts)
G = pwxf.GFPWrapper(include_ghosts=include_ghosts)
Expml = pwxf.ExFPPMLWrapper(include_ghosts=include_ghosts)
Eypml = pwxf.EyFPPMLWrapper(include_ghosts=include_ghosts)
Ezpml = pwxf.EzFPPMLWrapper(include_ghosts=include_ghosts)
Bxpml = pwxf.BxFPPMLWrapper(include_ghosts=include_ghosts)
Bypml = pwxf.ByFPPMLWrapper(include_ghosts=include_ghosts)
Bzpml = pwxf.BzFPPMLWrapper(include_ghosts=include_ghosts)
Fpml = pwxf.FFPPMLWrapper(include_ghosts=include_ghosts)
Gpml = pwxf.GFPPMLWrapper(include_ghosts=include_ghosts)

# Initialize fields data in valid domain
init_data(Ex)
init_data(Ey)
init_data(Ez)
init_data(Bx)
init_data(By)
init_data(Bz)
init_data(F)
init_data(G)

# Advance simulation until last time step
sim.step(max_steps)

# Plot E
plot_data(Ex, pml=False, title="Ex", name="Ex")
plot_data(Ey, pml=False, title="Ey", name="Ey")
plot_data(Ez, pml=False, title="Ez", name="Ez")

# Plot B
plot_data(Bx, pml=False, title="Bx", name="Bx")
plot_data(By, pml=False, title="By", name="By")
plot_data(Bz, pml=False, title="Bz", name="Bz")

# F and G
plot_data(F, pml=False, title="F", name="F")
plot_data(G, pml=False, title="G", name="G")

# Plot E in PML
plot_data(Expml[:, :, 0], pml=True, title="Exy in PML", name="Exy")
plot_data(Expml[:, :, 1], pml=True, title="Exz in PML", name="Exz")
plot_data(Expml[:, :, 2], pml=True, title="Exx in PML", name="Exx")
plot_data(Eypml[:, :, 0], pml=True, title="Eyz in PML", name="Eyz")
plot_data(Eypml[:, :, 1], pml=True, title="Eyx in PML", name="Eyx")
plot_data(Eypml[:, :, 2], pml=True, title="Eyy in PML", name="Eyy")  # zero
plot_data(Ezpml[:, :, 0], pml=True, title="Ezx in PML", name="Ezx")
plot_data(Ezpml[:, :, 1], pml=True, title="Ezy in PML", name="Ezy")  # zero
plot_data(Ezpml[:, :, 2], pml=True, title="Ezz in PML", name="Ezz")

# Plot B in PML
plot_data(Bxpml[:, :, 0], pml=True, title="Bxy in PML", name="Bxy")
plot_data(Bxpml[:, :, 1], pml=True, title="Bxz in PML", name="Bxz")
plot_data(Bxpml[:, :, 2], pml=True, title="Bxx in PML", name="Bxx")
plot_data(Bypml[:, :, 0], pml=True, title="Byz in PML", name="Byz")
plot_data(Bypml[:, :, 1], pml=True, title="Byx in PML", name="Byx")
plot_data(Bypml[:, :, 2], pml=True, title="Byy in PML", name="Byy")  # zero
plot_data(Bzpml[:, :, 0], pml=True, title="Bzx in PML", name="Bzx")
plot_data(Bzpml[:, :, 1], pml=True, title="Bzy in PML", name="Bzy")  # zero
plot_data(Bzpml[:, :, 2], pml=True, title="Bzz in PML", name="Bzz")

# Plot F and G in PML
plot_data(Fpml[:, :, 0], pml=True, title="Fx in PML", name="Fx")
plot_data(Fpml[:, :, 1], pml=True, title="Fy in PML", name="Fy")
plot_data(Fpml[:, :, 2], pml=True, title="Fz in PML", name="Fz")
plot_data(Gpml[:, :, 0], pml=True, title="Gx in PML", name="Gx")
plot_data(Gpml[:, :, 1], pml=True, title="Gy in PML", name="Gy")
plot_data(Gpml[:, :, 2], pml=True, title="Gz in PML", name="Gz")


# Check values with benchmarks (precomputed from the same Python arrays)
def check_values(benchmark, data, rtol, atol):
    passed = np.allclose(benchmark, np.sum(np.abs(data[:, :])), rtol=rtol, atol=atol)
    assert passed


rtol = 5e-08
atol = 1e-12

# E
check_values(1013263608.6369569, Ex[:, :], rtol, atol)
check_values(717278256.7957529, Ey[:, :], rtol, atol)
check_values(717866566.5718911, Ez[:, :], rtol, atol)
# B
check_values(3.0214509313437636, Bx[:, :], rtol, atol)
check_values(3.0242765102729985, By[:, :], rtol, atol)
check_values(3.0214509326970465, Bz[:, :], rtol, atol)
# F and G
check_values(3.0188584528062377, F[:, :], rtol, atol)
check_values(1013672631.8764204, G[:, :], rtol, atol)
# E in PML
check_values(364287936.1526477, Expml[:, :, 0], rtol, atol)
check_values(183582352.20753333, Expml[:, :, 1], rtol, atol)
check_values(190065766.41491824, Expml[:, :, 2], rtol, atol)
check_values(440581907.0828975, Eypml[:, :, 0], rtol, atol)
check_values(178117294.05871135, Eypml[:, :, 1], rtol, atol)
check_values(0.0, Eypml[:, :, 2], rtol, atol)
check_values(430277101.26568377, Ezpml[:, :, 0], rtol, atol)
check_values(0.0, Ezpml[:, :, 1], rtol, atol)
check_values(190919663.2167449, Ezpml[:, :, 2], rtol, atol)
# B in PML
check_values(1.0565189315366146, Bxpml[:, :, 0], rtol, atol)
check_values(0.46181913800643065, Bxpml[:, :, 1], rtol, atol)
check_values(0.6849858305343736, Bxpml[:, :, 2], rtol, atol)
check_values(1.7228584190213505, Bypml[:, :, 0], rtol, atol)
check_values(0.47697332248020935, Bypml[:, :, 1], rtol, atol)
check_values(0.0, Bypml[:, :, 2], rtol, atol)
check_values(1.518338068658267, Bzpml[:, :, 0], rtol, atol)
check_values(0.0, Bzpml[:, :, 1], rtol, atol)
check_values(0.6849858291863835, Bzpml[:, :, 2], rtol, atol)
# F and G in PML
check_values(1.7808748509425263, Fpml[:, :, 0], rtol, atol)
check_values(0.0, Fpml[:, :, 1], rtol, atol)
check_values(0.4307845604625681, Fpml[:, :, 2], rtol, atol)
check_values(536552745.42701197, Gpml[:, :, 0], rtol, atol)
check_values(0.0, Gpml[:, :, 1], rtol, atol)
check_values(196016270.97767758, Gpml[:, :, 2], rtol, atol)
