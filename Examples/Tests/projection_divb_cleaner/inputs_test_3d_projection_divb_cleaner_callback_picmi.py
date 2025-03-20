#!/usr/bin/env python3
#
# --- Input file for loading initial field from Python callback

import numpy as np
import scipy.constants as con
from mpi4py import MPI as mpi
from scipy.special import ellipe, ellipk

from pywarpx import fields, picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(verbose=True)


def augment_vector(v):
    d = v[1] - v[0]
    vp = (v[1:] + v[:-1]) / 2.0
    vp = np.insert(vp, 0, vp[0] - d)
    vp = np.append(vp, vp[-1] + d)
    return vp


class CurrentLoop(object):
    def __init__(self, **kw):
        self.radius = kw.pop("radius")  # (m)
        self.z0 = kw.pop("z0", 0.0)  # (m) Defines the middle of the current loop
        self.B0 = kw.pop("B0", 1.0)  # (T) Strength of field in middle of loop

        # Compute loop current to normalize the B field of the center of the current loop
        # from the following on axis equation for Bz

        self.I = 2.0 * self.radius * self.B0 / con.mu_0

    def psi(self, r, z):
        # Convert to spherical coordinates
        rho = np.sqrt(r**2 + (z - self.z0) ** 2)
        theta = np.arctan2(r, z - self.z0)

        coeff = con.mu_0 * self.I / (4.0 * np.pi)
        denom = self.radius**2 + rho**2 + 2.0 * self.radius * rho * np.sin(theta)

        k2 = 4.0 * self.radius * rho * np.sin(theta) / denom + 1e-12

        term1 = 4.0 * self.radius / np.sqrt(denom)
        term2 = ((2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2)) / k2

        return coeff * term1 * term2 * r

    def __call__(self, xv, yv, zv, coord="x"):
        # Generate B-field mesh
        XMB, YMB, ZMB = np.meshgrid(xv, yv, zv, indexing="ij")
        RMB = np.sqrt(XMB**2 + YMB**2)

        dx = xv[1] - xv[0]
        dy = yv[1] - yv[0]
        dz = zv[1] - zv[0]

        if coord == "x":
            # Gradient is along z direction
            zvp = augment_vector(zv)

            # A mesh, which will be reduced later after gradients taken
            XMP, YMP, ZMP = np.meshgrid(xv, yv, zvp, indexing="ij")
            RMP = np.sqrt(XMP**2 + YMP**2)
            psi = self.psi(RMP, ZMP)
            grad_psi_z = (psi[:, :, 1:] - psi[:, :, :-1]) / dz

            return -XMB * grad_psi_z / RMB**2
        elif coord == "y":
            # Gradient is along z direction
            zvp = augment_vector(zv)

            # Psi mesh, which will be reduced later after gradients taken
            XMP, YMP, ZMP = np.meshgrid(xv, yv, zvp, indexing="ij")
            RMP = np.sqrt(XMP**2 + YMP**2)
            psi = self.psi(RMP, ZMP)
            grad_psi_z = (psi[:, :, 1:] - psi[:, :, :-1]) / dz

            return -YMB * grad_psi_z / RMB**2
        elif coord == "z":
            # Gradient is along x,y directions
            xvp = augment_vector(xv)

            # Psi mesh, which will be reduced later after gradients taken
            XMP, YMP, ZMP = np.meshgrid(xvp, yv, zv, indexing="ij")
            RMP = np.sqrt(XMP**2 + YMP**2)
            psi = self.psi(RMP, ZMP)
            grad_psi_x = (psi[1:, :, :] - psi[:-1, :, :]) / dx

            yvp = augment_vector(yv)

            # Psi mesh, which will be reduced later after gradients taken
            XMP, YMP, ZMP = np.meshgrid(xv, yvp, zv, indexing="ij")
            RMP = np.sqrt(XMP**2 + YMP**2)
            psi = self.psi(RMP, ZMP)
            grad_psi_y = (psi[:, 1:, :] - psi[:, :-1, :]) / dy

            return (XMB * grad_psi_x + YMB * grad_psi_y) / RMB**2
        else:
            error("coord must be x/y/z")
            return None


class ProjectionDivCleanerTest(object):
    # Spatial domain
    Nx = 40  # number of cells in x direction
    Ny = 40  # number of cells in y direction
    Nz = 80  # number of cells in z direction

    MAX_GRID = 40
    BLOCKING_FACTOR = 8

    # Numerical parameters
    DT = 1e-9  # Time step

    def __init__(self):
        """Get input parameters for the specific case desired."""

        # output diagnostics 5 times per cyclotron period
        self.diag_steps = 1
        self.total_steps = 1

        self.Lx = 1.0
        self.Ly = 1.0
        self.Lz = 2.0

        self.DX = self.Lx / self.Nx
        self.DY = self.Ly / self.Ny
        self.DZ = self.Lz / self.Nz

        self.dt = self.DT

        """Setup simulation components."""

        #######################################################################
        # Set geometry and boundary conditions                                #
        #######################################################################

        self.grid = picmi.Cartesian3DGrid(
            number_of_cells=[self.Nx, self.Ny, self.Nz],
            warpx_max_grid_size_x=self.MAX_GRID,
            warpx_max_grid_size_y=self.MAX_GRID,
            warpx_max_grid_size_z=self.MAX_GRID,
            warpx_blocking_factor=self.BLOCKING_FACTOR,
            lower_bound=[-self.Lx / 2.0, -self.Ly / 2.0, -self.Lz / 2],
            upper_bound=[self.Lx / 2.0, self.Ly / 2.0, self.Lz / 2.0],
            lower_boundary_conditions=["periodic", "periodic", "neumann"],
            upper_boundary_conditions=["periodic", "periodic", "neumann"],
            lower_boundary_conditions_particles=["periodic", "periodic", "absorbing"],
            upper_boundary_conditions_particles=["periodic", "periodic", "absorbing"],
        )
        simulation.time_step_size = self.dt
        simulation.max_steps = self.total_steps

        #######################################################################
        # Field solver and external field                                     #
        #######################################################################

        self.solver = picmi.ElectrostaticSolver(grid=self.grid)
        simulation.solver = self.solver

        #######################################################################
        # Install Callbacks                                                   #
        #######################################################################
        init_field = picmi.LoadInitialFieldFromPython(
            load_from_python=load_current_ring,
            warpx_do_divb_cleaning_external=True,
            load_E=False,
        )
        simulation.add_applied_field(init_field)

        #######################################################################
        # Add diagnostics                                                     #
        #######################################################################

        field_diag = picmi.FieldDiagnostic(
            name="diag1",
            grid=self.grid,
            period=self.diag_steps,
            data_list=["B"],
            warpx_format="plotfile",
        )
        simulation.add_diagnostic(field_diag)

        comm.Barrier()

        # Initialize inputs and WarpX instance
        simulation.initialize_inputs()
        simulation.initialize_warpx()


def load_current_ring():
    curr_loop = CurrentLoop(radius=0.75)

    Bx = fields.BxFPExternalWrapper(include_ghosts=True)
    By = fields.ByFPExternalWrapper(include_ghosts=True)
    Bz = fields.BzFPExternalWrapper(include_ghosts=True)

    Bx[:, :, :] = curr_loop(Bx.mesh("x"), Bx.mesh("y"), Bx.mesh("z"), coord="x")

    By[:, :, :] = curr_loop(By.mesh("x"), By.mesh("y"), By.mesh("z"), coord="y")

    Bz[:, :, :] = curr_loop(Bz.mesh("x"), Bz.mesh("y"), Bz.mesh("z"), coord="z")

    comm.Barrier()


run = ProjectionDivCleanerTest()
simulation.step()

##############################################
# Post load image generation and error check #
##############################################
Bxg = fields.BxWrapper(include_ghosts=True)
Byg = fields.ByWrapper(include_ghosts=True)
Bzg = fields.BzWrapper(include_ghosts=True)

Bx_local = Bxg[:, :, :]
By_local = Byg[:, :, :]
Bz_local = Bzg[:, :, :]

dBxdx = (Bx_local[1:, :, :] - Bx_local[:-1, :, :]) / run.DX
dBydy = (By_local[:, 1:, :] - By_local[:, :-1, :]) / run.DY
dBzdz = (Bz_local[:, :, 1:] - Bz_local[:, :, :-1]) / run.DZ

divB = dBxdx + dBydy + dBzdz

import matplotlib.pyplot as plt

plt.imshow(np.log10(np.abs(divB[:, 24, :])))
plt.title("log10(|div(B)|)")
plt.colorbar()
plt.savefig("divb.png")

error = np.sqrt((divB[2:-2, 2:-2, 2:-2] ** 2).sum())
tolerance = 1e-12

print("error = ", error)
print("tolerance = ", tolerance)
assert error < tolerance
