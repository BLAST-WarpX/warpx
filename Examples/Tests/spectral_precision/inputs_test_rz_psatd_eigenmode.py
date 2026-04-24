#!/usr/bin/env python3
"""
RZ PSATD Hankel eigenmode precision test.

Initializes the full TE vacuum eigenmode (satisfying div B = 0) using the
first azimuthal (m=0) Bessel radial mode combined with one z-period:

  Bz(r,z,0) = B0 * J0(kr * r) * sin(kz * z)
  Br(r,z,0) = -B0 * (kz/kr) * J1(kr * r) * cos(kz * z)   [from div B = 0]
  Et = Er = Ez = 0

where kr = alpha01 / Lr  (alpha01 = first zero of J0 ~ 2.4048)
      kz = 2*pi / Lz

The exact vacuum solution at time t is:
  Bz(r,z,t) = B0 * J0(kr*r) * sin(kz*z) * cos(omega*t)
  Br(r,z,t) = -B0 * (kz/kr) * J1(kr*r) * cos(kz*z) * cos(omega*t)
  Et(r,z,t) = B0 * (omega/kr) * J1(kr*r) * sin(kz*z) * sin(omega*t)
where omega = c * sqrt(kr^2 + kz^2).

Initialization uses a Python callback (LoadInitialFieldFromPython) that
evaluates the exact formulas at each Yee cell centre using scipy, bypassing
the AMReX parser entirely.  This eliminates the ~1e-3 Hankel quadrature
aliasing error that arises when the parser evaluates Bessel functions at
cell centres.
"""

import argparse

from scipy.special import j0 as scipy_j0
from scipy.special import j1 as scipy_j1

parser = argparse.ArgumentParser(
    description="RZ PSATD Hankel eigenmode precision test.",
)
parser.add_argument(
    "--nr",
    type=int,
    default=64,
    help="Radial cells (default: 64).",
)
parser.add_argument(
    "--nz",
    type=int,
    default=64,
    help="Axial cells (default: 64).",
)
parser.add_argument(
    "--max_grid_size",
    type=int,
    default=None,
    help="Maximum box size (default: nz, i.e. single box).",
)
args, _ = parser.parse_known_args()

import numpy as np

from pywarpx import picmi

constants = picmi.constants

# Physical parameters
B0 = 1.0e-3  # T — magnetic field amplitude
c = constants.c
Lr = 1.0e-6  # m — radial domain [0, Lr]
Lz = 1.0e-6  # m — axial domain [0, Lz], periodic

nr = args.nr
nz = args.nz

# kr = alpha01 / Lr where alpha01 is the first zero of J0
alpha01 = 2.4048255577957827
kr = alpha01 / Lr
kz = 2.0 * np.pi / Lz

# Choose dt so omega*dt is not a small rational multiple of pi.
# Use 0.4 * CFL limit — keeps dispersion error well below tolerance.
dr = Lr / nr
dz = Lz / nz
dt_cfl = 1.0 / (c * np.sqrt(1.0 / dr**2 + 1.0 / dz**2))
dt = 0.4 * dt_cfl

max_steps = 40
n_modes = 1  # m=0 only; the eigenmode is azimuthally symmetric

mgs = args.max_grid_size if args.max_grid_size else nz

grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz],
    lower_bound=[0.0, 0.0],
    upper_bound=[Lr, Lz],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["none", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    n_azimuthal_modes=n_modes,
    warpx_max_grid_size=mgs,
)

solver = picmi.ElectromagneticSolver(
    grid=grid,
    method="PSATD",
    cfl=0.99,
)

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=max_steps,
    verbose=0,
)


# Full eigenmode initialization via Python callback.
# scipy_j0/j1 evaluate Bessel functions exactly at the Yee cell centres
# reported by mesh("r") / mesh("z"), bypassing the AMReX parser.
# Bz and Br are both set to satisfy div B = 0 at t=0:
#   (1/r) d(r*Br)/dr + dBz/dz = 0  =>  Br = -(kz/kr)*J1(kr*r)*cos(kz*z)*B0
def _fill_mfab_from_numpy(mfab, values_2d):
    """Fill a 2-D RZ MultiFab from a numpy array using the MFIter API.

    Works on both CPU and GPU (including SYCL builds without cupy):
    on GPU a pinned-arena copy of the full MultiFab is used as an
    intermediary: numpy data is written there (host memory), then
    htod_memcpy transfers everything to the device in one call.

    Parameters
    ----------
    mfab : amrex MultiFab (2-D, single component, F-order [r, z])
        Target field MultiFab.
    values_2d : np.ndarray, shape (nr, nz)
        Global data array in Fortran (r-major) order.  Indexed as
        values_2d[ir, iz] where ir/iz run over the valid-cell range.
    """
    import inspect

    amr = inspect.getmodule(mfab)
    have_gpu = amr.Config.have_gpu

    min_box = mfab.box_array().minimal_box()
    r0 = min_box.small_end[0]
    z0 = min_box.small_end[1]

    if have_gpu:
        # Allocate a pinned (host-accessible) MultiFab with the same
        # BoxArray/DistributionMapping as the target, fill it on the CPU,
        # then copy host→device in one htod_memcpy call.
        mf_type = type(mfab)
        mf_pin = mf_type(
            mfab.box_array(),
            mfab.dm(),
            mfab.n_comp,
            mfab.n_grow_vect,
            amr.MFInfo().set_arena(amr.The_Pinned_Arena()),
            mfab.factory,
        )
        ngrow_r = mfab.n_grow_vect[0]
        ngrow_z = mfab.n_grow_vect[1]
        for mfi in mf_pin:
            # tilebox() covers valid cells only (no ghost cells).
            box = mfi.tilebox()
            r_lo, r_hi = box.small_end[0], box.big_end[0]
            z_lo, z_hi = box.small_end[1], box.big_end[1]
            local_patch = values_2d[
                r_lo - r0 : r_hi - r0 + 1,
                z_lo - z0 : z_hi - z0 + 1,
            ]
            # to_numpy(copy=False) on Pinned arena is safe (host memory).
            # Array4.to_numpy(order="F") shape includes all dimensions:
            # (nr_with_ghost, nz_with_ghost, [1,] ncomp).
            # Use squeeze() on trailing size-1 axes, then fill valid cells.
            pin_arr = mf_pin.array(mfi).to_numpy(copy=False, order="F")
            nr_valid = r_hi - r_lo + 1
            nz_valid = z_hi - z_lo + 1
            pin_2d = pin_arr[
                ngrow_r : ngrow_r + nr_valid, ngrow_z : ngrow_z + nz_valid, ...
            ]
            pin_2d.squeeze()[:] = local_patch
        amr.htod_memcpy(mfab, mf_pin)
    else:
        # CPU build: write directly into each FAB via numpy view.
        ngrow_r = mfab.n_grow_vect[0]
        ngrow_z = mfab.n_grow_vect[1]
        for mfi in mfab:
            box = mfi.tilebox()
            r_lo, r_hi = box.small_end[0], box.big_end[0]
            z_lo, z_hi = box.small_end[1], box.big_end[1]
            local_patch = values_2d[
                r_lo - r0 : r_hi - r0 + 1,
                z_lo - z0 : z_hi - z0 + 1,
            ]
            arr_np = mfab.array(mfi).to_numpy(copy=False, order="F")
            nr_valid = r_hi - r_lo + 1
            nz_valid = z_hi - z_lo + 1
            arr_2d = arr_np[
                ngrow_r : ngrow_r + nr_valid, ngrow_z : ngrow_z + nz_valid, ...
            ]
            arr_2d.squeeze()[:] = local_patch


def load_exact_fields():
    Bz_ext = sim.fields.get("Bfield_fp_external", dir="z", level=0)
    RM, ZM = np.meshgrid(Bz_ext.mesh("r"), Bz_ext.mesh("z"), indexing="ij")
    Bz_vals = B0 * scipy_j0(kr * RM) * np.sin(kz * ZM)
    _fill_mfab_from_numpy(Bz_ext, Bz_vals)

    Br_ext = sim.fields.get("Bfield_fp_external", dir="r", level=0)
    RM_r, ZM_r = np.meshgrid(Br_ext.mesh("r"), Br_ext.mesh("z"), indexing="ij")
    Br_vals = -B0 * (kz / kr) * scipy_j1(kr * RM_r) * np.cos(kz * ZM_r)
    _fill_mfab_from_numpy(Br_ext, Br_vals)


field_init = picmi.LoadInitialFieldFromPython(
    load_from_python=load_exact_fields,
    load_E=False,
    load_B=True,
)
sim.add_applied_field(field_init)

# Diagnostics: period=max_steps writes at step 0 and step max_steps.
diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=max_steps,
    data_list=["Er", "Et", "Ez", "Br", "Bt", "Bz"],
    write_dir="diags",
)
sim.add_diagnostic(diag)

sim.step(max_steps)
