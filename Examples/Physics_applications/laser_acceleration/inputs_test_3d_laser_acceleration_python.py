#!/usr/bin/env python3
#
# Starting from an inputs file, define a WarpX simulation
# and extend it with Python logic.

from pywarpx import warpx
from pywarpx.callbacks import callfromafterstep

sim = warpx
sim.load_inputs_file("./inputs_test_3d_laser_acceleration")


# Optional: Define callbacks, e.g., after every step
@callfromafterstep
def my_simple_callback():
    """This simple callback uses particle container and MultiFab objects,
    https://warpx.readthedocs.io/en/latest/usage/workflows/python_extend.html#particles
    and
    https://warpx.readthedocs.io/en/latest/usage/workflows/python_extend.html#fields
    """
    print("  my_simple_callback")

    # electrons: access (and potentially manipulate)
    electrons = sim.particles.get("electrons")
    print(f"    {electrons}")

    # electric field: access (and potentially manipulate)
    Ex = sim.fields.get("Efield_fp", dir="x", level=0)
    print(f"    {Ex}")


@callfromafterstep
def my_advanced_callback():
    """This callback dives deeper using pyAMReX methods and data containers directly.
    https://pyamrex.readthedocs.io/en/latest/usage/compute.html
    """
    print("  my_advanced_callback")

    # the pyAMReX module
    amr = sim.extension.amr
    amr.Print(f"    {amr.ParallelDescriptor.NProcs()} MPI process(es) active")

    # electrons: access (and potentially manipulate)
    electrons = sim.particles.get("electrons")
    print(f"    {electrons}")

    # electric field: access (and potentially manipulate)
    Ex_mf = sim.fields.get("Efield_fp", dir="x", level=0)
    print(f"    {Ex_mf}")


# Advance simulation until the last time step
sim.step()
