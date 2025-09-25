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
    """This simple callback uses high-level wrappers in WarpX.
    https://warpx.readthedocs.io/en/latest/usage/workflows/python_extend.html#high-level-field-wrapper
    """
    from pywarpx import fields, particle_containers

    print("  my_simple_callback")

    # electrons: access (and potentially manipulate)
    electrons = particle_containers.ParticleContainerWrapper("electrons")
    print(f"    {electrons}")

    # electric field: access (and potentially manipulate)
    Ex = fields.ExWrapper(level=0)
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

    # the active simulation
    warpx = sim.extension.warpx

    # electrons: access (and potentially manipulate)
    electrons_pc = warpx.multi_particle_container().get_particle_container_from_name(
        "electrons"
    )
    print(f"    {electrons_pc}")

    # electric field: access (and potentially manipulate)
    multifab_register = warpx.multifab_register()
    Ex_mf = multifab_register.get("Efield_fp", dir="x", level=0)
    print(f"    {Ex_mf}")


# Advance simulation until the last time step
sim.evolve()
