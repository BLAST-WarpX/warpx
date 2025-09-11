#!/usr/bin/env python3

from pywarpx import libwarpx, particle_containers, warpx

# Callback locations: https://warpx.readthedocs.io/en/latest/usage/workflows/python_extend.html#callback-locations
from pywarpx.callbacks import callfromafterstep

# Define simulation from inputs file
sim = warpx
sim.load_inputs_file("./inputs_test_3d_laser_acceleration")


# Optional: Define callbacks, e.g., after every step
@callfromafterstep
def my_callback():
    # electrons: access (and potentially manipulate)
    electrons = particle_containers.ParticleContainerWrapper("electrons")
    print(electrons)

    # electric field: access (and potentially manipulate)
    multifab_register = libwarpx.warpx.multifab_register()
    #   vector field E, component x, on the fine patch of MR level 0
    dir_x = libwarpx.libwarpx_so.Direction.x
    E_x_mf = multifab_register.get("Efield_fp", dir=dir_x, level=0)
    print(E_x_mf)


# Advance simulation until last time step
sim.evolve()
