#!/usr/bin/env python3

from pywarpx import fields, particle_containers, warpx
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
    Ex = fields.ExWrapper(level=0)
    print(Ex)


# Advance simulation until last time step
sim.evolve()
