#!/usr/bin/env python3

from pywarpx import warpx

sim = warpx
sim.load_inputs_file("./inputs_test_3d_laser_acceleration")

# Advance simulation until last time step
sim.evolve()
