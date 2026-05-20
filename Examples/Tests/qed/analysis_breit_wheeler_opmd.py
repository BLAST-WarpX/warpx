#!/usr/bin/env python3
# Copyright 2019 Luca Fedeli
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# -*- coding: utf-8 -*-

import sys

import analysis_breit_wheeler_core as ac
import openpmd_api as io

# This script is a frontend for the analysis routines
# in analysis_breit_wheeler_core.py (please refer to this file for
# a full description). It reads output files in openPMD
# format and extracts the data needed for
# the analysis routines.


def main():
    print("Opening openPMD output")
    prefix = sys.argv[1]
    series = io.Series(prefix + "/openpmd_%T.h5", io.Access.read_only)
    data_set_end = series.iterations[2]

    # The Breit-Wheeler pair-production probability is sampled stochastically
    # at each timestep, so the expected number of pairs after the test's
    # `max_step` timesteps follows `1 - exp(-dN/dt * sim_time)`. We therefore
    # use the total simulation time as the effective dt in the analysis.
    dt = data_set_end.time

    # get particle data
    particle_data = {}

    names, types = ac.get_all_species_names_and_types()
    for spec_name_type in zip(names, types):
        spec_name = spec_name_type[0]
        is_photon = spec_name_type[1]
        data = {}

        px = data_set_end.particles[spec_name]["momentum"]["x"][:]
        py = data_set_end.particles[spec_name]["momentum"]["y"][:]
        pz = data_set_end.particles[spec_name]["momentum"]["z"][:]
        w = data_set_end.particles[spec_name]["weighting"][
            io.Mesh_Record_Component.SCALAR
        ][:]

        if not is_photon:
            opt = data_set_end.particles[spec_name]["opticalDepthQSR"][
                io.Mesh_Record_Component.SCALAR
            ][:]

        series.flush()

        data["px"] = px
        data["py"] = py
        data["pz"] = pz
        data["w"] = w
        if not is_photon:
            data["opt"] = opt

        particle_data[spec_name] = data

    ac.check(dt, particle_data)


if __name__ == "__main__":
    main()
