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

    # get simulation time
    sim_time = data_set_end.time
    # no particles can be created on the first timestep so we have 2 timesteps in the test case,
    # with only the second one resulting in particle creation
    dt = sim_time / 2.0

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

        if is_photon:
            opt = data_set_end.particles[spec_name]["opticalDepthBW"][
                io.Mesh_Record_Component.SCALAR
            ][:]
        else:
            opt = data_set_end.particles[spec_name]["opticalDepthQSR"][
                io.Mesh_Record_Component.SCALAR
            ][:]

        series.flush()

        data["px"] = px
        data["py"] = py
        data["pz"] = pz
        data["w"] = w
        data["opt"] = opt

        particle_data[spec_name] = data

    ac.check(dt, particle_data)


if __name__ == "__main__":
    main()
