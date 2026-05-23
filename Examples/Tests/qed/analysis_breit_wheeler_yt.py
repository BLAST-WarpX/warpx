#!/usr/bin/env python3
# Copyright 2019 Luca Fedeli
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

# -*- coding: utf-8 -*-

import sys

import analysis_breit_wheeler_core as ac
import yt

# This script is a frontend for the analysis routines
# in analysis_breit_wheeler_core.py (please refer to this file for
# a full description). It reads output files in yt
# format and extracts the data needed for
# the analysis routines.
yt


def main():
    print("Opening yt output")
    filename_end = sys.argv[1]
    data_set_end = yt.load(filename_end)

    # The Breit-Wheeler pair-production probability is sampled stochastically
    # at each timestep, so the expected number of pairs after the test's
    # `max_step` timesteps follows `1 - exp(-dN/dt * sim_time)`. We therefore
    # use the total simulation time as the effective dt in the analysis.
    dt = data_set_end.current_time.to_value()

    # get particle data
    all_data_end = data_set_end.all_data()
    particle_data = {}

    names, types = ac.get_all_species_names_and_types()
    for spec_name_type in zip(names, types):
        spec_name = spec_name_type[0]
        is_photon = spec_name_type[1]
        data = {}
        data["px"] = all_data_end[spec_name, "particle_momentum_x"].v
        data["py"] = all_data_end[spec_name, "particle_momentum_y"].v
        data["pz"] = all_data_end[spec_name, "particle_momentum_z"].v
        data["w"] = all_data_end[spec_name, "particle_weighting"].v

        if not is_photon:
            data["opt"] = all_data_end[spec_name, "particle_opticalDepthQSR"].v

        particle_data[spec_name] = data

    ac.check(dt, particle_data)


if __name__ == "__main__":
    main()
