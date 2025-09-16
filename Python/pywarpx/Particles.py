# Copyright 2017-2020 Andrew Myers, David Grote
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

from .Bucket import Bucket

particles = Bucket("particles", species_names=[], rigid_injected_species=[])
particles_list = []


def new_species(name):
    result = Bucket(name)
    particles_list.append(result)
    return result


def valid_species(name):
    for sp in particles_list:
        if sp.instancename == name:
            return True
    return False
