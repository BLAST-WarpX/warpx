# Copyright 2017-2020 Andrew Myers, David Grote
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

from .Bucket import Bucket

particles = Bucket("particles", species_names=[], rigid_injected_species=[])
particles_list = []


def new_species(name):
    result = Species(name)
    particles_list.append(result)
    return result


def valid_species(name):
    for sp in particles_list:
        if sp.instancename == name:
            return True
    return False


class Species(Bucket):
    def __getattr__(self, name):
        try:
            return Bucket.__getattr__(self, name)
        except AttributeError:
            # Create a new attibute if the name is a valid injection source name
            if name not in self.injection_sources:
                raise AttributeError(
                    "Only valid injection source names can be added as new attributes"
                )
            new = Bucket(f"{self.instancename}.{name}")
            self.argvattrs[name] = new
            return new
