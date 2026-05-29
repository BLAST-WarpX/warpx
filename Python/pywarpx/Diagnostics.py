# Copyright 2017-2020 David Grote
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

from .Bucket import Bucket
from .Particles import valid_species

diagnostics = Bucket("diagnostics", _diagnostics_dict={})
reduced_diagnostics = Bucket("warpx", _diagnostics_dict={})


def new_diagnostic(name):
    diag = Diagnostic(name, _species_dict={})
    diagnostics._diagnostics_dict[name] = diag
    return diag


class Diagnostic(Bucket):
    """
    This is the same as a Bucket, but checks that any attributes are always given the same value.
    """

    def add_new_attr_with_check(self, name, value):
        if name.startswith("_"):
            self._localsetattr(name, value)
        else:
            if name in self.argvattrs:
                assert value == self.argvattrs[name], Exception(
                    f"Diagnostic attributes not consistent for "
                    f'"{self.instancename}": '
                    f'"{value}" != "{self.argvattrs[name]}"'
                )
            self.argvattrs[name] = value

    def __getattr__(self, name):
        try:
            return Bucket.__getattr__(self, name)
        except AttributeError:
            # Create a new attibute if the name is a valid species name
            if not valid_species(name):
                raise AttributeError(
                    "Only valid species names can be added as new attributes"
                )
            new = Bucket(f"{self.instancename}.{name}")
            self.argvattrs[name] = new
            return new

    def __setattr__(self, name, value):
        self.add_new_attr_with_check(name, value)

    def set_or_replace_attr(self, name, value):
        """
        Explicitly set or replace an existing attribute
        (since __setattr__ cannot be used for replacing
        as it would raise an Exception)
        """
        assert not name.startswith("_")
        self.argvattrs[name] = value
