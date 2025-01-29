# Copyright 2016-2020 Andrew Myers, David Grote, Maxence Thevenet
#
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import numpy as np


class Bucket(object):
    """
    The purpose of this class is to be a named bucket for holding attributes.
    This attributes will be concatenated into a string and passed into argv during initialization.
    """

    def __init__(self, instancename, **defaults):
        self._localsetattr("instancename", instancename)
        self._localsetattr("argvattrs", {})
        for name, value in defaults.items():
            self.add_new_attr(name, value)

    def _localsetattr(self, name, value):
        object.__setattr__(self, name, value)

    def add_new_attr(self, name, value):
        """Names starting with "_" are made instance attributes.
        Otherwise the attribute is added to the args list.
        """
        if name.startswith("_"):
            self._localsetattr(name, value)
        else:
            self.argvattrs[name] = value

    def add_new_group_attr(self, group, name, value):
        """The attribute is added to the args list in the form "group.name" if
        group is not an empty string, otherwise as only "name".
        """
        if group:
            self.argvattrs[f"{group}.{name}"] = value
        else:
            self.argvattrs[name] = value

    def __setattr__(self, name, value):
        self.add_new_attr(name, value)

    def __getattr__(self, name):
        try:
            return self.argvattrs[name]
        except KeyError:
            return object.__getattribute__(self, name)

    def check_consistency(self, vname, value, errmsg):
        if vname in self.argvattrs:
            assert (self.argvattrs[vname] is None) or (
                self.argvattrs[vname] == value
            ), Exception(errmsg)

    def attrlist(self):
        "Concatenate the attributes into a string"
        result = []
        for attr, value in self.argvattrs.items():
            if value is None:
                continue
            if isinstance(value, str):
                if value.find("=") > -1:
                    # --- Expressions with temporary variables need to be inside quotes
                    rhs = f'"{value}"'
                else:
                    rhs = value
            elif np.iterable(value):
                if len(value) == 0:
                    # --- Skip if empty
                    continue
                # --- For lists, tuples, and arrays make a space delimited string of the values.
                # --- The lambda is needed in case this is a list of strings.
                rhs = " ".join(map(lambda s: f"{s}", value))
            elif isinstance(value, bool):
                rhs = 1 if value else 0
            else:
                rhs = value
            attrstring = f"{self.instancename}.{attr} = {rhs}"
            result += [attrstring]
        return result
