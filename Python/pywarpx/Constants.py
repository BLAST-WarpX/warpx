# Copyright 2018-2019 David Grote
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import re

from .Bucket import Bucket


class Constants(Bucket):
    """
    The purpose of this class is to be hold user defined constants
    """

    def __init__(self):
        Bucket.__init__(self, "my_constants")

    def __setattr__(self, name, value):
        # Make sure that any constants redefined have a consistent value
        if name in self.argvattrs:
            assert self.argvattrs[name] == value, Exception(
                "Inconsistent values given for user defined constants"
            )
        Bucket.__setattr__(self, name, value)

    def add_keywords(self, kwdict):
        mangle_dict = {}
        for k, v in kwdict.items():
            # WarpX has a single global dictionary of expression variables, my_constants,
            # so each variable must be unique.
            # Check if keyword has already been defined. If so and it has a different
            # value than the already defined value, then mangle the name.
            mangle_number = 0
            k_mangled = k
            while k_mangled in self.argvattrs and self.argvattrs[k_mangled] != v:
                mangle_number += 1
                k_mangled = f"{k}{mangle_number}"
            if mangle_number > 0:
                # The mangle_dict contains only mangled names
                mangle_dict[k] = k_mangled
            setattr(self, k_mangled, v)
        return mangle_dict

    def mangle_expression(self, expression, mangle_dict):
        if expression is None:
            return None
        # For each key in mangle_dict, modify the expression replacing
        # the key with its value, the mangled version of key
        for k, v in mangle_dict.items():
            expression = re.sub(r"\b%s\b" % k, v, expression)
        return expression


my_constants = Constants()
