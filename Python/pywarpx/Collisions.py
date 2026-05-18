# Copyright 2021 Modern Electron
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

from .Bucket import Bucket

collisions = Bucket("collisions")
collisions_list = []


def newcollision(name):
    result = Bucket(name)
    collisions_list.append(result)
    return result
