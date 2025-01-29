"""
Copyright 2020

This file is part of WarpX.

License: BSD-3-Clause-LBNL
"""

import json
import os

import config


class Benchmark:
    """
    Holds data and functions for referenc benchmark of one checksum test.
    """

    def __init__(self, test_name, data=None):
        """
        Benchmark constructor.
        Store test name and reference checksum value, either from benchmark
        (used for comparison) or from a plotfile (used to reset a benchmark).

        Parameters
        ----------
        test_name: string
            Name of test, as found between [] in .ini file.

        data: dictionary, optional
            Checksum value.
            If None, it is read from benchmark.
        """

        self.test_name = test_name
        self.json_file = os.path.join(
            config.benchmark_location, self.test_name + ".json"
        )
        if data is None:
            self.data = self.get()
        else:
            self.data = data

    def reset(self):
        """
        Update the benchmark (overwrites reference json file).
        """
        with open(self.json_file, "w") as outfile:
            json.dump(self.data, outfile, sort_keys=True, indent=2)

    def get(self):
        """
        Read benchmark from reference json file.
        """
        with open(self.json_file) as infile:
            data = json.load(infile)

        return data
