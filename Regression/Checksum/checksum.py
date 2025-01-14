"""
Copyright 2020

This file is part of WarpX.

License: BSD-3-Clause-LBNL
"""

import json
import sys

import numpy as np
import yt
from benchmark import Benchmark
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c

yt.funcs.mylog.setLevel(50)


class Checksum:
    """Class for checksum comparison of one test."""

    def __init__(
        self,
        test_name,
        output_file,
        output_format="plotfile",
        do_fields=True,
        do_particles=True,
    ):
        """
        Checksum constructor.
        Store test_name, output file name and format, compute checksum
        from output file and store it in self.data.

        Parameters
        ----------
        test_name: string
            Name of test, as found between [] in .ini file.

        output_file: string
            Output file from which the checksum is computed.

        output_format: string
            Format of the output file (plotfile, openpmd).

        do_fields: bool, default=True
            Whether to compare fields in the checksum.

        do_particles: bool, default=True
            Whether to compare particles in the checksum.
        """

        self.test_name = test_name
        self.output_file = output_file
        self.output_format = output_format
        self.data = self.read_output_file(
            do_fields=do_fields, do_particles=do_particles
        )

    def read_output_file(self, do_fields=True, do_particles=True):
        """
        Get checksum from output file.
        Read an AMReX plotfile with yt or an openPMD file with openPMD viewer,
        compute 1 checksum per field and return all checksums in a dictionary.
        The checksum of quantity Q is max(abs(Q)).

        Parameters
        ----------
        do_fields: bool, default=True
            Whether to read fields from the output file.

        do_particles: bool, default=True
            Whether to read particles from the output file.
        """

        if self.output_format == "plotfile":
            ds = yt.load(self.output_file)
            # yt 4.0+ has rounding issues with our domain data:
            # RuntimeError: yt attempted to read outside the boundaries
            # of a non-periodic domain along dimension 0.
            if "force_periodicity" in dir(ds):
                ds.force_periodicity()
            grid_fields = [item for item in ds.field_list if item[0] == "boxlib"]

            # "fields"/"species" we remove:
            #   - nbody: added by yt by default, unused by us
            species_list = set(
                [
                    item[0]
                    for item in ds.field_list
                    if item[1][:9] == "particle_"
                    and item[0] != "all"
                    and item[0] != "nbody"
                ]
            )

            data = {}

            # Compute checksum for field quantities
            if do_fields:
                for lev in range(ds.max_level + 1):
                    data_lev = {}
                    lev_grids = [grid for grid in ds.index.grids if grid.Level == lev]
                    # Warning: For now, we assume all levels are rectangular
                    LeftEdge = np.min(
                        np.array([grid.LeftEdge.v for grid in lev_grids]), axis=0
                    )
                    all_data_level = ds.covering_grid(
                        level=lev, left_edge=LeftEdge, dims=ds.domain_dimensions
                    )
                    for field in grid_fields:
                        Q = all_data_level[field].v.squeeze()
                        data_lev[field[1]] = np.sum(np.abs(Q))
                    data["lev=" + str(lev)] = data_lev

            # Compute checksum for particle quantities
            if do_particles:
                ad = ds.all_data()
                for species in species_list:
                    # properties we remove:
                    #   - particle_cpu/id: they depend on the parallelism: MPI-ranks and
                    #                      on-node acceleration scheme, thus not portable
                    #                      and irrelevant for physics benchmarking
                    part_fields = [
                        item[1]
                        for item in ds.field_list
                        if item[0] == species
                        and item[1] != "particle_cpu"
                        and item[1] != "particle_id"
                    ]
                    data_species = {}
                    for field in part_fields:
                        Q = ad[(species, field)].v
                        data_species[field] = np.sum(np.abs(Q))
                    data[species] = data_species

        elif self.output_format == "openpmd":
            # Load time series
            ts = OpenPMDTimeSeries(self.output_file)
            data = {}
            # Compute number of MR levels
            # TODO This calculation of nlevels assumes that the last element
            #      of level_fields is by default on the highest MR level.
            level_fields = [field for field in ts.avail_fields if "lvl" in field]
            nlevels = 0 if level_fields == [] else int(level_fields[-1][-1])
            # Compute checksum for field quantities
            if do_fields:
                for lev in range(nlevels + 1):
                    # Create list of fields specific to level lev
                    grid_fields = []
                    if lev == 0:
                        grid_fields = [
                            field for field in ts.avail_fields if "lvl" not in field
                        ]
                    else:
                        grid_fields = [
                            field for field in ts.avail_fields if f"lvl{lev}" in field
                        ]
                    data_lev = {}
                    for field in grid_fields:
                        vector_components = ts.fields_metadata[field][
                            "avail_components"
                        ]
                        if vector_components != []:
                            for coord in vector_components:
                                Q, info = ts.get_field(
                                    field=field,
                                    iteration=ts.iterations[-1],
                                    coord=coord,
                                )
                                # key stores strings composed of field name and vector components
                                # (e.g., field='B' or field='B_lvl1' + coord='y' results in key='By')
                                key = field.replace(f"_lvl{lev}", "") + coord
                                data_lev[key] = np.sum(np.abs(Q))
                        else:  # scalar field
                            Q, info = ts.get_field(
                                field=field, iteration=ts.iterations[-1]
                            )
                            data_lev[field] = np.sum(np.abs(Q))
                    data[f"lev={lev}"] = data_lev
            # Compute checksum for particle quantities
            if do_particles:
                species_list = []
                if ts.avail_record_components is not None:
                    species_list = ts.avail_record_components.keys()
                for species in species_list:
                    data_species = {}
                    part_fields = [
                        item
                        for item in ts.avail_record_components[species]
                        if item != "id" and item != "charge" and item != "mass"
                    ]
                    # Convert the field name to the name used in plotfiles
                    for field in part_fields:
                        Q = ts.get_particle(
                            var_list=[field],
                            species=species,
                            iteration=ts.iterations[-1],
                        )
                        if field in ["x", "y", "z"]:
                            field_name = "particle_position_" + field
                        elif field in ["ux", "uy", "uz"]:
                            field_name = "particle_momentum_" + field[-1]
                            (m,) = ts.get_particle(
                                ["mass"], species=species, iteration=ts.iterations[-1]
                            )
                            Q *= m * c
                        elif field in ["w"]:
                            field_name = "particle_weight"
                        else:
                            field_name = "particle_" + field
                        data_species[field_name] = np.sum(np.abs(Q))
                    data[species] = data_species

        return data

    def evaluate(self, rtol=1.0e-9, atol=1.0e-40):
        """
        Compare output file checksum with benchmark.
        Read checksum from output file, read benchmark
        corresponding to test_name, and assert that they are equal.
        Almost all the body of this functions is for
        user-readable print statements.

        Parameters
        ----------
        rtol: float, default=1.e-9
            Relative tolerance on the benchmark

        atol: float, default=1.e-40
            Absolute tolerance on the benchmark
        """

        ref_benchmark = Benchmark(self.test_name)

        # Dictionaries have same outer keys (levels, species)?
        if self.data.keys() != ref_benchmark.data.keys():
            print(
                "ERROR: Benchmark and output file checksum have different outer keys:"
            )
            print("Benchmark: %s" % ref_benchmark.data.keys())
            print("Test file: %s" % self.data.keys())
            print(f"\nNew checksums file {self.test_name}.json:")
            print(json.dumps(self.data, indent=2))
            sys.exit(1)

        # Dictionaries have same inner keys (field and particle quantities)?
        for key1 in ref_benchmark.data.keys():
            if self.data[key1].keys() != ref_benchmark.data[key1].keys():
                print(
                    "ERROR: Benchmark and output file checksum have "
                    "different inner keys:"
                )
                print("Common outer keys: %s" % ref_benchmark.data.keys())
                print(
                    "Benchmark inner keys in %s: %s"
                    % (key1, ref_benchmark.data[key1].keys())
                )
                print("Test file inner keys in %s: %s" % (key1, self.data[key1].keys()))
                print(f"\nNew checksums file {self.test_name}.json:")
                print(json.dumps(self.data, indent=2))
                sys.exit(1)

        # Dictionaries have same values?
        checksums_differ = False
        for key1 in ref_benchmark.data.keys():
            for key2 in ref_benchmark.data[key1].keys():
                passed = np.isclose(
                    self.data[key1][key2],
                    ref_benchmark.data[key1][key2],
                    rtol=rtol,
                    atol=atol,
                )
                if not passed:
                    print(
                        "ERROR: Benchmark and output file checksum have "
                        "different value for key [%s,%s]" % (key1, key2)
                    )
                    print(
                        "Benchmark: [%s,%s] %.15e"
                        % (key1, key2, ref_benchmark.data[key1][key2])
                    )
                    print(
                        "Test file: [%s,%s] %.15e" % (key1, key2, self.data[key1][key2])
                    )
                    checksums_differ = True
                    # Print absolute and relative error for each failing key
                    x = ref_benchmark.data[key1][key2]
                    y = self.data[key1][key2]
                    abs_err = np.abs(x - y)
                    print("Absolute error: {:.2e}".format(abs_err))
                    if np.abs(x) != 0.0:
                        rel_err = abs_err / np.abs(x)
                        print("Relative error: {:.2e}".format(rel_err))
        if checksums_differ:
            print(f"\nNew checksums file {self.test_name}.json:")
            print(json.dumps(self.data, indent=2))
            sys.exit(1)
