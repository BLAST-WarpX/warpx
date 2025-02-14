# Copyright 2023 Neil Zaim, Edoardo Zoni
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import json
import re
import sys

"""
This Python script updates the Azure benchmarks automatically using a raw
Azure output text file that is passed as command line argument of the script.
"""

# read path to Azure output text file
azure_output = sys.argv[1]

# string to identify failing tests that require a checksums reset
new_checksums = "New checksums"
failing_test = ""

# path of all checksums benchmark files
benchmark_path = "../../Regression/Checksum/benchmarks_json/"

with open(azure_output, "r") as f:
    # find length of Azure prefix to be removed from each line,
    # first line of Azure output starts with "##[section]Starting:"
    first_line = f.readline()
    prefix_length = first_line.find("#")
    # loop over lines
    for line in f:
        # remove Azure prefix from line
        line = line[prefix_length:]
        if failing_test == "":
            # no failing test found yet
            if re.search(new_checksums, line):
                # failing test found, set failing test name
                failing_test = line[line.find("test_") : line.find(".json")]
                json_file_string = ""
        else:
            # extract and dump new checksums of failing test
            json_file_string += line
            if line.startswith("}"):  # end of new checksums
                json_file = json.loads(json_file_string)
                json_filename = failing_test + ".json"
                json_filepath = benchmark_path + json_filename
                print(f"\nDumping new checksums file {json_filename}:")
                print(json_file_string)
                with open(json_filepath, "w") as json_f:
                    json.dump(json_file, json_f, indent=2)
                # reset to empty string to continue search of failing tests
                failing_test = ""
