#!/usr/bin/env python3
#
# Copyright 2021 Axel Huebl
#
# This file is part of WarpX.
#

# This file is a maintainer tool to bump the pyAMReX version that we pull in
# when building WarpX.
#
import datetime
import re
import sys
from pathlib import Path

import requests

# Maintainer Inputs ###########################################################

print("""Hi there, this is a WarpX maintainer tool to update the source
code of WarpX to a new commit/release of pyAMReX.
For it to work, you need write access on the source directory and
you should be working in a clean git branch without ongoing
rebase/merge/conflict resolves and without unstaged changes.""")

# check source dir
REPO_DIR = Path(__file__).parent.parent.parent.absolute()
print(f"\nYour current source directory is: {REPO_DIR}")

REPLY = input("Are you sure you want to continue? [y/N] ")
print()
if REPLY not in ["Y", "y"]:
    print("You did not confirm with 'y', aborting.")
    sys.exit(1)


# Current Versions ############################################################

# pyAMReX development HEAD
pyamrex_gh = requests.get(
    "https://api.github.com/repos/AMReX-Codes/pyamrex/commits/development"
)
pyamrex_HEAD = pyamrex_gh.json()["sha"]

# WarpX references to pyAMReX: cmake/dependencies/pyAMReX.cmake
pyamrex_cmake_path = str(REPO_DIR.joinpath("cmake/dependencies/pyAMReX.cmake"))
#   branch/commit/tag (git fetcher) version
#     set(WarpX_pyamrex_branch "development" ...
pyamrex_branch = f"unknown (format issue in {pyamrex_cmake_path})"
with open(pyamrex_cmake_path, encoding="utf-8") as f:
    r_minimal = re.findall(
        r'.*set\(WarpX_pyamrex_branch\s+"(.+)"\s+.*', f.read(), re.MULTILINE
    )
    if len(r_minimal) >= 1:
        pyamrex_branch = r_minimal[0]

#   minimal (external) version
#     find_package(AMReX YY.MM CONFIG ...
pyamrex_minimal = f"unknown (format issue in {pyamrex_cmake_path})"
with open(pyamrex_cmake_path, encoding="utf-8") as f:
    r_minimal = re.findall(
        r".*find_package\(pyAMReX\s+(.+)\s+CONFIG\s+.*", f.read(), re.MULTILINE
    )
    if len(r_minimal) >= 1:
        pyamrex_minimal = r_minimal[0]


# Ask for new #################################################################

print("""We will now run a few sed commands on your source directory.
Please answer the following questions about the version number
you want to require from pyAMReX:\n""")

print(
    f"Currently, WarpX builds against this pyAMReX commit/branch/sha: {pyamrex_branch}"
)
print(f"pyAMReX HEAD commit (development branch): {pyamrex_HEAD}")
pyamrex_new_branch = input("Update pyAMReX commit/branch/sha: ").strip()
if not pyamrex_new_branch:
    pyamrex_new_branch = pyamrex_HEAD
    print(f"--> Nothing entered, will use: {pyamrex_HEAD}")
print()

print(
    f"Currently, a pre-installed pyAMReX is required at least at version: {pyamrex_minimal}"
)
today = datetime.date.today().strftime("%y.%m")
pyamrex_new_minimal = input(f"New minimal pyAMReX version (e.g. {today})? ").strip()
if not pyamrex_new_minimal:
    pyamrex_new_minimal = pyamrex_minimal
    print(f"--> Nothing entered, will keep: {pyamrex_minimal}")

print()
print(f"New pyAMReX commit/branch/sha: {pyamrex_new_branch}")
print(f"New minimal pyAMReX version:   {pyamrex_new_minimal}\n")

REPLY = input("Is this information correct? Will now start updating! [y/N] ")
print()
if REPLY not in ["Y", "y"]:
    print("You did not confirm with 'y', aborting.")
    sys.exit(1)


# Updates #####################################################################

# WarpX references to pyAMReX: cmake/dependencies/pyAMReX.cmake
with open(pyamrex_cmake_path, encoding="utf-8") as f:
    pyAMReX_cmake_content = f.read()

    #   branch/commit/tag (git fetcher) version
    #     set(WarpX_pyamrex_branch "development" ...
    pyAMReX_cmake_content = re.sub(
        r'(.*set\(WarpX_pyamrex_branch\s+")(.+)("\s+.*)',
        r"\g<1>{}\g<3>".format(pyamrex_new_branch),
        pyAMReX_cmake_content,
        flags=re.MULTILINE,
    )

    #   minimal (external) version
    #     find_package(AMReX YY.MM CONFIG ...
    pyAMReX_cmake_content = re.sub(
        r"(.*find_package\(pyAMReX\s+)(.+)(\s+CONFIG\s+.*)",
        r"\g<1>{}\g<3>".format(pyamrex_new_minimal),
        pyAMReX_cmake_content,
        flags=re.MULTILINE,
    )

with open(pyamrex_cmake_path, "w", encoding="utf-8") as f:
    f.write(pyAMReX_cmake_content)


# Epilogue ####################################################################

print("""Done. Please check your source, e.g. via
  git diff
now and commit the changes if no errors occurred.""")
