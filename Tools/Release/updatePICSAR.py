#!/usr/bin/env python3
#
# Copyright 2021 Axel Huebl
#
# This file is part of WarpX.
#

# This file is a maintainer tool to bump the PICSAR version that we pull in
# when building WarpX.
#
import datetime
import re
import sys
from pathlib import Path

import requests

# Maintainer Inputs ###########################################################

print("""Hi there, this is a WarpX maintainer tool to update the source
code of WarpX to a new commit/release of PICSAR.
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

# PICSAR development HEAD
PICSAR_gh = requests.get(
    "https://api.github.com/repos/ECP-WarpX/picsar/commits/development"
)
PICSAR_HEAD = PICSAR_gh.json()["sha"]

# WarpX references to PICSAR: cmake/dependencies/PICSAR.cmake
PICSAR_cmake_path = str(REPO_DIR.joinpath("cmake/dependencies/PICSAR.cmake"))
#   branch/commit/tag (git fetcher) version
#     set(WarpX_picsar_branch "development" ...
PICSAR_branch = f"unknown (format issue in {PICSAR_cmake_path})"
with open(PICSAR_cmake_path, encoding="utf-8") as f:
    r_minimal = re.findall(
        r'.*set\(WarpX_picsar_branch\s+"(.+)"\s+.*', f.read(), re.MULTILINE
    )
    if len(r_minimal) >= 1:
        PICSAR_branch = r_minimal[0]

#   minimal (external) version
#     find_package(PICSAR YY.MM CONFIG ...
PICSAR_minimal = f"unknown (format issue in {PICSAR_cmake_path})"
with open(PICSAR_cmake_path, encoding="utf-8") as f:
    r_minimal = re.findall(
        r".*find_package\(PICSAR\s+(.+)\s+CONFIG\s+.*", f.read(), re.MULTILINE
    )
    if len(r_minimal) >= 1:
        PICSAR_minimal = r_minimal[0]


# Ask for new #################################################################

print("""We will now run a few sed commands on your source directory.
Please answer the following questions about the version number
you want to require from PICSAR:\n""")

print(f"Currently, WarpX builds against this PICSAR commit/branch/sha: {PICSAR_branch}")
print(f"PICSAR HEAD commit (development branch): {PICSAR_HEAD}")
PICSAR_new_branch = input("Update PICSAR commit/branch/sha: ").strip()
if not PICSAR_new_branch:
    PICSAR_new_branch = PICSAR_HEAD
    print(f"--> Nothing entered, will use: {PICSAR_HEAD}")
print()

print(
    f"Currently, a pre-installed PICSAR is required at least at version: {PICSAR_minimal}"
)
today = datetime.date.today().strftime("%y.%m")
PICSAR_new_minimal = input(f"New minimal PICSAR version (e.g. {today})? ").strip()
if not PICSAR_new_minimal:
    PICSAR_new_minimal = PICSAR_minimal
    print(f"--> Nothing entered, will keep: {PICSAR_minimal}")

print()
print(f"New PICSAR commit/branch/sha: {PICSAR_new_branch}")
print(f"New minimal PICSAR version:   {PICSAR_new_minimal}\n")

REPLY = input("Is this information correct? Will now start updating! [y/N] ")
print()
if REPLY not in ["Y", "y"]:
    print("You did not confirm with 'y', aborting.")
    sys.exit(1)


# Updates #####################################################################

# WarpX references to PICSAR: cmake/dependencies/PICSAR.cmake
with open(PICSAR_cmake_path, encoding="utf-8") as f:
    PICSAR_cmake_content = f.read()

    #   branch/commit/tag (git fetcher) version
    #     set(WarpX_picsar_branch "development" ...
    PICSAR_cmake_content = re.sub(
        r'(.*set\(WarpX_picsar_branch\s+")(.+)("\s+.*)',
        r"\g<1>{}\g<3>".format(PICSAR_new_branch),
        PICSAR_cmake_content,
        flags=re.MULTILINE,
    )

    #   minimal (external) version
    #     find_package(PICSAR YY.MM CONFIG ...
    PICSAR_cmake_content = re.sub(
        r"(.*find_package\(PICSAR\s+)(.+)(\s+CONFIG\s+.*)",
        r"\g<1>{}\g<3>".format(PICSAR_new_minimal),
        PICSAR_cmake_content,
        flags=re.MULTILINE,
    )

with open(PICSAR_cmake_path, "w", encoding="utf-8") as f:
    f.write(PICSAR_cmake_content)


# Epilogue ####################################################################

print("""Done. Please check your source, e.g. via
  git diff
now and commit the changes if no errors occurred.""")
