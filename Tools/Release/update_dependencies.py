#!/usr/bin/env python3
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import requests


def update(args):
    # list of repositories to update
    repo_dict = {}
    if args.all or args.amrex:
        repo_dict["amrex"] = (
            "https://api.github.com/repos/AMReX-Codes/amrex/commits/development"
        )
    if args.all or args.pyamrex:
        repo_dict["pyamrex"] = (
            "https://api.github.com/repos/AMReX-Codes/pyamrex/commits/development"
        )
    if args.all or args.picsar:
        repo_dict["picsar"] = (
            "https://api.github.com/repos/ECP-WarpX/picsar/commits/development"
        )
    if args.all or args.warpx:
        repo_dict["warpx"] = (
            "https://api.github.com/repos/BLAST-WarpX/warpx/commits/development"
        )

    # list of repositories labels for logging convenience
    repo_labels = {
        "amrex": "AMReX",
        "pyamrex": "pyAMReX",
        "picsar": "PICSAR",
        "warpx": "WarpX",
    }

    # read from JSON file with dependencies data
    repo_dir = Path(__file__).parent.parent.parent.absolute()
    dependencies_file = os.path.join(repo_dir, "dependencies.json")
    try:
        with open(dependencies_file, "r") as file:
            dependencies_data = json.load(file)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit()

    # loop over repositories and update dependencies data
    for repo_name, repo_url in repo_dict.items():
        print(f"\nUpdating {repo_labels[repo_name]}...")
        # set keys to access dependencies data
        commit_key = f"commit_{repo_name}"
        version_key = f"version_{repo_name}"
        # set new repository commit
        repo_gh = requests.get(repo_url)
        repo_commit = repo_gh.json()["sha"]
        # set new repository version
        repo_version = datetime.date.today().strftime("%y.%m")
        # update repository commit
        if repo_name != "warpx":
            print(f"- old commit: {dependencies_data[commit_key]}")
            print(f"- new commit: {repo_commit}")
            proceed = input("Do you want to continue? [y/n] ")
            if proceed not in ["y", "Y"]:
                print("Skipping commit update...")
            else:
                print("Updating commit...")
                dependencies_data[f"commit_{repo_name}"] = repo_commit
        # update repository version
        print(f"- old version: {dependencies_data[version_key]}")
        print(f"- new version: {repo_version}")
        proceed = input("Do you want to continue? [y/n] ")
        if proceed not in ["y", "Y"]:
            print("Skipping version update...")
        else:
            print("Updating version...")
            dependencies_data[f"version_{repo_name}"] = repo_version

    # write to JSON file with dependencies data
    with open(dependencies_file, "w") as file:
        json.dump(dependencies_data, file, indent=4)


if __name__ == "__main__":
    # define parser
    parser = argparse.ArgumentParser()

    # add arguments: AMReX option
    parser.add_argument(
        "--amrex",
        help="Update AMReX only",
        action="store_true",
        dest="amrex",
    )

    # add arguments: pyAMReX option
    parser.add_argument(
        "--pyamrex",
        help="Update pyAMReX only",
        action="store_true",
        dest="pyamrex",
    )

    # add arguments: PICSAR option
    parser.add_argument(
        "--picsar",
        help="Update PICSAR only",
        action="store_true",
        dest="picsar",
    )

    # add arguments: WarpX option
    parser.add_argument(
        "--warpx",
        help="Update WarpX only",
        action="store_true",
        dest="warpx",
    )

    # parse arguments
    args = parser.parse_args()

    # set args.all automatically
    args.all = (
        False if (args.amrex or args.pyamrex or args.picsar or args.warpx) else True
    )

    # update
    update(args)
