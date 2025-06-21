#!/usr/bin/env python3
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import argparse
import datetime
import json

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

    # list of repositories labels for logging convenience
    repo_labels = {"amrex": "AMReX", "pyamrex": "pyAMReX", "picsar": "PICSAR"}

    # read from JSON file with configuration data
    config_file = "../../config.json"
    with open(config_file, "r") as file:
        config_data = json.load(file)

    # loop over repositories and update configuration data
    for repo_name, repo_url in repo_dict.items():
        print(f"\nUpdating {repo_labels[repo_name]} dependency...")
        # set keys to access configuration data
        commit_key = f"commit_{repo_name}"
        version_key = f"version_{repo_name}"
        # set new repository commit
        repo_gh = requests.get(repo_url)
        repo_commit = repo_gh.json()["sha"]
        # set new repository version
        repo_version = datetime.date.today().strftime("%y.%m")
        # update repository commit
        print(f"- old commit/branch/sha: {config_data[commit_key]}")
        print(f"- new commit/branch/sha: {repo_commit}")
        proceed = input("Do you want to continue? [y/n] ")
        if proceed not in ["y", "Y"]:
            print("Skipping update of commit/branch/sha...")
        else:
            print("Updating commit/branch/sha...")
            config_data[f"commit_{repo_name}"] = repo_commit
        # update repository version
        print(f"- old minimal version required: {config_data[version_key]}")
        print(f"- new minimal version required: {repo_version}")
        proceed = input("Do you want to continue? [y/n] ")
        if proceed not in ["y", "Y"]:
            print("Skipping update of minimal version required...")
        else:
            print("Updating minimal version required...")
            config_data[f"version_{repo_name}"] = repo_version

    # write to JSON file with configuration data
    with open(config_file, "w") as file:
        json.dump(config_data, file, indent=4)


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

    # parse arguments
    args = parser.parse_args()

    # set args.all automatically
    args.all = False if (args.amrex or args.pyamrex or args.picsar) else True

    # update
    update(args)
