#!/usr/bin/env python3
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import argparse
import copy
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
        repo_dict["amrex"] = {}
        repo_dict["amrex"]["commit"] = (
            "https://api.github.com/repos/AMReX-Codes/amrex/commits/development"
        )
        repo_dict["amrex"]["tags"] = (
            "https://api.github.com/repos/AMReX-Codes/amrex/tags"
        )
    if args.all or args.pyamrex:
        repo_dict["pyamrex"] = {}
        repo_dict["pyamrex"]["commit"] = (
            "https://api.github.com/repos/AMReX-Codes/pyamrex/commits/development"
        )
        repo_dict["pyamrex"]["tags"] = (
            "https://api.github.com/repos/AMReX-Codes/pyamrex/tags"
        )
    if args.all or args.picsar:
        repo_dict["picsar"] = {}
        repo_dict["picsar"]["commit"] = (
            "https://api.github.com/repos/ECP-WarpX/picsar/commits/development"
        )
        repo_dict["picsar"]["tags"] = (
            "https://api.github.com/repos/ECP-WarpX/picsar/tags"
        )
    if args.all or args.pybind11:
        repo_dict["pybind11"] = {}
        repo_dict["pybind11"]["commit"] = (
            "https://api.github.com/repos/pybind/pybind11/commits/master"
        )
        repo_dict["pybind11"]["tags"] = (
            "https://api.github.com/repos/pybind/pybind11/tags"
        )
    if args.all or args.picmi:
        repo_dict["picmi"] = {}
        repo_dict["picmi"]["commit"] = (
            "https://api.github.com/repos/picmi-standard/picmi/commits/master"
        )
        repo_dict["picmi"]["tags"] = (
            "https://api.github.com/repos/picmi-standard/picmi/tags"
        )
    if args.all or args.warpx:
        repo_dict["warpx"] = {}
        repo_dict["warpx"]["commit"] = (
            "https://api.github.com/repos/BLAST-WarpX/warpx/commits/development"
        )
        repo_dict["warpx"]["tags"] = (
            "https://api.github.com/repos/BLAST-WarpX/warpx/tags"
        )

    # list of repositories labels for logging convenience
    repo_labels = {
        "amrex": "AMReX",
        "pyamrex": "pyAMReX",
        "picsar": "PICSAR",
        "pybind11": "pybind11",
        "picmi": "PICMI",
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
    for repo_name, repo_subdict in repo_dict.items():
        print(f"\nUpdating {repo_labels[repo_name]}...")

        # set keys to access dependencies data
        commit_key = f"commit_{repo_name}"
        version_key = f"version_{repo_name}"

        # get new commit information
        commit_response = requests.get(repo_subdict["commit"])
        commit_dict = commit_response.json()

        # set new commit
        repo_commit_sha = commit_dict["sha"]

        # get new version tag information
        tags_response = requests.get(repo_subdict["tags"])
        tags_list = tags_response.json()

        # filter out old-format tags for specific repositories
        tags_list_filtered = copy.deepcopy(tags_list)
        if repo_name == "amrex":
            tags_list_filtered = [
                tag_dict
                for tag_dict in tags_list
                if (tag_dict["name"] != "boxlib" and tag_dict["name"] != "v2024")
            ]
        elif repo_name == "picsar":
            tags_list_filtered = [
                tag_dict
                for tag_dict in tags_list
                if (tag_dict["name"] != "PICSARlite-0.1")
            ]

        # set new version tag
        if repo_name == "warpx":
            # current date version for the WarpX release update
            repo_version_tag = datetime.date.today().strftime("%y.%m")
        else:
            # latest available tag (index 0) for all other dependencies
            repo_version_tag = tags_list_filtered[0]["name"]

        # update commit
        if repo_name != "warpx":
            # use version tag instead of commit sha:
            # - for a release update
            # - for pybind11 (always)
            # - if the commit has not changed since the last version tag
            use_version_tag = (
                args.release
                or (repo_name == "pybind11")
                or (repo_commit_sha == tags_list_filtered[0]["commit"]["sha"])
            )
            new_commit_sha = repo_version_tag if use_version_tag else repo_commit_sha
            print(f"- old commit: {dependencies_data[commit_key]}")
            print(f"- new commit: {new_commit_sha}")
            if dependencies_data[commit_key] == new_commit_sha:
                print("Skipping commit update...")
            else:
                print("Updating commit...")
                dependencies_data[f"commit_{repo_name}"] = new_commit_sha

        # update version
        print(f"- old version: {dependencies_data[version_key]}")
        print(f"- new version: {repo_version_tag}")
        if dependencies_data[version_key] == repo_version_tag:
            print("Skipping version update...")
        else:
            print("Updating version...")
            dependencies_data[f"version_{repo_name}"] = repo_version_tag

            # update PICMI version in requirements.txt files manually
            if repo_name == "picmi":
                files = [
                    os.path.join(repo_dir, "requirements.txt"),
                    os.path.join(repo_dir, "Docs", "requirements.txt"),
                ]
                for filename in files:
                    with open(filename) as f:
                        lines = f.readlines()
                    with open(filename, "w") as f:
                        for line in lines:
                            if line.startswith("picmistandard=="):
                                line = f"picmistandard=={repo_version_tag}\n"
                            f.write(line)

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

    # add arguments: pybind11 option
    parser.add_argument(
        "--pybind11",
        help="Update pybind11 only",
        action="store_true",
        dest="pybind11",
    )

    # add arguments: PICMI option
    parser.add_argument(
        "--picmi",
        help="Update PICMI only",
        action="store_true",
        dest="picmi",
    )

    # add arguments: WarpX option
    parser.add_argument(
        "--warpx",
        help="Update WarpX only",
        action="store_true",
        dest="warpx",
    )

    # add arguments: release option
    parser.add_argument(
        "--release",
        help="New release",
        action="store_true",
        dest="release",
    )

    # parse arguments
    args = parser.parse_args()

    # set args.all automatically
    args.all = (
        False
        if (
            args.amrex
            or args.pyamrex
            or args.picsar
            or args.pybind11
            or args.picmi
            or args.warpx
        )
        else True
    )

    # update
    update(args)
