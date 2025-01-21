#!/usr/bin/env python3
#
# Copyright 2025 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl
#

# This file is a maintainer tool to open a release PR for WarpX.
# It is highly automated and does a few assumptions, e.g., that you
# are releasing for the current month.
#
# You also need to have git and the GitHub CLI tool "gh" installed and properly
# configured for it to work:
# https://cli.github.com/
#
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Maintainer Inputs ###########################################################

print("""Hi there, this is a WarpX maintainer tool to ...\n.
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

release_repo = input("What is the name of your git remote? (e.g., ax3l) ")
commit_sign = input("How to sign the commit? (e.g., -sS) ")


# Helpers #####################################################################


def concat_answers(answers):
    return "\n".join(answers) + "\n"


# Stash current work ##########################################################

subprocess.run(["git", "stash"], capture_output=True, text=True)


# Git Branch ##################################################################

WarpX_version_yr = f"{datetime.now().strftime('%y')}"
WarpX_version_mn = f"{datetime.now().strftime('%m')}"
WarpX_version = f"{WarpX_version_yr}.{WarpX_version_mn}"
release_branch = f"release-{WarpX_version}"
subprocess.run(["git", "checkout", "development"], capture_output=True, text=True)
subprocess.run(["git", "fetch"], capture_output=True, text=True)
subprocess.run(["git", "pull", "--ff-only"], capture_output=True, text=True)
subprocess.run(["git", "branch", "-D", release_branch], capture_output=True, text=True)
subprocess.run(
    ["git", "checkout", "-b", release_branch], capture_output=True, text=True
)


# AMReX New Version ###########################################################

AMReX_version = f"{datetime.now().strftime('%y')}.{datetime.now().strftime('%m')}"
answers = concat_answers(["y", AMReX_version, AMReX_version, "y"])

process = subprocess.Popen(
    [Path(REPO_DIR).joinpath("Tools/Release/updateAMReX.py")],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

process.communicate(answers)
del process

# commit
subprocess.run(["git", "add", "-u"], capture_output=True, text=True)
subprocess.run(
    ["git", "commit", commit_sign, "-m", f"AMReX: {AMReX_version}"], text=True
)


# PICSAR New Version ##########################################################

PICSAR_version = "24.09"
answers = concat_answers(["y", PICSAR_version, PICSAR_version, "y"])

process = subprocess.Popen(
    [Path(REPO_DIR).joinpath("Tools/Release/updatePICSAR.py")],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

process.communicate(answers)
del process

# commit
subprocess.run(["git", "add", "-u"], capture_output=True, text=True)
subprocess.run(
    ["git", "commit", commit_sign, "-m", f"PICSAR: {PICSAR_version}"], text=True
)


# pyAMReX New Version #########################################################

pyAMReX_version = f"{datetime.now().strftime('%y')}.{datetime.now().strftime('%m')}"
answers = concat_answers(["y", pyAMReX_version, pyAMReX_version, "y"])

process = subprocess.Popen(
    [Path(REPO_DIR).joinpath("Tools/Release/updatepyAMReX.py")],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

process.communicate(answers)
del process

# commit
subprocess.run(["git", "add", "-u"], capture_output=True, text=True)
subprocess.run(
    ["git", "commit", commit_sign, "-m", f"pyAMReX: {pyAMReX_version}"], text=True
)


# WarpX New Version ###########################################################

answers = concat_answers(["y", WarpX_version_yr, WarpX_version_mn, "", "", "y"])

process = subprocess.Popen(
    [Path(REPO_DIR).joinpath("Tools/Release/newVersion.sh")],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

process.communicate(answers)
del process

# commit
subprocess.run(["git", "add", "-u"], capture_output=True, text=True)
subprocess.run(
    ["git", "commit", commit_sign, "-m", f"WarpX: {WarpX_version}"], text=True
)


# GitHub PR ###################################################################

subprocess.run(["git", "push", "-u", release_repo, release_branch], text=True)

subprocess.run(
    [
        "gh",
        "pr",
        "create",
        "--title",
        f"Release {WarpX_version}",
        "--body",
        f"""Prepare the {datetime.now().strftime("%B")} release of WarpX:
```bash
# update dependencies
./Tools/Release/updateAMReX.py
./Tools/Release/updatePICSAR.py  # no changes, still {PICSAR_version}
./Tools/Release/updatepyAMReX.py
# bump version number
./Tools/Release/newVersion.sh
```

Following this workflow: https://warpx.readthedocs.io/en/latest/maintenance/release.html
""",
        "--label",
        "component: documentation",
        "--label",
        "component: third party",
        "--web",
    ],
    text=True,
)


# Epilogue ####################################################################

print("""Done. Please check your source, e.g. via
  git diff
now and commit the changes if no errors occurred.""")
