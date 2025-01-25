#!/usr/bin/env python3
#
# Copyright 2025 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Axel Huebl
#

# This file is a maintainer tool to open a weekly dependency update PR for WarpX.
#
# You also need to have git and the GitHub CLI tool "gh" installed and properly
# configured for it to work:
# https://cli.github.com/
#
import subprocess
import sys
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

update_repo = input("What is the name of your git remote? (e.g., ax3l) ")
commit_sign = input("How to sign the commit? (e.g., -sS) ")


# Helpers #####################################################################


def concat_answers(answers):
    return "\n".join(answers) + "\n"


# Stash current work ##########################################################

subprocess.run(["git", "stash"], capture_output=True, text=True)


# Git Branch ##################################################################

update_branch = "topic-amrexWeekly"
subprocess.run(["git", "checkout", "development"], capture_output=True, text=True)
subprocess.run(["git", "fetch"], capture_output=True, text=True)
subprocess.run(["git", "pull", "--ff-only"], capture_output=True, text=True)
subprocess.run(["git", "branch", "-D", update_branch], capture_output=True, text=True)
subprocess.run(["git", "checkout", "-b", update_branch], capture_output=True, text=True)


# AMReX New Version ###########################################################

answers = concat_answers(["y", "", "", "y"])

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
amrex_diff = subprocess.run(["git", "diff", "--cached"], capture_output=True, text=True)
print("AMReX Commit...")
subprocess.run(
    ["git", "commit", commit_sign, "-m", "AMReX: Weekly Update"],
    capture_output=True,
    text=True,
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
picsar_diff = subprocess.run(
    ["git", "diff", "--cached"], capture_output=True, text=True
)
print("PICSAR Commit...")
subprocess.run(
    ["git", "commit", commit_sign, "-m", "PICSAR: Weekly Update"],
    capture_output=True,
    text=True,
)


# pyAMReX New Version #########################################################

answers = concat_answers(["y", "", "", "y"])

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
pyamrex_diff = subprocess.run(
    ["git", "diff", "--cached"], capture_output=True, text=True
)
print("pyAMReX Commit...")
subprocess.run(
    ["git", "commit", commit_sign, "-m", "pyAMReX: Weekly Update"],
    capture_output=True,
    text=True,
)

# GitHub PR ###################################################################

subprocess.run(["git", "push", "-f", "-u", update_repo, update_branch], text=True)

amrex_changes = " (no changes)" if amrex_diff.stdout == "" else ""
picsar_changes = " (no changes)" if picsar_diff.stdout == "" else ""
pyamrex_changes = " (no changes)" if pyamrex_diff.stdout == "" else ""

subprocess.run(
    [
        "gh",
        "pr",
        "create",
        "--title",
        "AMReX/pyAMReX/PICSAR: Weekly Update",
        "--body",
        f"""Weekly update to latest AMReX{amrex_changes}.
Weekly update to latest pyAMReX{pyamrex_changes}.
Weekly update to latest PICSAR{picsar_changes}.

```console
./Tools/Release/updateAMReX.py
./Tools/Release/updatepyAMReX.py
./Tools/Release/updatePICSAR.py
```
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
